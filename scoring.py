"""
Dynamic Infrastructure Defect Prioritization & Cost Estimation Engine.
Calculates Priority Value Index (PVI) and estimates repair costs in USD:

PVI Formula:
PVI = (Severity * 0.30) + (Road_Hierarchy * 0.20) + (Traffic_Density * 0.20) + 
      (Population_Impact * 0.15) + (Verification_Count * 0.15)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# -----------------------------------------------------------------------------
# Hierarchy & Constant Dictionaries
# -----------------------------------------------------------------------------
ROAD_HIERARCHY_SCORES = {
    "expressway": 10.0,
    "highway": 10.0,
    "freeway": 10.0,
    "arterial": 7.0,
    "collector": 5.0,
    "residential": 3.0,
    "local": 3.0,
    "alley": 2.0,
}

# Baseline category unit repair cost parameters in USD [min_cost, max_cost, baseline_severity]
REPAIR_COST_MODELS = {
    "pothole": {
        "min_usd": 150.0,
        "max_usd": 800.0,
        "base_severity": 5.0,
        "unit": "asphalt patch (sq meters)",
    },
    "open_manhole": {
        "min_usd": 400.0,
        "max_usd": 1500.0,
        "base_severity": 9.0,
        "unit": "casting ring & ductile iron cover replacement",
    },
    "broken_sign": {
        "min_usd": 180.0,
        "max_usd": 650.0,
        "base_severity": 4.0,
        "unit": "retroreflective panel & galvanized pole hardware",
    },
    "waterlogging": {
        "min_usd": 500.0,
        "max_usd": 3800.0,
        "base_severity": 7.5,
        "unit": "vacuum pump discharge & storm drain culvert clearing",
    },
    "streetlight": {
        "min_usd": 200.0,
        "max_usd": 850.0,
        "base_severity": 4.5,
        "unit": "LED luminaire driver & photocell fixture",
    },
}


@dataclass
class ScoringResult:
    """Structured breakdown and outcome of dynamic infrastructure assessment."""
    pvi_score: float
    risk_level: str
    estimated_cost: float
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Component Calculation Functions
# -----------------------------------------------------------------------------
def calculate_severity(
    category: str,
    bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    image_resolution: Tuple[int, int] = (640, 640),
    confidence: Optional[float] = None,
    explicit_severity: Optional[float] = None,
) -> float:
    """
    Computes Severity (1.0 to 10.0):
    1. If explicit_severity provided, clamps and returns.
    2. Otherwise calculates from defect category baseline and bounding box pixel volume percentage.
    """
    if explicit_severity is not None:
        return max(1.0, min(10.0, round(float(explicit_severity), 1)))

    cat_key = category.lower().replace(" ", "_")
    cat_spec = REPAIR_COST_MODELS.get(cat_key, {"base_severity": 5.0})
    base_severity = cat_spec["base_severity"]

    # Calculate area ratio if bounding boxes exist
    area_boost = 0.0
    if bounding_boxes and len(bounding_boxes) > 0:
        total_defect_area = 0.0
        for b in bounding_boxes:
            w = float(b.get("width", 0.0))
            h = float(b.get("height", 0.0))
            total_defect_area += w * h

        img_w, img_h = image_resolution
        total_img_area = max(1.0, float(img_w * img_h))
        area_pct = min(1.0, total_defect_area / total_img_area)

        # Large defect coverage (>20% of frame) scales severity up by up to +2.5
        area_boost = min(2.5, area_pct * 12.0)

    # Confidence scaling
    conf_factor = float(confidence) if confidence is not None else 0.85
    score = (base_severity * 0.75) + (base_severity * conf_factor * 0.25) + area_boost
    return max(1.0, min(10.0, round(score, 1)))


def calculate_road_hierarchy(road_type: Union[str, float, int]) -> float:
    """
    Computes Road_Hierarchy score (1.0 to 10.0):
    - Expressways / Highways = 10.0
    - Arterials = 7.0
    - Collectors = 5.0
    - Residential / Local = 3.0
    """
    if isinstance(road_type, (int, float)):
        return max(1.0, min(10.0, round(float(road_type), 1)))

    return ROAD_HIERARCHY_SCORES.get(str(road_type).lower().strip(), 3.0)


def calculate_traffic_density(
    traffic_density: Optional[Union[float, int, str]] = None,
    road_type: Optional[str] = None,
) -> float:
    """
    Computes Traffic_Density (1.0 to 10.0):
    Scaled from traffic API telemetry or inferred from road classification.
    """
    if traffic_density is not None:
        try:
            val = float(traffic_density)
            return max(1.0, min(10.0, round(val, 1)))
        except (ValueError, TypeError):
            pass

    # Inferred default based on road type
    defaults = {
        "expressway": 8.5,
        "highway": 8.0,
        "arterial": 7.0,
        "collector": 5.0,
        "residential": 3.0,
        "local": 3.0,
    }
    return defaults.get(str(road_type).lower().strip() if road_type else "local", 5.0)


def calculate_population_impact(
    proximity_meters_to_poi: Optional[float] = None,
    poi_type: Optional[str] = None,
    explicit_population_score: Optional[float] = None,
) -> float:
    """
    Computes Population_Impact score (1.0 to 10.0):
    Evaluates proximity to sensitive civic hubs (schools, hospitals, transit hubs).
    - Within 50m of hospital/school -> 9.0 to 10.0
    - Within 200m -> 7.0 to 8.5
    - Within 500m -> 5.0 to 6.5
    - General residential -> 3.0 to 4.0
    """
    if explicit_population_score is not None:
        return max(1.0, min(10.0, round(float(explicit_population_score), 1)))

    if proximity_meters_to_poi is not None:
        dist = max(0.0, float(proximity_meters_to_poi))
        poi_multiplier = 1.2 if str(poi_type).lower() in ("hospital", "school", "transit_hub", "metro") else 1.0

        if dist <= 50.0:
            score = 10.0 * poi_multiplier
        elif dist <= 200.0:
            score = 8.0 * poi_multiplier
        elif dist <= 500.0:
            score = 6.0 * poi_multiplier
        elif dist <= 1000.0:
            score = 4.0
        else:
            score = 2.0
        return max(1.0, min(10.0, round(score, 1)))

    # Baseline civic population impact
    return 5.0


def calculate_verification_boost(verification_count: int) -> float:
    """
    Computes Verification_Count component (1.0 to 10.0):
    Logarithmic boost based on total citizens/cameras reporting the same issue:
    Formula: min(10.0, log2(verification_count + 1) * 3.0)
    
    Examples:
    - 1 verification:  log2(2) * 3 = 3.00
    - 2 verifications: log2(3) * 3 = 4.75
    - 3 verifications: log2(4) * 3 = 6.00
    - 7 verifications: log2(8) * 3 = 9.00
    - 10+ verifications: 10.00
    """
    count = max(1, int(verification_count))
    boost = math.log2(count + 1) * 3.0
    return max(1.0, min(10.0, round(boost, 2)))


# -----------------------------------------------------------------------------
# Repair Cost Estimation
# -----------------------------------------------------------------------------
def estimate_repair_cost(
    category: str,
    severity: float,
    bounding_box: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Calculates estimated infrastructure remediation cost in USD based on defect
    type, severity score (1-10), and bounding box area dimensions.
    
    Examples:
    - Pothole: $150 to $800 based on severity and asphalt fill volume.
    - Open Manhole: $400 to $1,500 based on ductile iron casting & safety barrier.
    - Waterlogging: $500 to $3,800 based on hydro-excavation & culvert routing.
    """
    cat_key = category.lower().replace(" ", "_")
    spec = REPAIR_COST_MODELS.get(cat_key, REPAIR_COST_MODELS["pothole"])

    min_usd = spec["min_usd"]
    max_usd = spec["max_usd"]

    # Severity factor: 1.0 -> 0.0, 10.0 -> 1.0
    sev_factor = max(0.0, min(1.0, (severity - 1.0) / 9.0))

    # Optional box volume factor if dimensional data is provided
    box_factor = 1.0
    if bounding_box and bounding_box.get("width") and bounding_box.get("height"):
        w = float(bounding_box["width"])
        h = float(bounding_box["height"])
        area = w * h
        # Normalized against typical 640x640 frame
        box_factor = min(1.3, max(0.8, (area / (300.0 * 200.0)) ** 0.5))

    cost = (min_usd + (max_usd - min_usd) * sev_factor) * box_factor
    return round(cost, 2)


# -----------------------------------------------------------------------------
# Main Dynamic Scoring Engine
# -----------------------------------------------------------------------------
def calculate_dynamic_pvi(
    category: str,
    severity: Optional[float] = None,
    bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    road_type: Union[str, float] = "local",
    traffic_density: Optional[float] = None,
    population_impact: Optional[float] = None,
    proximity_meters_to_poi: Optional[float] = None,
    poi_type: Optional[str] = None,
    verification_count: int = 1,
) -> ScoringResult:
    """
    Calculates the dynamic Priority Value Index (PVI) and ranks infrastructure defects:
    
    PVI = (Severity * 0.30) + (Road_Hierarchy * 0.20) + (Traffic_Density * 0.20) + 
          (Population_Impact * 0.15) + (Verification_Count * 0.15)
    
    Returns:
        ScoringResult: {
            pvi_score: float (0.0 to 10.0),
            risk_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW',
            estimated_cost: float (USD),
            components: dict
        }
    """
    # 1. Compute Individual Component Scores (1.0 to 10.0)
    sev_score = calculate_severity(
        category=category,
        bounding_boxes=bounding_boxes,
        explicit_severity=severity,
    )
    road_score = calculate_road_hierarchy(road_type=road_type)
    traffic_score = calculate_traffic_density(traffic_density=traffic_density, road_type=str(road_type))
    pop_score = calculate_population_impact(
        proximity_meters_to_poi=proximity_meters_to_poi,
        poi_type=poi_type,
        explicit_population_score=population_impact,
    )
    verif_score = calculate_verification_boost(verification_count=verification_count)

    # 2. Weighted PVI Computation
    pvi = (
        (sev_score * 0.30)
        + (road_score * 0.20)
        + (traffic_score * 0.20)
        + (pop_score * 0.15)
        + (verif_score * 0.15)
    )
    pvi_clamped = max(0.0, min(10.0, round(pvi, 2)))

    # 3. Categorize Risk Level
    if pvi_clamped >= 8.0:
        risk_level = "CRITICAL"
    elif pvi_clamped >= 6.5:
        risk_level = "HIGH"
    elif pvi_clamped >= 4.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 4. Calculate Repair Cost
    primary_box = bounding_boxes[0] if bounding_boxes else None
    estimated_cost = estimate_repair_cost(
        category=category,
        severity=sev_score,
        bounding_box=primary_box,
    )

    components = {
        "severity": sev_score,
        "road_hierarchy": road_score,
        "traffic_density": traffic_score,
        "population_impact": pop_score,
        "verification_boost": verif_score,
    }

    return ScoringResult(
        pvi_score=pvi_clamped,
        risk_level=risk_level,
        estimated_cost=estimated_cost,
        components=components,
    )


# -----------------------------------------------------------------------------
# Standalone CLI Verification
# -----------------------------------------------------------------------------
def main():
    print("=== Dynamic Infrastructure Scoring Engine Tests ===\n")

    # Test Case 1: Low-priority residential pothole (1 report)
    res_low = calculate_dynamic_pvi(
        category="pothole",
        severity=3.5,
        road_type="local",
        traffic_density=2.0,
        population_impact=2.0,
        verification_count=1,
    )
    print("1. Low Priority Case (Residential Pothole):")
    print(f"   PVI: {res_low.pvi_score} | Risk: {res_low.risk_level} | Cost: ${res_low.estimated_cost:.2f}")
    assert res_low.risk_level in ("LOW", "MEDIUM")

    # Test Case 2: Critical open manhole on Expressway with 5 reports
    res_crit = calculate_dynamic_pvi(
        category="open_manhole",
        severity=9.5,
        road_type="expressway",
        traffic_density=9.0,
        population_impact=8.5,
        verification_count=5,
    )
    print("\n2. Critical Case (Expressway Open Manhole with 5 reports):")
    print(f"   PVI: {res_crit.pvi_score} | Risk: {res_crit.risk_level} | Cost: ${res_crit.estimated_cost:.2f}")
    assert res_crit.risk_level == "CRITICAL"
    assert res_crit.pvi_score >= 8.0

    # Test Case 3: Waterlogging near school (high population impact)
    res_water = calculate_dynamic_pvi(
        category="waterlogging",
        road_type="arterial",
        traffic_density=7.0,
        proximity_meters_to_poi=35.0,
        poi_type="school",
        verification_count=3,
    )
    print("\n3. Arterial Waterlogging near School:")
    print(f"   PVI: {res_water.pvi_score} | Risk: {res_water.risk_level} | Cost: ${res_water.estimated_cost:.2f}")
    assert res_water.risk_level in ("HIGH", "CRITICAL")

    print("\nComponent Breakdown for Case 3:")
    for k, v in res_water.components.items():
        print(f"   - {k}: {v}")

    print("\nALL DYNAMIC SCORING TESTS PASSED!")


if __name__ == "__main__":
    main()
