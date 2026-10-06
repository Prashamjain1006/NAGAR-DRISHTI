"""
Integration test suite for main.py FastAPI application.
"""

import asyncio
from main import app, seed_data
import httpx


async def test_platform_end_to_end():
    print("--- 1. Seeding Demo Incidents ---")
    await seed_data()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        print("--- 2. Health Endpoint ---")
        res_health = await client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        print("Health status:", health_data["status"])
        assert health_data["cached_incidents_count"] >= 20

        print("--- 3. Prioritized Incidents Endpoint ---")
        res_inc = await client.get("/api/v1/incidents/prioritized?limit=10")
        assert res_inc.status_code == 200
        data_inc = res_inc.json()
        assert data_inc["total"] >= 20
        incidents = data_inc["incidents"]
        top = incidents[0]
        print(f"Top Incident: {top['category']} at {top['address']} (PVI: {top['pvi_score']}, Cost: ${top['estimated_repair_cost']:.2f})")

        # Verify descending sort order
        for i in range(len(incidents) - 1):
            assert incidents[i]["pvi_score"] >= incidents[i + 1]["pvi_score"]

        # Filter by category
        res_cat = await client.get("/api/v1/incidents/prioritized?category=open_manhole")
        assert res_cat.status_code == 200
        for item in res_cat.json()["incidents"]:
            assert item["category"] == "open_manhole"
        print(f"Category filter verified: {len(res_cat.json()['incidents'])} open manholes found.")

        print("--- 4. Analytics Summary Endpoint ---")
        res_ana = await client.get("/api/v1/analytics/summary")
        assert res_ana.status_code == 200
        data_ana = res_ana.json()
        print(f"Total Defects: {data_ana['total_defects']}")
        print(f"Total Cost: ${data_ana['total_estimated_cost_usd']:,.2f}")
        print(f"Citizen Reports: {data_ana['reporting_sources']['citizen_reports']}")
        print(f"Automated Reports: {data_ana['reporting_sources']['automated_reports']}")
        print(f"Dedup Savings: ${data_ana['deduplication_savings']['estimated_cost_saved_usd']:,.2f}")
        print(f"Risk Levels: {data_ana['risk_distribution']}")
        assert data_ana["total_defects"] >= 20
        assert data_ana["total_estimated_cost_usd"] > 0

        print("--- 5. Status Transition Endpoint ---")
        first_id = top["id"]
        res_patch = await client.patch(f"/api/v1/incidents/{first_id}/status?new_status=in_progress")
        assert res_patch.status_code == 200
        patch_data = res_patch.json()
        assert patch_data["status"] == "in_progress"
        print(f"Status transition successful for incident {first_id} -> in_progress")

    print("\n>>> ALL TEST VERIFICATIONS COMPLETED SUCCESSFULLY! <<<")


if __name__ == "__main__":
    asyncio.run(test_platform_end_to_end())
