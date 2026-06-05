"""
Comprehensive Integration Test
===============================
Tests all backend modules working together:
- FastAPI main app
- API CRUD and health endpoints
- Store-based prediction (heuristic)
- Competitor analysis with clustering
- Features (success prediction)
"""
import json
from fastapi.testclient import TestClient


def run_integration_tests():
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKEND INTEGRATION TEST")
    print("="*80 + "\n")

    # Import the main app
    try:
        from app.main import app
        print("✓ Successfully imported app.main")
    except Exception as e:
        print(f"✗ Failed to import app.main: {e}")
        raise

    client = TestClient(app)
    test_results = {
        "passed": [],
        "failed": [],
        "warnings": [],
    }

    def test_endpoint(name, method, path, payload=None, expected_status=200, description=""):
        """Helper to test an endpoint and track results."""
        try:
            if method == "GET":
                response = client.get(path)
            elif method == "POST":
                response = client.post(path, json=payload)
            elif method == "PUT":
                response = client.put(path, json=payload)
            elif method == "DELETE":
                response = client.delete(path)
            else:
                raise ValueError(f"Unsupported method: {method}")

            status = response.status_code
            success = status == expected_status or (expected_status == 200 and 200 <= status < 300)

            print(f"\n[TEST] {name}")
            print(f"  {method} {path}")
            if payload:
                print(f"  Payload: {json.dumps(payload, indent=4)[:200]}...")
            print(f"  Status: {status} (expected {expected_status})")

            try:
                data = response.json()
                print(f"  Response: {json.dumps(data, indent=4)[:300]}...")
            except Exception:
                print(f"  Response: {response.text[:300]}...")

            if success:
                test_results["passed"].append(name)
                print(f"  ✓ PASS")
            else:
                test_results["failed"].append((name, f"Status {status}, expected {expected_status}"))
                print(f"  ✗ FAIL - Status mismatch")

            return response
        except Exception as e:
            test_results["failed"].append((name, str(e)))
            print(f"\n[TEST] {name}")
            print(f"  ✗ FAIL - Exception: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 1: System & Health
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("TIER 1: System Health & Root Endpoints")
    print("─"*80)

    test_endpoint(
        "Root endpoint",
        "GET", "/",
        description="Verify API is running"
    )

    test_endpoint(
        "API health check",
        "GET", "/api/health",
        description="Verify backend has loaded startup data"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 2: Startup CRUD Operations
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("TIER 2: Startup CRUD Operations (via Store)")
    print("─"*80)

    test_endpoint(
        "List all startups (paginated)",
        "GET", "/api/startups?limit=5&offset=0",
        description="Retrieve paginated startup list"
    )

    test_endpoint(
        "Filter startups by status",
        "GET", "/api/startups?status=Operating&limit=5",
        description="Filter startups by operating status"
    )

    test_endpoint(
        "Filter startups by country",
        "GET", "/api/startups?country=USA&limit=5",
        description="Filter startups by headquarters country"
    )

    # Create a test startup
    new_startup = {
        "company": "TestAI Corp",
        "status": "Operating",
        "year_founded": 2022,
        "description": "AI-powered analytics platform for startups",
        "categories": ["Artificial Intelligence", "Analytics"],
        "founders": ["Alice Chen", "Bob Smith"],
        "investors": ["Sequoia Capital"],
        "funding_rounds": ["$2M Seed"],
        "city": "San Francisco",
        "state": "CA",
        "country": "USA",
    }

    create_resp = test_endpoint(
        "Create new startup",
        "POST", "/api/startups",
        payload=new_startup,
        expected_status=201,
        description="Add a new startup to the store"
    )

    startup_id = None
    if create_resp and create_resp.status_code == 201:
        startup_id = create_resp.json().get("startup_id")
        print(f"  Created startup ID: {startup_id}")

    if startup_id:
        test_endpoint(
            "Get startup by ID",
            "GET", f"/api/startups/{startup_id}",
            description="Retrieve the created startup"
        )

        # Update the startup
        update_payload = {
            "description": "Updated: AI-powered analytics for startup success prediction",
        }
        test_endpoint(
            "Update startup",
            "PUT", f"/api/startups/{startup_id}",
            payload=update_payload,
            description="Modify startup details"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 3: Prediction Engine (Heuristic-based)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("TIER 3: Success Prediction (Store-based Heuristic)")
    print("─"*80)

    prediction_payload_1 = {
        "company": "SuccessfulAI",
        "status": "Operating",
        "year_founded": 2018,
        "description": "Leading AI analytics startup with strong backing",
        "categories": ["Artificial Intelligence", "SaaS"],
        "founders": ["Dr. Jane Doe", "John Smith"],
        "investors": ["Andreessen Horowitz", "Sequoia Capital"],
        "funding_rounds": ["$5M Seed", "$20M Series A"],
        "city": "San Francisco",
        "state": "CA",
        "country": "USA",
    }

    test_endpoint(
        "Predict success (high-confidence startup)",
        "POST", "/api/predict/success",
        payload=prediction_payload_1,
        description="Predict success for well-funded operating startup"
    )

    prediction_payload_2 = {
        "company": "UnknownStartup",
        "status": "Dead",
        "year_founded": 2005,
        "description": "Inactive startup",
        "categories": [],
        "founders": [],
        "investors": [],
        "funding_rounds": [],
        "city": None,
        "state": None,
        "country": None,
    }

    test_endpoint(
        "Predict success (low-confidence startup)",
        "POST", "/api/predict/success",
        payload=prediction_payload_2,
        description="Predict success for inactive/dead startup"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 4: Competitor Analysis (with Clustering)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("TIER 4: Competitor Analysis (Clustering-based)")
    print("─"*80)

    # First, get a real startup from the dataset to query
    startups_resp = client.get("/api/startups?limit=1&offset=0")
    real_company = None
    if startups_resp.status_code == 200:
        items = startups_resp.json().get("items", [])
        if items:
            real_company = items[0]["company"]
            print(f"\n  Using real dataset company: {real_company}")

    if real_company:
        test_endpoint(
            "Find competitors by company name",
            "GET", f"/competitors?company={real_company}&top_n=5",
            description="Retrieve similar startups from same cluster"
        )

    # Test competitor analysis by features (for new/unknown startup)
    competitor_by_features = {
        "company_name": "NewTechStartup",
        "categories": "Artificial Intelligence, Machine Learning",
        "year_founded": 2023,
        "total_funding": 1500000,
        "headquarters_city": "San Francisco",
        "headquarters_country": "United States",
        "top_n": 5,
    }

    test_endpoint(
        "Find competitors by features (new startup)",
        "POST", "/competitors/by-features",
        payload=competitor_by_features,
        description="Analyze competitors for a startup not in dataset"
    )

    test_endpoint(
        "Get cluster summary statistics",
        "GET", "/competitors/clusters/summary",
        description="View segmentation and cluster insights"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TIER 5: Cleanup & Final Results
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "─"*80)
    print("TIER 5: Cleanup")
    print("─"*80)

    if startup_id:
        test_endpoint(
            "Delete created startup",
            "DELETE", f"/api/startups/{startup_id}",
            expected_status=204,
            description="Remove test startup"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Summary Report
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✓ Passed: {len(test_results['passed'])}")
    for test_name in test_results["passed"]:
        print(f"  • {test_name}")

    if test_results["failed"]:
        print(f"\n✗ Failed: {len(test_results['failed'])}")
        for test_name, reason in test_results["failed"]:
            print(f"  • {test_name}: {reason}")

    if test_results["warnings"]:
        print(f"\n⚠ Warnings: {len(test_results['warnings'])}")
        for warning in test_results["warnings"]:
            print(f"  • {warning}")

    total = len(test_results["passed"]) + len(test_results["failed"])
    success_rate = (len(test_results["passed"]) / total * 100) if total > 0 else 0

    print(f"\nSuccess Rate: {success_rate:.1f}% ({len(test_results['passed'])}/{total})")

    if test_results["failed"]:
        print("\n❌ INTEGRATION TEST FAILED")
        return False
    else:
        print("\n✅ INTEGRATION TEST PASSED - All modules working together!")
        return True


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
