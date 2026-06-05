from fastapi.testclient import TestClient
import json


def run_tests():
    failures = []
    try:
        from app.main import app
    except Exception as e:
        print("IMPORT_ERROR", repr(e))
        raise

    client = TestClient(app)

    def safe_get(path, **kwargs):
        try:
            r = client.get(path, **kwargs)
            print(path, r.status_code)
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception:
                print(r.text)
            return r
        except Exception as e:
            print(path, "EXC", repr(e))
            failures.append((path, repr(e)))

    def safe_post(path, payload):
        try:
            r = client.post(path, json=payload)
            print(path, r.status_code)
            try:
                print(json.dumps(r.json(), indent=2))
            except Exception:
                print(r.text)
            return r
        except Exception as e:
            print(path, "EXC", repr(e))
            failures.append((path, repr(e)))

    # Root
    safe_get("/")

    # API health
    safe_get("/api/health")

    # Store-based prediction endpoint
    payload = {
        "company": "TestCo",
        "status": "Operating",
        "year_founded": 2020,
        "description": "Test startup",
        "categories": ["SaaS"],
        "founders": ["Alice"],
        "investors": ["Acme VC"],
        "funding_rounds": [],
        "city": "San Francisco",
        "state": "CA",
        "country": "USA",
    }
    safe_post("/api/predict/success", payload)

    # Competitor endpoint (may fail if clustering model missing)
    safe_get("/competitors", params={"company": "Google", "top_n": 3})

    # ANN predict (may be unmounted)
    safe_post("/ann/predict", {"company": "TestCo"})

    if failures:
        print("\nFailures detected:\n", failures)
        raise SystemExit(2)
    else:
        print("\nAll tested endpoints returned responses.")


if __name__ == "__main__":
    run_tests()
