from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLOUD_TESTS") != "1",
    reason="Cloud API integration test is disabled by default. Set RUN_CLOUD_TESTS=1 to enable.",
)


def test_cloud_service():
    import httpx

    cloud_url = "https://sns-mlops-api-593564032726.asia-east2.run.app"

    response = httpx.post(f"{cloud_url}/predict", json={"text": "Everything is great."}, timeout=10.0)

    if response.status_code == 200:
        print("Cloud service validation passed.")
        print(f"Result: {response.json()}")
    else:
        print(f"Cloud service validation failed with status: {response.status_code}")


if __name__ == "__main__":
    test_cloud_service()
