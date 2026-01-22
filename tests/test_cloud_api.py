import requests

def test_cloud_service():
    cloud_url = "https://sns-mlops-api-593564032726.asia-east2.run.app"
    
    # Test prediction endpoint
    response = requests.post(
        f"{cloud_url}/predict",
        json={"text": "Everything is great."}
    )
    
    if response.status_code == 200:
        print("Cloud service validation passed.")
        print(f"Result: {response.json()}")
    else:
        print(f"Cloud service validation failed with status: {response.status_code}")

if __name__ == "__main__":
    test_cloud_service()