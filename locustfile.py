from locust import HttpUser, task, between

class SentimentApiUser(HttpUser):
    # Simulate random wait time between 1 to 2 seconds between requests
    wait_time = between(1, 2)

    @task(3)
    def predict(self):
        # Simulate high-frequency prediction requests
        self.client.post("/predict", json={
            "text": "The company shows strong growth in the second quarter."
        })

    @task(1)
    def health_check(self):
        # Simulate low-frequency health checks
        self.client.get("/health")