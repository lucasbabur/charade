"""Locust load test: /v1/rank with 100 candidates, plus impression events on 10 % of requests."""

import json
import os
import random
from pathlib import Path

from locust import FastHttpUser, constant_throughput, task

REQUESTS = [json.loads(line) for line in Path("artifacts/loadtest/requests.jsonl").read_text().splitlines()]


class Ranker(FastHttpUser):
    """One caller: rank, and sometimes report the served impression back."""

    # Open-ish loop: each simulated caller sends RPS_PER_USER requests/s, so latency is measured at a
    # stated load instead of at saturation (where it only measures queueing).
    wait_time = constant_throughput(float(os.environ.get("RPS_PER_USER", "25")))

    @task
    def rank(self) -> None:
        """POST /v1/rank."""
        body = random.choice(REQUESTS)
        with self.client.post("/v1/rank", json=body, name="/v1/rank", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"status {response.status_code}: {(response.text or '')[:120]}")
                return
            payload = response.json()
            chosen = payload["chosen_id"] if payload else None
        if chosen and random.random() < 0.1:
            ad = next(c for c in body["candidates"] if c["candidate_id"] == chosen)
            event = {k: body[k] for k in ("hour", "device_id", "device_ip", "device_model")}
            impression_id = f"{body['request_id']}-{random.getrandbits(32)}"
            self.client.post(
                "/v1/events/impression",
                json={
                    **event,
                    "impression_id": impression_id,
                    "request_id": body["request_id"],
                    "candidate_id": chosen,
                    "campaign_id": ad["C17"],
                },
                name="/v1/events/impression",
            )
            if random.random() < 0.18:
                self.client.post("/v1/events/click", json={"impression_id": impression_id}, name="/v1/events/click")
