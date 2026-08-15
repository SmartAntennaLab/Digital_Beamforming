import os
import unittest
import uuid

from distributed_coordination import RedisComputeCoordinator
from resource_policy import ResourcePolicy


@unittest.skipUnless(
    os.getenv("RUN_REDIS_INTEGRATION") == "1",
    "Set RUN_REDIS_INTEGRATION=1 with DBF_REDIS_URL for Redis integration.",
)
class RedisCoordinationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import redis

        cls.client = redis.Redis.from_url(
            os.environ["DBF_REDIS_URL"],
            decode_responses=False,
        )
        cls.prefix = f"dbf-test-{uuid.uuid4().hex}"

    @classmethod
    def tearDownClass(cls):
        for key in cls.client.scan_iter(match=f"{cls.prefix}:*"):
            cls.client.delete(key)
        cls.client.close()

    def coordinator(self) -> RedisComputeCoordinator:
        return RedisComputeCoordinator(
            self.client,
            ResourcePolicy(
                max_concurrent_calculations=1,
                global_max_concurrent_calculations=1,
                compute_queue_timeout_seconds=0.05,
                compute_timeout_seconds=2.0,
                session_calculations_per_minute=60,
                session_burst=1,
            ),
            key_prefix=self.prefix,
        )

    def test_two_replicas_share_rate_and_concurrency_state(self):
        first = self.coordinator()
        second = self.coordinator()

        self.assertTrue(first.consume_session_token("same-user").allowed)
        rate_rejected = second.consume_session_token("same-user")
        self.assertFalse(rate_rejected.allowed)
        self.assertGreater(rate_rejected.retry_after_seconds, 0.0)

        self.assertTrue(first.acquire_slot("replica-one"))
        self.assertFalse(second.acquire_slot("replica-two"))
        snapshot = second.snapshot()
        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.global_active_calculations, 1)

        first.release_slot("replica-one")
        self.assertTrue(second.acquire_slot("replica-two"))
        second.release_slot("replica-two")


if __name__ == "__main__":
    unittest.main()
