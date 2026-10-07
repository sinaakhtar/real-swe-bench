"""Webhook delivery retry and backoff scheduling."""


def compute_retry_delay(
    attempt: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    max_attempts: int = 5,
    retry_after_header: float | None = None,
) -> float:
    """Compute the next retry delay in seconds for a failed webhook delivery.

    Rules:
    1. `attempt` is 0-indexed (0 is the first retry). If `attempt >= max_attempts` or `attempt < 0`,
       return -1.0 to indicate retries are exhausted.
    2. Standard exponential backoff is `base_delay * (2 ** attempt)`.
    3. If `retry_after_header` is provided and positive, the delay before capping must be
       `max(exponential_delay, retry_after_header)`.
    4. Cap the returned delay at `max_delay`.
    """
    # BUG (PR-106): Does not check max_attempts, ignores retry_after_header, and forgets max_delay cap.
    return float(base_delay * (2 ** attempt))
