class RetryableError(Exception):
    """An error that may succeed if the operation is retried."""


class NonRetryableError(Exception):
    """An error that should fail immediately without retrying."""