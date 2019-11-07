class InitializationError(Exception):
    """
    The exception raised when a initialization is misconfigured.
    """

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)
