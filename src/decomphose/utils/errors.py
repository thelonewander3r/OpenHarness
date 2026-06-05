class HarnessError(Exception):
    def __init__(
        self,
        message: str,
        code: str,
        status: int = 500,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status
        self.cause = cause


class StrategyError(HarnessError):
    def __init__(
        self,
        message: str,
        code: str,
        status: int = 502,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message, code, status, cause)


class DecompositionError(StrategyError):
    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message, "INVALID_DECOMPOSITION", 502, cause)


class MicroTaskError(HarnessError):
    def __init__(self, task_id: str, message: str, cause: BaseException | None = None) -> None:
        super().__init__(
            f"Micro-task {task_id} failed: {message}",
            "MICRO_TASK_FAILED",
            502,
            cause,
        )
        self.task_id = task_id
