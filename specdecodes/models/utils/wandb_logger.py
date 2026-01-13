from threading import Lock
from typing import Any, Dict

class WandbLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(WandbLogger, cls).__new__(cls)
                    cls._instance.log_data = {}
                    cls._instance.flags = {}
                    cls._instance.internal_data = {}
        return cls._instance

    def set_flag(self, key: str, value: Any) -> None:
        self.flags[key] = value

    def set_flags(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.flags[key] = value

    def get_flag(self, key: str, default: Any = None) -> Any:
        return self.flags.get(key, default)

    def clear_log_data(self) -> None:
        self.log_data.clear()
        # Keep internal running stats in sync with public log data.
        self.internal_data.clear()

    def clear_flags(self) -> None:
        self.flags.clear()

    def clear_internal_data(self) -> None:
        self.internal_data.clear()

wandb_logger = WandbLogger()
