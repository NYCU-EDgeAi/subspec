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

    def clear_flags(self) -> None:
        self.flags.clear()

wandb_logger = WandbLogger()
