from threading import Lock

class WandbLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(WandbLogger, cls).__new__(cls)
                    cls._instance.log_data = {}
        return cls._instance

wandb_logger = WandbLogger()
