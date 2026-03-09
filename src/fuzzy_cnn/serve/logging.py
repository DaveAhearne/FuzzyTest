import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fuzzy_cnn.common.config import settings
from fuzzy_cnn.serve.context import request_id_ctx

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        return True

def configure_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s"
    )

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if settings.log_to_file:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
        log_dir.mkdir(exist_ok=True)
        handlers.append(
            RotatingFileHandler(log_dir / "app.log", maxBytes=10 * 1024 * 1024, backupCount=5)
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root_logger.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)