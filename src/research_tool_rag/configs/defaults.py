import os

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(process)s:%(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {"handlers": ["console"], "level": ("DEBUG" if os.getenv("DEBUG") else "INFO")},
        "wklri": {
            "handlers": ["console"],
            "level": ("DEBUG" if os.getenv("DEBUG") else "INFO"),
            "propagate": False,
        },
    },
}

OFFLINE_MODEL = {"model_name": "mistral", "model_provider": "google_genai"}

ONLINE_MODEL = {"model_name": "gemini-2.0-flash"}
