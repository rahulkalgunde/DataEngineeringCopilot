import uvicorn

from data_engineering_copilot.config.logging import setup_logging

if __name__ == "__main__":  # pragma: no cover
    setup_logging()
    uvicorn.run("data_engineering_copilot.api.app:app", host="0.0.0.0", port=8000)
