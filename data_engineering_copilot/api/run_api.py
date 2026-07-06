# File: data_engineering_copilot/api/run_api.py
from logger_config import setup_logging

import uvicorn
from data_engineering_copilot.api.app import app

if __name__ == "__main__":  # pragma: no cover
    setup_logging()
    uvicorn.run(app, host="0.0.0.0", port=8000)
