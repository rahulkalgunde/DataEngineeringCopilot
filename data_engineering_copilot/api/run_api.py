# File: data_engineering_copilot/api/run_api.py
import uvicorn
from data_engineering_copilot.api.routes import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
