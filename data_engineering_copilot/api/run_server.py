import uvicorn

if __name__ == "__main__":
    uvicorn.run("data_engineering_copilot.api.app:app", host="0.0.0.0", port=8000)