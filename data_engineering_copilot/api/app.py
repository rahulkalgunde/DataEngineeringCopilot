from fastapi import FastAPI
from .routes import router

app = FastAPI(
    title="DataEngineeringCopilot API",
    description="Async ingestion and RAG service endpoints",
    version="1.0.0",
)

app.include_router(router)