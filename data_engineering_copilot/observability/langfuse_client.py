import os
from langfuse import Langfuse
import logging

logger = logging.getLogger(__name__)

def get_langfuse_instance():
    """
    Create a Langfuse client using environment variables.
    FUTURE_ARCHITECTURE.md defines the required env vars in the docker‑compose file.
    """
    try:
        lf = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
        )
        return lf
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None

# Helper functions for tracing
def start_trace(name: str, **kwargs):
    lf = get_langfuse_instance()
    if lf:
        return lf.trace(name=name, **kwargs)
    return None

def end_trace(trace, **kwargs):
    if trace:
        trace.end(**kwargs)

def log_event(trace, name: str, **kwargs):
    if trace:
        span = trace.span(name=name, **kwargs)
        span.end()