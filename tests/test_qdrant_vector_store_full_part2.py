import pytest  
from unittest.mock import Mock, patch  
from data_engineering_copilot.infrastructure.qdrant_store import QdrantVectorStore  
from data_engineering_copilot.domain.models import DocumentChunk, RetrievedChunk  
 
