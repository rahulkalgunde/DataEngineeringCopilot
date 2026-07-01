import pytest
from unittest.mock import AsyncMock, patch
from data_engineering_copilot.infrastructure.crawl4ai_wrapper import Crawl4AIWrapper
from data_engineering_copilot.domain.models import RawDocument

class TestCrawl4AIWrapper:
    @pytest.fixture
    def mock_async_web_crawler(self):
        with patch("data_engineering_copilot.infrastructure.crawl4ai_wrapper.AsyncWebCrawler") as mock_crawler_class:
            mock_crawler_instance = AsyncMock()
            mock_crawler_class.return_value.__aenter__.return_value = mock_crawler_instance
            yield mock_crawler_instance

    def test_crawl_success(self, mock_async_web_crawler):
        mock_result = AsyncMock()
        mock_result.success = True
        mock_result.url = "http://example.com"
        mock_result.html = "<html><body>Test content</body></html>"
        
        mock_async_web_crawler.arun.return_value = mock_result
        
        wrapper = Crawl4AIWrapper(timeout_seconds=5, delay_seconds=0.1)
        documents = wrapper.crawl(["http://example.com"])
        
        assert len(documents) == 1
        assert isinstance(documents[0], RawDocument)
        assert documents[0].source_name == "Crawl4AI"
        assert documents[0].url == "http://example.com"
        assert documents[0].html == "<html><body>Test content</body></html>"
        
        mock_async_web_crawler.arun.assert_called_once_with(url="http://example.com")

    def test_crawl_failure(self, mock_async_web_crawler):
        mock_result = AsyncMock()
        mock_result.success = False
        
        mock_async_web_crawler.arun.return_value = mock_result
        
        wrapper = Crawl4AIWrapper(timeout_seconds=5, delay_seconds=0.1)
        documents = wrapper.crawl(["http://example.com"])
        
        assert len(documents) == 0
        mock_async_web_crawler.arun.assert_called_once_with(url="http://example.com")
