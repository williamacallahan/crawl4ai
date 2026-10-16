from typing import TYPE_CHECKING, Union

# Crawler core types
AsyncWebCrawler = Union['AsyncWebCrawlerType']
CrawlResult = Union['CrawlResultType']

# Configuration types
CrawlerRunConfig = Union['CrawlerRunConfigType']
LLMConfig = Union['LLMConfigType']

# Dispatcher types
RunManyReturn = Union['RunManyReturnType']

# Only import types during type checking to avoid circular imports
if TYPE_CHECKING:
    # Crawler core imports
    from .async_webcrawler import AsyncWebCrawler as AsyncWebCrawlerType
    from .models import CrawlResult as CrawlResultType

    # Configuration imports
    from .async_configs import (
        CrawlerRunConfig as CrawlerRunConfigType,
        LLMConfig as LLMConfigType,
    )

    # Dispatcher imports
    from .async_dispatcher import RunManyReturn as RunManyReturnType


def create_llm_config(*args, **kwargs) -> 'LLMConfigType':
    from .async_configs import LLMConfig
    return LLMConfig(*args, **kwargs)
