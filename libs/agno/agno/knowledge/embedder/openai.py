from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Literal

from agno.knowledge.embedder.base import Embedder
from agno.utils.log import logger

try:
    from openai import AsyncOpenAI
    from openai import OpenAI as OpenAIClient
    from openai.types.create_embedding_response import CreateEmbeddingResponse
except ImportError:
    raise ImportError("`openai` not installed")


@dataclass
class OpenAIEmbedder(Embedder):
    id: str = "text-embedding-3-small"
    dimensions: int = 1536
    encoding_format: Literal["float", "base64"] = "float"
    user: Optional[str] = None
    api_key: Optional[str] = None
    organization: Optional[str] = None
    base_url: Optional[str] = None
    request_params: Optional[Dict[str, Any]] = None
    client_params: Optional[Dict[str, Any]] = None
    openai_client: Optional[OpenAIClient] = None
    async_client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> OpenAIClient:
        if self.openai_client:
            return self.openai_client

        _client_params: Dict[str, Any] = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
        }
        _client_params = {k: v for k, v in _client_params.items() if v is not None}
        if self.client_params:
            _client_params.update(self.client_params)
        self.openai_client = OpenAIClient(**_client_params)
        return self.openai_client

    @property
    def aclient(self) -> AsyncOpenAI:
        if self.async_client:
            return self.async_client
        params = {
            "api_key": self.api_key,
            "organization": self.organization,
            "base_url": self.base_url,
        }
        params = {k: v for k, v in params.items() if v is not None}
        if self.client_params:
            params.update(self.client_params)
        self.async_client = AsyncOpenAI(**params)
        return self.async_client

    # @property
    # def aclient(self) -> AsyncOpenAI:
    #     if self.async_client:
    #         return self.async_client

    #     params = {
    #         "api_key": self.api_key,
    #         "organization": self.organization,
    #         "base_url": self.base_url,
    #     }
    #     params = {k: v for k, v in params.items() if v is not None}
    #     if self.client_params:
    #         params.update(self.client_params)

    #     # Add connection limits and timeout settings
    #     import httpx
    #     limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    #     timeout = httpx.Timeout(60.0, connect=5.0)

    #     # Create HTTP client with proper settings
    #     http_client = httpx.AsyncClient(
    #         limits=limits,
    #         timeout=timeout,
    #         # Important: Set follow_redirects and other settings
    #         follow_redirects=True
    #     )
    #     params["http_client"] = http_client

    #     self.async_client = AsyncOpenAI(**params)
    #     return self.async_client

    def response(self, text: str) -> CreateEmbeddingResponse:
        _request_params: Dict[str, Any] = {
            "input": text,
            "model": self.id,
            "encoding_format": self.encoding_format,
        }
        if self.user is not None:
            _request_params["user"] = self.user
        if self.id.startswith("text-embedding-3"):
            _request_params["dimensions"] = self.dimensions
        if self.request_params:
            _request_params.update(self.request_params)
        return self.client.embeddings.create(**_request_params)

    def get_embedding(self, text: str) -> List[float]:
        response: CreateEmbeddingResponse = self.response(text=text)
        try:
            return response.data[0].embedding
        except Exception as e:
            logger.warning(e)
            return []

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        response: CreateEmbeddingResponse = self.response(text=text)

        embedding = response.data[0].embedding
        usage = response.usage
        if usage:
            return embedding, usage.model_dump()
        return embedding, None

    async def async_get_embedding(self, text: str) -> List[float]:
        req = {
            "input": text,
            "model": self.id,
            "encoding_format": self.encoding_format,
        }
        if self.user is not None:
            req["user"] = self.user
        if self.id.startswith("text-embedding-3"):
            req["dimensions"] = self.dimensions
        if self.request_params:
            req.update(self.request_params)

        try:
            response: CreateEmbeddingResponse = await self.aclient.embeddings.create(**req)
            return response.data[0].embedding
        except Exception as e:
            logger.warning(e)
            return []

    async def async_get_embedding_and_usage(self, text: str):
        req = {
            "input": text,
            "model": self.id,
            "encoding_format": self.encoding_format,
        }
        if self.user is not None:
            req["user"] = self.user
        if self.id.startswith("text-embedding-3"):
            req["dimensions"] = self.dimensions
        if self.request_params:
            req.update(self.request_params)

        response = await self.aclient.embeddings.create(**req)
        embedding = response.data[0].embedding
        usage = response.usage
        return embedding, usage.model_dump() if usage else None

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batches.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each API call (max ~2048)

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            req = {
                "input": batch_texts,
                "model": self.id,
                "encoding_format": self.encoding_format,
            }
            if self.user is not None:
                req["user"] = self.user
            if self.id.startswith("text-embedding-3"):
                req["dimensions"] = self.dimensions
            if self.request_params:
                req.update(self.request_params)

            try:
                response: CreateEmbeddingResponse = self.client.embeddings.create(**req)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Error in batch embedding: {e}")
                # Fallback to individual calls for this batch
                for text in batch_texts:
                    try:
                        embedding = self.get_embedding(text)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.warning(f"Error in individual embedding fallback: {e2}")
                        all_embeddings.append([])

        return all_embeddings

    async def async_get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batches (async version).

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each API call (max ~2048)

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            req = {
                "input": batch_texts,
                "model": self.id,
                "encoding_format": self.encoding_format,
            }
            if self.user is not None:
                req["user"] = self.user
            if self.id.startswith("text-embedding-3"):
                req["dimensions"] = self.dimensions
            if self.request_params:
                req.update(self.request_params)

            try:
                response: CreateEmbeddingResponse = await self.aclient.embeddings.create(**req)
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.warning(f"Error in async batch embedding: {e}")
                # Fallback to individual async calls for this batch
                for text in batch_texts:
                    try:
                        embedding = await self.async_get_embedding(text)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        logger.warning(f"Error in individual async embedding fallback: {e2}")
                        all_embeddings.append([])

        return all_embeddings
