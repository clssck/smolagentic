"""Reranker integration for enhanced RAG retrieval using cross-encoder models.

This module provides reranking capabilities to improve retrieval accuracy
by reordering search results based on query-document relevance scores.
"""

from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import reranking libraries
try:
    from rerankers import Reranker
    RERANKERS_AVAILABLE = True
    logger.info("Rerankers library available")
except ImportError:
    RERANKERS_AVAILABLE = False
    logger.warning("Rerankers library not available. Install with: pip install rerankers")

try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.info("Sentence-transformers available for cross-encoder")
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence-transformers not available")


@dataclass
class RerankResult:
    """Result from reranking operation."""
    text: str
    score: float
    original_index: int
    metadata: dict[str, Any] | None = None


class RerankerProcessor:
    """Processor for reranking search results using cross-encoder models."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialize the reranker processor.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.reranker = None
        self.cross_encoder = None

        self._initialize_reranker()

    def _initialize_reranker(self) -> None:
        """Initialize the reranker model."""
        try:
            if RERANKERS_AVAILABLE:
                # Try with rerankers library first
                try:
                    self.reranker = Reranker(self.model_name, model_type="cross-encoder")
                    logger.info(f"Initialized reranker with model: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize with rerankers library: {e}")
                    self.reranker = None

            if self.reranker is None and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Fallback to sentence-transformers
                try:
                    self.cross_encoder = CrossEncoder(self.model_name)
                    logger.info(f"Initialized cross-encoder with model: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize cross-encoder: {e}")
                    self.cross_encoder = None

            if self.reranker is None and self.cross_encoder is None:
                logger.warning("No reranker models available. Install rerankers or sentence-transformers.")

        except Exception as e:
            logger.exception("Error initializing reranker: %s", e)
            self.reranker = None
            self.cross_encoder = None

    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.reranker is not None or self.cross_encoder is not None

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[RerankResult]:
        """Rerank documents based on query relevance.

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None for all)
            metadata: Optional metadata for each document

        Returns:
            List of reranked results with scores
        """
        if not self.is_available():
            logger.warning("No reranker available, returning original order")
            return self._create_fallback_results(documents, metadata)

        try:
            if self.reranker is not None:
                return self._rerank_with_rerankers(query, documents, top_k, metadata)
            if self.cross_encoder is not None:
                return self._rerank_with_cross_encoder(query, documents, top_k, metadata)
        except Exception as e:
            logger.exception("Error during reranking: %s", e)
            return self._create_fallback_results(documents, metadata)

    def _rerank_with_rerankers(
        self,
        query: str,
        documents: list[str],
        top_k: int | None,
        metadata: list[dict[str, Any]] | None,
    ) -> list[RerankResult]:
        """Rerank using the rerankers library."""
        try:
            # Use rerankers library
            results = self.reranker.rank(query, documents, top_k=top_k)

            rerank_results = []
            for result in results:
                # Extract result information (format may vary by library version)
                if hasattr(result, "text") and hasattr(result, "score"):
                    text = result.text
                    score = result.score
                    original_idx = getattr(result, "index", 0)
                # Fallback for different result formats
                elif isinstance(result, dict):
                    text = result.get("text", documents[0])
                    score = result.get("score", 0.0)
                    original_idx = result.get("index", 0)
                else:
                    text = str(result)
                    score = 0.0
                    original_idx = 0

                doc_metadata = metadata[original_idx] if metadata and original_idx < len(metadata) else None

                rerank_results.append(RerankResult(
                    text=text,
                    score=score,
                    original_index=original_idx,
                    metadata=doc_metadata,
                ))

            return rerank_results

        except Exception as e:
            logger.exception("Error with rerankers library: %s", e)
            raise

    def _rerank_with_cross_encoder(
        self,
        query: str,
        documents: list[str],
        top_k: int | None,
        metadata: list[dict[str, Any]] | None,
    ) -> list[RerankResult]:
        """Rerank using sentence-transformers cross-encoder."""
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)

            # Create results with scores and original indices
            scored_results = []
            for i, (doc, score) in enumerate(zip(documents, scores, strict=False)):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else None
                scored_results.append(RerankResult(
                    text=doc,
                    score=float(score),
                    original_index=i,
                    metadata=doc_metadata,
                ))

            # Sort by score (descending)
            scored_results.sort(key=lambda x: x.score, reverse=True)

            # Return top_k results if specified
            if top_k is not None:
                scored_results = scored_results[:top_k]

            return scored_results

        except Exception as e:
            logger.exception("Error with cross-encoder: %s", e)
            raise

    def _create_fallback_results(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None,
    ) -> list[RerankResult]:
        """Create fallback results without reranking."""
        results = []
        for i, doc in enumerate(documents):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else None
            results.append(RerankResult(
                text=doc,
                score=1.0,  # Default score
                original_index=i,
                metadata=doc_metadata,
            ))
        return results


def create_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> RerankerProcessor:
    """Create a reranker processor instance.

    Args:
        model_name: Cross-encoder model name

    Returns:
        RerankerProcessor instance
    """
    return RerankerProcessor(model_name)


def get_qwen_reranker() -> RerankerProcessor | None:
    """Get a Qwen-based reranker if available.
    Note: Qwen3-Reranker models are not yet widely available via APIs.

    Returns:
        RerankerProcessor or None if not available
    """
    # Try different potential Qwen reranker model names
    qwen_models = [
        "Qwen/Qwen3-Reranker-8B",
        "Qwen/Qwen3-Reranker-4B",
        "Qwen/Qwen3-Reranker-0.6B",
    ]

    for model_name in qwen_models:
        try:
            reranker = RerankerProcessor(model_name)
            if reranker.is_available():
                logger.info("Successfully initialized Qwen reranker: %s", model_name)
                return reranker
        except Exception as e:
            logger.debug("Failed to initialize %s: %s", model_name, e)
            continue

    logger.warning("No Qwen reranker models available. Using default cross-encoder.")
    return None


# Singleton instance
_default_reranker = None

def get_default_reranker() -> RerankerProcessor:
    """Get the default reranker instance."""
    global _default_reranker
    if _default_reranker is None:
        # Try Qwen first, fallback to default
        _default_reranker = get_qwen_reranker()
        if _default_reranker is None:
            _default_reranker = create_reranker()
    return _default_reranker
