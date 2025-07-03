#!/usr/bin/env python3
"""
Cross-encoder reranker for improving retrieval quality.
Uses ms-marco-MiniLM-L-6-v2 for high-quality query-document relevance scoring.
"""

import logging
import time
from typing import Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval quality.

    Uses a pre-trained cross-encoder model to rerank search results
    based on query-document relevance scores.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

        # Performance tracking
        self.rerank_count = 0
        self.total_rerank_time = 0.0

    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("✅ Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load cross-encoder model: {e}")
            self.model = None

    def should_rerank(
        self, results: list[dict[str, Any]], threshold: float = 0.05
    ) -> bool:
        """
        Determine if reranking would be beneficial.

        Args:
            results: List of search results with scores
            threshold: Score difference threshold for reranking

        Returns:
            True if reranking is recommended
        """
        if not results or len(results) < 2:
            return False

        scores = [r.get("score", 0.0) for r in results]

        # Check if top results have similar scores
        if len(scores) >= 2:
            score_gap = scores[0] - scores[1]
            if score_gap < threshold:
                return True

        # Check if multiple results are within threshold of top score
        close_scores = sum(1 for score in scores[1:] if scores[0] - score < threshold)
        return close_scores >= 2

    def rerank(
        self, query: str, results: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Rerank search results using cross-encoder.

        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return (None = all)

        Returns:
            Reranked list of results with updated scores
        """
        if not self.model:
            logger.warning(
                "Cross-encoder model not available, returning original results"
            )
            return results

        if not results:
            return results

        start_time = time.time()

        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                # Extract content from various possible fields
                content = self._extract_content(result)
                if content:
                    pairs.append([query, content])
                else:
                    pairs.append([query, ""])  # Empty content fallback

            if not pairs:
                logger.warning("No content found for reranking")
                return results

            # Get cross-encoder scores
            logger.debug(f"Reranking {len(pairs)} results for query: '{query[:50]}...'")
            cross_scores = self.model.predict(pairs)

            # Update results with new scores
            reranked_results = []
            for i, result in enumerate(results):
                new_result = result.copy()
                new_result["original_score"] = result.get("score", 0.0)
                new_result["cross_encoder_score"] = float(cross_scores[i])
                new_result["score"] = float(cross_scores[i])  # Replace original score
                reranked_results.append(new_result)

            # Sort by new scores (descending)
            reranked_results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

            # Apply top_k limit if specified
            if top_k:
                reranked_results = reranked_results[:top_k]

            # Update performance metrics
            elapsed = time.time() - start_time
            self.rerank_count += 1
            self.total_rerank_time += elapsed

            logger.debug(f"✅ Reranked {len(results)} results in {elapsed:.3f}s")

            return reranked_results

        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}")
            return results

    def _extract_content(self, result: dict[str, Any]) -> str:
        """
        Extract content from a search result for reranking.

        Args:
            result: Search result dictionary

        Returns:
            Extracted content string
        """
        # Try different content fields
        content = (
            result.get("content", "")
            or result.get("text", "")
            or result.get("payload", {}).get("content", "")
            or result.get("payload", {}).get("text", "")
        )

        # Parse _node_content if available
        if not content and "payload" in result and "_node_content" in result["payload"]:
            try:
                import json

                node_data = json.loads(result["payload"]["_node_content"])
                content = node_data.get("text", "")
            except (json.JSONDecodeError, KeyError):
                pass

        # Truncate content for efficiency (cross-encoders have token limits)
        if content and len(content) > 512:
            content = content[:512] + "..."

        return content.strip()

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get reranker performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        avg_time = (
            self.total_rerank_time / self.rerank_count if self.rerank_count > 0 else 0.0
        )

        return {
            "model_name": self.model_name,
            "rerank_count": self.rerank_count,
            "total_time": self.total_rerank_time,
            "avg_time_per_rerank": avg_time,
            "model_loaded": self.model is not None,
        }


# Global reranker instance (lazy loading)
_reranker_instance = None


def get_reranker() -> CrossEncoderReranker:
    """
    Get the global reranker instance (singleton pattern).

    Returns:
        CrossEncoderReranker instance
    """
    global _reranker_instance  # noqa: PLW0603
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance


def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    force: bool = False,
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Convenience function to rerank results with automatic decision logic.

    Args:
        query: Search query
        results: List of search results
        force: Force reranking even if not recommended
        top_k: Number of top results to return

    Returns:
        Tuple of (reranked_results, was_reranked)
    """
    reranker = get_reranker()

    # Decide whether to rerank
    should_rerank = force or reranker.should_rerank(results)

    if should_rerank:
        reranked = reranker.rerank(query, results, top_k)
        return reranked, True
    else:
        # Apply top_k limit without reranking
        limited_results = results[:top_k] if top_k else results
        return limited_results, False


if __name__ == "__main__":
    # Test the reranker
    print("🧪 Testing Cross-Encoder Reranker")

    # Sample test data
    test_query = "what is ultrafiltration?"
    test_results = [
        {
            "score": 0.75,
            "content": (
                "Ultrafiltration is a membrane separation process used in water "
                "treatment."
            ),
            "id": "1",
        },
        {
            "score": 0.74,
            "content": (
                "Membrane filtration includes microfiltration, ultrafiltration, "
                "and nanofiltration."
            ),
            "id": "2",
        },
        {
            "score": 0.73,
            "content": (
                "Water treatment plants use various filtration methods for "
                "purification."
            ),
            "id": "3",
        },
    ]

    reranker = CrossEncoderReranker()

    print(f"Original scores: {[r['score'] for r in test_results]}")

    reranked = reranker.rerank(test_query, test_results)
    print(f"Reranked scores: {[r['cross_encoder_score'] for r in reranked]}")

    print("✅ Reranker test completed")
