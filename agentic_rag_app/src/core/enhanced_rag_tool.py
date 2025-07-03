"""
Enhanced RAG Tool with Better Citations and Debug Output

This module provides an improved RAG tool that shows better document sources,
citations, and debug information in the terminal.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)


class EnhancedRAGTool(Tool):
    """Enhanced RAG tool with improved citations and debug output."""

    name = "knowledge_search"
    description = "Search knowledge base with enhanced citations and source tracking"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for knowledge base",
        }
    }
    output_type = "string"

    def __init__(self, vector_store=None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.search_history = []
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _debug_print(self, message: str, level: str = "INFO"):
        """Print debug message to terminal."""
        emoji_map = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SEARCH": "🔍",
            "RESULT": "📄",
            "CITATION": "📚",
        }
        emoji = emoji_map.get(level, "•")
        print(f"{emoji} {message}")

    def _extract_document_info(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract comprehensive document information."""
        doc_info = {
            "content": "",
            "filename": "Unknown Document",
            "filepath": "",
            "section": "",
            "score": result.get("score", 0.0),
            "method": result.get("search_method", "unknown"),
        }

        # Extract content
        if "payload" in result:
            payload = result["payload"]

            # Try different content fields
            content = (
                result.get("content", "")
                or result.get("text", "")
                or payload.get("content", "")
                or payload.get("text", "")
            )

            # Try to parse _node_content JSON
            if not content and "_node_content" in payload:
                try:
                    node_data = json.loads(payload["_node_content"])
                    content = node_data.get("text", "")
                except json.JSONDecodeError:
                    pass

            doc_info["content"] = (
                content[:1000] + "..." if len(content) > 1000 else content
            )

            # Extract metadata
            doc_info["filename"] = payload.get(
                "file_name", payload.get("title", "Unknown Document")
            )
            doc_info["filepath"] = payload.get("file_path", "")
            doc_info["section"] = payload.get("section", payload.get("chunk_id", ""))

            # Try to read full file if path exists
            if doc_info["filepath"] and os.path.exists(doc_info["filepath"]):
                try:
                    with open(doc_info["filepath"], encoding="utf-8") as f:
                        file_content = f.read()
                        # Extract relevant section around the found content
                        if content and content in file_content:
                            start_idx = max(0, file_content.find(content) - 200)
                            end_idx = min(
                                len(file_content),
                                file_content.find(content) + len(content) + 200,
                            )
                            doc_info["content"] = file_content[start_idx:end_idx]
                except Exception as e:
                    self._debug_print(f"Could not read full file: {e}", "WARNING")

        return doc_info

    def _create_citation(self, doc_info: dict[str, Any]) -> str:
        """Create a proper citation with source information."""
        citation_parts = []

        # Document name
        if doc_info["filename"] != "Unknown Document":
            citation_parts.append(f"📄 {doc_info['filename']}")

        # File path (make it more readable)
        if doc_info["filepath"]:
            if "test_data" in doc_info["filepath"]:
                clean_path = doc_info["filepath"][
                    doc_info["filepath"].find("test_data") :
                ]
                citation_parts.append(f"📁 {clean_path}")
            else:
                citation_parts.append(f"📁 {os.path.basename(doc_info['filepath'])}")

        # Section/chunk info
        if doc_info["section"]:
            citation_parts.append(f"📍 Section: {doc_info['section']}")

        # Relevance score
        score = doc_info["score"]
        relevance = (
            "🎯 High" if score > 0.8 else "🔸 Medium" if score > 0.6 else "🔹 Low"
        )
        citation_parts.append(f"{relevance} relevance ({score:.2f})")

        return " | ".join(citation_parts)

    def _synthesize_coherent_answer(
        self, query: str, documents: list[dict[str, Any]]
    ) -> str:
        """Create a coherent answer from the document content."""
        if not documents:
            return f"No information found for: '{query}'"

        # Extract key information from documents
        all_content = []
        for doc in documents:
            if doc["content"]:
                all_content.append(doc["content"])

        if not all_content:
            return f"No content available for: '{query}'"

        # For "what is" questions, create a definition-style answer
        if query.lower().startswith(("what is", "what are", "define")):
            # Extract the subject from the query
            subject = (
                query.lower()
                .replace("what is", "")
                .replace("what are", "")
                .replace("define", "")
                .strip()
                .replace("the ", "")
            )
            subject_title = subject.title() if subject else "The concept"

            # Look for the best definition across all documents
            best_definition = ""

            # Look for document 3 first as it often has the main introduction
            for doc in documents:
                content = doc["content"]
                lines = content.split("\n")

                for line in lines:
                    line = line.strip()
                    # Look for lines that contain the subject and definition keywords
                    if subject.lower() in line.lower() and any(
                        word in line.lower()
                        for word in [
                            "is a",
                            "is an",
                            "describes",
                            "refers to",
                            "phenomenon",
                            "effect",
                            "process",
                        ]
                    ):
                        # Clean up the line
                        if line.startswith("#"):
                            continue
                        if len(line) > 50:  # Ensure it's substantial
                            best_definition = line
                            break

                if best_definition:
                    break

            # Fallback: look for any substantial sentence containing the subject
            if not best_definition:
                for content in all_content:
                    lines = content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if (
                            line
                            and len(line) > 30
                            and not line.startswith("#")
                            and not line.startswith("-")
                            and not line.startswith("*")
                        ):
                            best_definition = line
                            break
                    if best_definition:
                        break

            # Start building the coherent answer
            answer_parts = []

            if best_definition:
                # Clean up the definition
                clean_def = best_definition.replace(
                    "The Donnan effect,", "The Donnan effect"
                ).replace("The donnan effect,", "The Donnan effect")
                answer_parts.append(clean_def)
            else:
                answer_parts.append(
                    f"**{subject_title}** is a concept described in the available "
                    "documentation."
                )

            # Add key aspects from all documents
            key_aspects = []
            for i, content in enumerate(all_content[:3]):  # Use first 3 documents
                # Extract bullet points or key information
                lines = content.split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("-") or line.startswith("*"):
                        clean_line = line.lstrip("-* ").strip()
                        if clean_line and len(clean_line) > 10:
                            key_aspects.append(f"• {clean_line} [{i + 1}]")
                            if len(key_aspects) >= 5:  # Limit to 5 key aspects
                                break
                if len(key_aspects) >= 5:
                    break

            if key_aspects:
                answer_parts.append("\n**Key aspects:**")
                answer_parts.extend(key_aspects)

            return "\n".join(answer_parts)

        else:
            # For other types of questions, provide a more general synthesis
            content_preview = all_content[0][:300]
            return (
                f"Based on the available documentation:\\n\\n{content_preview}... [1]"
            )

    def _format_answer_with_sources(
        self, query: str, results: list[dict[str, Any]]
    ) -> str:
        """Format the answer with clear source attribution."""
        if not results:
            return f"❌ No relevant documents found for: '{query}'"

        # Extract document information
        documents = [self._extract_document_info(result) for result in results]

        # Create response with coherent answer + citations
        response_parts = []

        # Generate a coherent answer by synthesizing the content
        coherent_answer = self._synthesize_coherent_answer(query, documents)
        response_parts.append(coherent_answer)

        # Add detailed source citations
        response_parts.append("\n📚 SOURCES:")
        response_parts.append("=" * 50)

        # Add each source with citation
        for i, doc in enumerate(documents, 1):
            citation = self._create_citation(doc)
            response_parts.append(f"\n[{i}] {citation}")

            # Show key excerpt from this source
            content_preview = (
                doc["content"][:200] + "..."
                if len(doc["content"]) > 200
                else doc["content"]
            )
            response_parts.append(f'    📝 "{content_preview}"')

        return "\n".join(response_parts)

    def forward(self, query: str) -> str:
        """Enhanced search with debug output and better citations."""
        start_time = time.time()

        self._debug_print(f"Starting knowledge search for: '{query}'", "SEARCH")

        # Check cache
        cache_key = hashlib.md5(query.lower().encode()).hexdigest()
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry["timestamp"] < 1800:  # 30 minutes
                self.cache_hits += 1
                self._debug_print("Cache hit! Returning cached result", "SUCCESS")
                return cache_entry["result"]

        self.cache_misses += 1

        if not self.vector_store:
            error_msg = "❌ Vector store not available"
            self._debug_print(error_msg, "ERROR")
            return error_msg

        try:
            all_results = []
            search_methods = []

            # Method 1: Semantic search
            try:
                self._debug_print("Performing semantic search...", "SEARCH")
                from src.utils.embedding_service import embed_query

                query_embedding = embed_query(query)
                semantic_results = self.vector_store.search(
                    query_vector=query_embedding, top_k=5, use_cache=True
                )

                if semantic_results:
                    for result in semantic_results:
                        result["search_method"] = "semantic"
                    all_results.extend(semantic_results)
                    search_methods.append("semantic")
                    self._debug_print(
                        f"Semantic search found {len(semantic_results)} results",
                        "SUCCESS",
                    )
                else:
                    self._debug_print("No semantic results found", "WARNING")

            except Exception as e:
                self._debug_print(f"Semantic search failed: {e}", "ERROR")

            # Method 2: Text search
            try:
                self._debug_print("Performing text search...", "SEARCH")
                text_results = self.vector_store.search_by_text_filter(query, limit=3)

                if text_results:
                    for result in text_results:
                        result["search_method"] = "text"
                    all_results.extend(text_results)
                    search_methods.append("text")
                    self._debug_print(
                        f"Text search found {len(text_results)} results", "SUCCESS"
                    )
                else:
                    self._debug_print("No text results found", "WARNING")

            except Exception as e:
                self._debug_print(f"Text search failed: {e}", "ERROR")

            # Deduplicate results
            seen_ids = set()
            unique_results = []
            for result in all_results:
                result_id = result.get("id")
                if result_id not in seen_ids:
                    unique_results.append(result)
                    seen_ids.add(result_id)

            # Sort by relevance
            unique_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            # Apply reranking if beneficial
            try:
                from src.utils.reranker import rerank_results

                reranked_results, was_reranked = rerank_results(
                    query, unique_results[:10], top_k=5
                )
                if was_reranked:
                    self._debug_print("🔄 Applied cross-encoder reranking", "RERANK")
                    unique_results = reranked_results
                else:
                    unique_results = unique_results[
                        :5
                    ]  # Limit to top 5 without reranking
                    self._debug_print(
                        "⏭️ Skipped reranking (good score separation)", "RERANK"
                    )
            except Exception as e:
                self._debug_print(
                    f"⚠️ Reranking failed, using original order: {e}", "WARNING"
                )
                unique_results = unique_results[:5]

            search_time = time.time() - start_time
            self._debug_print(f"Search completed in {search_time:.2f}s", "SUCCESS")
            self._debug_print(f"Methods used: {', '.join(search_methods)}", "INFO")
            self._debug_print(f"Total unique results: {len(unique_results)}", "RESULT")

            # Format the response
            formatted_response = self._format_answer_with_sources(query, unique_results)

            # Cache the result
            self.cache[cache_key] = {
                "result": formatted_response,
                "timestamp": time.time(),
            }

            # Clean cache if too large
            if len(self.cache) > 100:
                old_keys = sorted(
                    self.cache.keys(), key=lambda k: self.cache[k]["timestamp"]
                )[:20]
                for old_key in old_keys:
                    del self.cache[old_key]

            return formatted_response

        except Exception as e:
            error_msg = f"❌ Search failed: {e!s}"
            self._debug_print(error_msg, "ERROR")
            return error_msg


def create_enhanced_rag_tool(vector_store=None):
    """Create an enhanced RAG tool instance."""
    return EnhancedRAGTool(vector_store=vector_store)
