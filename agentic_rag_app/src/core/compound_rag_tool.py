"""
Compound RAG Tool using Smolagents Tool Composition

This module creates a compound tool that chains:
RAG Search → Content Synthesis → Citation Formatting
"""

import json
import logging
import time
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)


class RAGSearchTool(Tool):
    """Step 1: RAG Search Tool"""

    name = "rag_search"
    description = "Search the knowledge base for relevant documents"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for knowledge base",
        }
    }
    output_type = "object"

    def __init__(self, vector_store=None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> dict[str, Any]:
        """Search and return structured document data."""
        if not self.vector_store:
            return {"error": "Vector store not available", "documents": []}

        try:
            print(f"🔍 RAG Search: '{query}'")

            # Perform searches
            from src.utils.embedding_service import embed_query

            query_embedding = embed_query(query)

            semantic_results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=10,  # Get more results for reranking
                use_cache=True,
            )

            # Apply reranking if beneficial
            try:
                from src.utils.reranker import rerank_results

                reranked_results, was_reranked = rerank_results(
                    query, semantic_results, top_k=5
                )
                if was_reranked:
                    print("🔄 Applied cross-encoder reranking")
                    semantic_results = reranked_results
                else:
                    semantic_results = semantic_results[:5]  # Limit to top 5
            except Exception as e:
                print(f"⚠️ Reranking failed: {e}")
                semantic_results = semantic_results[:5]

            # Structure the results
            documents = []
            for result in semantic_results:
                doc_info = self._extract_document_info(result)
                documents.append(doc_info)

            return {
                "query": query,
                "documents": documents,
                "search_time": time.time(),
                "method": "semantic",
            }

        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return {"error": str(e), "documents": []}

    def _extract_document_info(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract document information from search result."""
        import os

        doc_info = {
            "content": "",
            "filename": "Unknown Document",
            "filepath": "",
            "score": result.get("score", 0.0),
            "id": result.get("id", ""),
        }

        # Extract content and metadata
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

            doc_info["filename"] = payload.get(
                "file_name", payload.get("title", "Unknown Document")
            )
            doc_info["filepath"] = payload.get("file_path", "")

            # Try to read full file if we have limited content and a file path
            if (
                doc_info["filepath"]
                and os.path.exists(doc_info["filepath"])
                and len(content) < 100
            ):
                try:
                    with open(doc_info["filepath"], encoding="utf-8") as f:
                        file_content = f.read()
                        # If we found content in the file, use a relevant section
                        if content and content.strip() in file_content:
                            # Get context around the found content
                            start_idx = max(0, file_content.find(content) - 300)
                            end_idx = min(
                                len(file_content),
                                file_content.find(content) + len(content) + 700,
                            )
                            content = file_content[start_idx:end_idx]
                        else:
                            # Use the first 1000 characters as a sample
                            content = file_content[:1000]
                except Exception as e:
                    print(f"⚠️ Could not read full file {doc_info['filepath']}: {e}")

            doc_info["content"] = content[:1000] if len(content) > 1000 else content

        return doc_info


class ContentSynthesisTool(Tool):
    """Step 2: Content Synthesis Tool"""

    name = "content_synthesis"
    description = "Synthesize coherent answer from search results"
    inputs = {
        "search_results": {
            "type": "object",
            "description": "Results from RAG search containing query and documents",
        }
    }
    output_type = "object"

    def forward(self, search_results: dict[str, Any]) -> dict[str, Any]:
        """Synthesize coherent answer from documents."""
        print("🧠 Synthesizing content...")

        if "error" in search_results:
            return {
                "synthesis": f"❌ Search failed: {search_results['error']}",
                "key_points": [],
                "definition": "",
            }

        documents = search_results.get("documents", [])
        query = search_results.get("query", "")

        if not documents:
            return {
                "synthesis": f"❌ No relevant documents found for: '{query}'",
                "key_points": [],
                "definition": "",
            }

        # Extract the main definition
        definition = self._extract_definition(query, documents)

        # Extract key points from all documents
        key_points = self._extract_key_points(documents)

        # Create coherent synthesis
        synthesis_parts = []

        if definition:
            synthesis_parts.append(definition)

        if key_points:
            synthesis_parts.append("\n**Key aspects:**")
            for i, point in enumerate(key_points, 1):
                synthesis_parts.append(f"• {point}")

        synthesis = (
            "\n".join(synthesis_parts)
            if synthesis_parts
            else f"Information found about '{query}' but needs better synthesis."
        )

        return {
            "synthesis": synthesis,
            "key_points": key_points,
            "definition": definition,
            "query": query,
        }

    def _extract_definition(self, query: str, documents: list[dict[str, Any]]) -> str:
        """Extract the main definition from documents."""
        subject = (
            query.lower()
            .replace("what is", "")
            .replace("what are", "")
            .replace("define", "")
            .strip()
            .replace("the ", "")
        )

        # Debug: print content to see what we're working with
        print(f"🔍 Looking for definition of '{subject}' in {len(documents)} documents")

        # Look for definition sentences across all documents
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            print(f"Document {i + 1} content preview: {content[:100]}...")

            if not content:
                continue

            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                # Look for lines that likely contain definitions
                if (
                    len(line) > 40
                    and not line.startswith(("#", "-", "*", "```"))
                    and any(
                        word in line.lower()
                        for word in [
                            "effect",
                            "phenomenon",
                            "process",
                            "theory",
                            "equilibrium",
                            "described",
                            "fundamental",
                        ]
                    )
                ):
                    print(f"✅ Found potential definition: {line[:100]}...")
                    return line

        # Fallback: look for any substantial sentence
        for doc in documents:
            content = doc.get("content", "")
            if content:
                # Split by sentences (rough)
                sentences = content.replace("\n", " ").split(". ")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 50 and not sentence.startswith(("#", "-", "*")):
                        print(f"📝 Using fallback definition: {sentence[:100]}...")
                        return sentence + "."

        return ""

    def _extract_key_points(self, documents: list[dict[str, Any]]) -> list[str]:
        """Extract key bullet points from documents."""
        key_points = []

        for doc in documents:
            content = doc.get("content", "")
            lines = content.split("\n")

            for line in lines:
                line = line.strip()
                if line.startswith(("-", "*")) and "**" in line:
                    # Extract bullet points with bold text
                    clean_line = line.lstrip("-* ").strip()
                    if clean_line and len(clean_line) > 10:
                        key_points.append(clean_line)
                        if len(key_points) >= 6:  # Limit key points
                            break

            if len(key_points) >= 6:
                break

        return key_points


class CitationFormatterTool(Tool):
    """Step 3: Citation Formatting Tool"""

    name = "citation_formatter"
    description = "Format final answer with proper citations and sources"
    inputs = {
        "synthesis_results": {
            "type": "object",
            "description": "Results from content synthesis",
        },
        "search_results": {
            "type": "object",
            "description": "Original search results with document details",
        },
    }
    output_type = "string"

    def forward(
        self, synthesis_results: dict[str, Any], search_results: dict[str, Any]
    ) -> str:
        """Format the final answer with citations."""
        print("📚 Formatting citations...")

        synthesis = synthesis_results.get("synthesis", "")
        documents = search_results.get("documents", [])

        if not documents:
            return synthesis

        # Build the final formatted response
        response_parts = []

        # Add the synthesized content
        response_parts.append(synthesis)

        # Add detailed source citations
        response_parts.append("\n📚 **SOURCES:**")
        response_parts.append("=" * 50)

        for i, doc in enumerate(documents, 1):
            citation = self._create_detailed_citation(doc, i)
            response_parts.append(f"\n{citation}")

            # Add content preview
            content = doc.get("content", "")
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                response_parts.append(f'    📝 "{preview}"')

        return "\n".join(response_parts)

    def _create_detailed_citation(self, doc: dict[str, Any], index: int) -> str:
        """Create a detailed citation for a document."""
        filename = doc.get("filename", "Unknown Document")
        filepath = doc.get("filepath", "")
        score = doc.get("score", 0.0)

        citation_parts = [f"[{index}]"]

        # Document name
        if filename != "Unknown Document":
            citation_parts.append(f"📄 {filename}")

        # File path (cleaned)
        if filepath:
            if "test_data" in filepath:
                clean_path = filepath[filepath.find("test_data") :]
                citation_parts.append(f"📁 {clean_path}")
            else:
                citation_parts.append(f"📁 {filepath}")

        # Relevance score
        relevance = (
            "🎯 High" if score > 0.8 else "🔸 Medium" if score > 0.6 else "🔹 Low"
        )
        citation_parts.append(f"{relevance} ({score:.2f})")

        return " | ".join(citation_parts)


class CompoundRAGTool(Tool):
    """Compound tool that chains RAG search → Synthesis → Citation formatting"""

    name = "compound_knowledge_search"
    description = "Comprehensive knowledge search with synthesis and proper citations"
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query for knowledge base",
        }
    }
    output_type = "string"

    def __init__(self, vector_store=None, **kwargs):
        super().__init__(**kwargs)

        # Initialize component tools
        self.search_tool = RAGSearchTool(vector_store=vector_store)
        self.synthesis_tool = ContentSynthesisTool()
        self.citation_tool = CitationFormatterTool()

        print("🔗 Compound RAG Tool initialized with tool chain:")
        print("   1. RAG Search → 2. Content Synthesis → 3. Citation Formatting")

    def forward(self, query: str) -> str:
        """Execute the complete RAG pipeline."""
        start_time = time.time()
        print(f"🚀 Starting compound RAG pipeline for: '{query}'")

        try:
            # Step 1: RAG Search
            search_results = self.search_tool.forward(query)

            # Step 2: Content Synthesis
            synthesis_results = self.synthesis_tool.forward(search_results)

            # Step 3: Citation Formatting
            final_result = self.citation_tool.forward(synthesis_results, search_results)

            elapsed = time.time() - start_time
            print(f"✅ Compound RAG pipeline completed in {elapsed:.2f}s")

            return final_result

        except Exception as e:
            error_msg = f"❌ Compound RAG pipeline failed: {e!s}"
            print(error_msg)
            return error_msg


def create_compound_rag_tool(vector_store=None):
    """Create a compound RAG tool instance."""
    return CompoundRAGTool(vector_store=vector_store)
