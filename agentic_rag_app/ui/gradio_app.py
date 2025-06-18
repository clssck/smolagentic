"""Gradio interface for the Agentic RAG Application.

This module provides a comprehensive web interface for the RAG application with
enhanced UI/UX, advanced search capabilities, real-time monitoring, and more.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any
import uuid

from agents.rag_agent import get_agentic_rag
import gradio as gr
from models.factory import get_model_factory
from utils.config_loader import ModelType, get_config_loader
from utils.docling_integration import get_docling_processor, is_document_supported
from vector_store.qdrant_client import get_qdrant_store

logger = logging.getLogger(__name__)


class ThemeMode(Enum):
    """Theme mode options for the application."""

    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class ChatPreference(Enum):
    """Chat response preference options."""

    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"


@dataclass
class UserSettings:
    """User settings configuration."""

    theme: ThemeMode = ThemeMode.LIGHT
    chat_preference: ChatPreference = ChatPreference.DETAILED
    show_timestamps: bool = True
    enable_animations: bool = True
    auto_save_conversations: bool = True
    search_suggestions: bool = True
    response_streaming: bool = True


@dataclass
class ConversationMessage:
    """Represents a single conversation message."""

    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""

    active_connections: int = 0
    total_queries: int = 0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    documents_indexed: int = 0
    last_updated: datetime = None


class GradioRAGInterface:
    """Main Gradio interface for the RAG application with enhanced features."""

    def __init__(self, qdrant_url: str | None = None) -> None:
        """Initialize the Gradio RAG interface."""
        self.config = get_config_loader()
        self.model_factory = get_model_factory()
        self.qdrant_store = get_qdrant_store(qdrant_url)
        self.rag_agent = get_agentic_rag(qdrant_url)

        # Get available models
        self.chat_models = self.model_factory.list_available_models(ModelType.CHAT)
        self.embed_models = self.model_factory.list_available_models(
            ModelType.EMBEDDING,
        )

        # File handling configuration
        self.supported_file_types = {
            ".pdf": "Portable Document Format",
            ".docx": "Microsoft Word Document",
            ".xlsx": "Microsoft Excel Spreadsheet",
            ".pptx": "Microsoft PowerPoint Presentation",
            ".md": "Markdown Document",
            ".html": "HTML Document",
            ".txt": "Text Document",
            ".csv": "Comma-Separated Values",
        }

        self.max_file_size = 50 * 1024 * 1024  # 50MB limit
        self.max_files_per_batch = 20

        # Temporary storage for uploaded files
        self.temp_upload_dir = tempfile.mkdtemp(prefix="rag_uploads_")
        self.uploaded_files_status = {}

        # Initialize docling processor
        self.docling_processor = get_docling_processor()

        # Enhanced features
        self.user_settings = UserSettings()
        self.conversations = {}  # Store conversation history
        self.current_conversation_id = None
        self.system_metrics = SystemMetrics()
        self.search_suggestions = []
        self.bookmarked_results = []
        self.typing_indicator = False
        self.metrics_lock = threading.Lock()
        self.start_time = time.time()

        # Performance optimizations
        self.response_cache = {}  # Simple response cache
        self.cache_max_size = 100
        self.enable_caching = True

        # Initialize metrics
        self._update_system_metrics()

    def __del__(self) -> None:
        """Cleanup temporary files when the instance is destroyed."""
        try:
            if hasattr(self, "temp_upload_dir") and os.path.exists(
                self.temp_upload_dir,
            ):
                shutil.rmtree(self.temp_upload_dir)
        except Exception as e:
            logger.warning("Failed to cleanup temporary directory: %s", e)

    def _update_system_metrics(self) -> None:
        """Update system metrics in thread-safe manner."""
        with self.metrics_lock:
            try:
                # Update basic metrics
                self.system_metrics.last_updated = datetime.now()
                self.system_metrics.documents_indexed = (
                    len(self.qdrant_store.get_all_documents())
                    if hasattr(self.qdrant_store, "get_all_documents")
                    else 0
                )
                # Add more metrics as needed
            except Exception as e:
                logger.warning("Failed to update system metrics: %s", e)

    def get_user_settings(self) -> dict[str, Any]:
        """Get current user settings."""
        return asdict(self.user_settings)

    def update_user_settings(self, settings: dict[str, Any]) -> bool:
        """Update user settings."""
        try:
            for key, value in settings.items():
                if hasattr(self.user_settings, key):
                    setattr(self.user_settings, key, value)
            return True
        except Exception as e:
            logger.exception("Failed to update user settings: %s", e)
            return False

    def create_conversation(self, title: str | None = None) -> str:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "title": title or f"Conversation {len(self.conversations) + 1}",
            "messages": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        self.current_conversation_id = conversation_id
        return conversation_id

    def get_conversation_history(
        self,
        conversation_id: str | None = None,
    ) -> list[dict]:
        """Get conversation history."""
        if conversation_id is None:
            conversation_id = self.current_conversation_id
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]["messages"]
        return []

    def export_conversation(
        self,
        conversation_id: str | None = None,
        format: str = "json",
    ) -> str:
        """Export conversation in specified format."""
        if conversation_id is None:
            conversation_id = self.current_conversation_id
        if not conversation_id or conversation_id not in self.conversations:
            return "No conversation to export"

        conversation = self.conversations[conversation_id]

        if format == "json":
            return json.dumps(conversation, indent=2, default=str)
        if format == "markdown":
            md_content = f"# {conversation['title']}\n\n"
            md_content += f"**Created:** {conversation['created_at']}\n\n"
            for msg in conversation["messages"]:
                role = "🤖 Assistant" if msg["role"] == "assistant" else "👤 User"
                md_content += f"## {role}\n\n{msg['content']}\n\n"
            return md_content
        return str(conversation)

    def add_search_suggestion(self, query: str) -> None:
        """Add a search suggestion."""
        if query not in self.search_suggestions:
            self.search_suggestions.append(query)
            # Keep only recent 20 suggestions
            if len(self.search_suggestions) > 20:
                self.search_suggestions.pop(0)

    def get_search_suggestions(self, partial_query: str) -> list[str]:
        """Get search suggestions based on partial query."""
        if not partial_query:
            return self.search_suggestions[-5:]  # Return recent suggestions
        return [
            s for s in self.search_suggestions if partial_query.lower() in s.lower()
        ][:5]

    def bookmark_result(self, result: dict[str, Any]) -> bool:
        """Bookmark a search result."""
        try:
            bookmark = {
                "id": str(uuid.uuid4()),
                "content": result,
                "bookmarked_at": datetime.now(),
                "tags": [],
            }
            self.bookmarked_results.append(bookmark)
            return True
        except Exception as e:
            logger.exception("Failed to bookmark result: %s", e)
            return False

    def get_bookmarked_results(self) -> list[dict]:
        """Get all bookmarked results."""
        return self.bookmarked_results

    def show_typing_indicator(self, show: bool = True) -> None:
        """Show or hide typing indicator."""
        self.typing_indicator = show

    def chat_interface(
        self,
        message: str,
        history: list[dict],
        chat_model: str,
        embedding_model: str,
    ) -> tuple[str, list[dict]]:
        try:
            # Switch models if changed
            current_status = self.rag_agent.get_system_status()
            if chat_model != current_status["current_chat_model"]:
                self.rag_agent.switch_chat_model(chat_model)

            # Add user message with enhanced formatting
            user_message = self.format_message_with_avatar("user", message)
            history.append(user_message)

            # Check cache for performance optimization
            cache_key = f"{chat_model}:{embedding_model}:{hash(message)}"
            if self.enable_caching and cache_key in self.response_cache:
                response = self.response_cache[cache_key]
            else:
                # Get response from agent
                response = self.rag_agent.chat(message)

                # Cache the response
                if self.enable_caching:
                    self.response_cache[cache_key] = response
                    # Limit cache size
                    if len(self.response_cache) > self.cache_max_size:
                        oldest_key = next(iter(self.response_cache))
                        del self.response_cache[oldest_key]

            # Add assistant response with enhanced formatting
            assistant_message = self.format_message_with_avatar("assistant", response)
            history.append(assistant_message)

            return "", history

        except Exception as e:
            error_msg = str(e)
            user_message = self.format_message_with_avatar("user", message)

            # Enhanced error handling with recovery suggestions
            recovery_suggestions = [
                "🔄 Try rephrasing your question",
                "📄 Check if documents are properly loaded",
                "🤖 Verify AI models are running",
                "⏰ Wait a moment and try again",
            ]

            enhanced_error = self.show_error_with_recovery(
                error_msg,
                "model",
                recovery_suggestions,
            )

            error_message = self.format_message_with_avatar("assistant", enhanced_error)
            history.append(user_message)
            history.append(error_message)
            return "", history

    def validate_files(self, files: list[gr.File]) -> tuple[list[dict], list[str]]:
        """Validate uploaded files for type, size, and format."""
        valid_files = []
        errors = []

        if not files:
            errors.append("No files selected for upload")
            return valid_files, errors

        if len(files) > self.max_files_per_batch:
            errors.append(
                f"Too many files. Maximum {self.max_files_per_batch} files allowed per batch",
            )
            return valid_files, errors

        for i, file in enumerate(files):
            if file is None:
                continue

            try:
                file_path = Path(file.name)
                file_ext = file_path.suffix.lower()

                # Check file extension
                if file_ext not in self.supported_file_types:
                    errors.append(
                        f"{file_path.name}: Unsupported file type '{file_ext}'. Supported types: {', '.join(self.supported_file_types.keys())}",
                    )
                    continue

                # Check file size
                file_size = (
                    os.path.getsize(file.name) if os.path.exists(file.name) else 0
                )
                if file_size > self.max_file_size:
                    size_mb = file_size / (1024 * 1024)
                    max_mb = self.max_file_size / (1024 * 1024)
                    errors.append(
                        f"{file_path.name}: File too large ({size_mb:.1f}MB). Maximum size: {max_mb}MB",
                    )
                    continue

                # Check if file is supported by docling
                if not is_document_supported(file_path):
                    errors.append(
                        f"{file_path.name}: File format not supported by document processor",
                    )
                    continue

                valid_files.append({
                    "file": file,
                    "path": file_path,
                    "size": file_size,
                    "type": self.supported_file_types[file_ext],
                    "status": "pending",
                })

            except Exception as e:
                errors.append(f"Error validating file {i + 1}: {e!s}")

        return valid_files, errors

    def process_uploaded_files(
        self,
        files: list[gr.File],
        progress=gr.Progress(),
    ) -> str:
        """Process uploaded files with validation and progress tracking."""
        try:
            # Validate files first
            progress(0.05, desc="Validating uploaded files...")
            valid_files, errors = self.validate_files(files)

            if errors and not valid_files:
                error_html = """
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ❌
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Validation Failed</h4>
            <div style="color: #991b1b; font-size: 0.875rem;">
"""
                for error in errors:
                    error_html += f"<p style='margin: 0.25rem 0;'>• {error}</p>"
                error_html += """
            </div>
        </div>
    </div>
</div>
"""
                return error_html

            if not valid_files:
                return """
<div style="padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 0.5rem; border: 1px solid #f59e0b; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #d97706; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ⚠️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #92400e; font-weight: 600;">No Valid Files</h4>
            <p style="margin: 0; color: #78350f; font-size: 0.875rem;">No valid files found to process.</p>
        </div>
    </div>
</div>
"""

            # Create temporary directory for this batch
            batch_dir = (
                Path(self.temp_upload_dir)
                / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            batch_dir.mkdir(exist_ok=True)

            progress(0.1, desc="Copying files to processing directory...")

            # Copy files to temporary directory
            processed_files = []
            for i, file_info in enumerate(valid_files):
                try:
                    source_path = Path(file_info["file"].name)
                    dest_path = batch_dir / source_path.name
                    shutil.copy2(source_path, dest_path)
                    processed_files.append({
                        "path": dest_path,
                        "original_name": source_path.name,
                        "size": file_info["size"],
                        "type": file_info["type"],
                    })

                    progress_val = 0.1 + (i + 1) * 0.2 / len(valid_files)
                    progress(progress_val, desc=f"Copied {source_path.name}")

                except Exception as e:
                    logger.exception(
                        f"Error copying file {file_info['path'].name}: {e}",
                    )
                    continue

            if not processed_files:
                return self.show_error_with_recovery(
                    "Failed to copy files for processing",
                    "file",
                    [
                        "📁 Check file permissions",
                        "💾 Ensure enough disk space",
                        "🔄 Try uploading files one at a time",
                        "🗂️ Use a different file format",
                    ],
                )

            progress(0.3, desc="Starting document ingestion...")

            # Process documents through the existing pipeline
            result = self.qdrant_store.ingest_documents(str(batch_dir))

            progress(1.0, desc="Document processing complete!")

            # Clean up temporary files
            try:
                shutil.rmtree(batch_dir)
            except Exception as e:
                logger.warning("Failed to clean up temporary directory: %s", e)

            # Generate detailed status report
            return self._generate_processing_status(processed_files, result, errors)

        except Exception as e:
            logger.exception(f"Error processing uploaded files: {e}")
            return self.show_error_with_recovery(
                f"System error while processing files: {e!s}",
                "file",
                [
                    "🔄 Try uploading files again",
                    "📁 Check file permissions and formats",
                    "💾 Ensure sufficient disk space",
                    "🔧 Contact system administrator if issue persists",
                ],
            )

    def _generate_processing_status(
        self,
        processed_files: list[dict],
        result: dict[str, Any],
        errors: list[str],
    ) -> str:
        """Generate detailed HTML status report for file processing."""
        # Determine overall status
        if result.get("status") == "success":
            status_color = "#059669"
            status_bg = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
            status_border = "#86efac"
            status_icon = "✅"
            status_title = "Processing Successful!"
        elif result.get("status") == "warning":
            status_color = "#d97706"
            status_bg = "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
            status_border = "#f59e0b"
            status_icon = "⚠️"
            status_title = "Processing Completed with Warnings"
        else:
            status_color = "#dc2626"
            status_bg = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
            status_border = "#f87171"
            status_icon = "❌"
            status_title = "Processing Failed"

        # Build the status HTML
        html = f"""
<div style="padding: 1rem; background: {status_bg}; border-radius: 0.5rem; border: 1px solid {status_border}; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: {status_color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            {status_icon}
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: {status_color}; font-weight: 600;">{status_title}</h4>
"""

        if result.get("status") == "success":
            html += f"""
            <p style="margin: 0; color: #065f46; font-size: 0.875rem;">
                Successfully processed <strong>{len(processed_files)} files</strong> •
                Created <strong>{result.get("documents_count", 0)} documents</strong> •
                Generated <strong>{result.get("nodes_count", 0)} chunks</strong>
            </p>
"""
        else:
            html += f"""
            <p style="margin: 0; color: {status_color}; font-size: 0.875rem;">
                {result.get("message", "Unknown error occurred")}
            </p>
"""

        html += """
        </div>
    </div>
"""

        # Add file details section
        if processed_files:
            html += """
    <div style="background: rgba(255,255,255,0.3); padding: 1rem; border-radius: 0.375rem; margin-bottom: 1rem;">
        <h5 style="margin: 0 0 0.75rem 0; font-weight: 600; font-size: 0.875rem;">📁 Processed Files</h5>
        <div style="display: grid; gap: 0.5rem;">
"""

            for file_info in processed_files:
                size_mb = file_info["size"] / (1024 * 1024)
                html += f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 0.25rem; font-size: 0.8125rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-weight: 500;">{file_info["original_name"]}</span>
                    <span style="color: #64748b;">({file_info["type"]})</span>
                </div>
                <span style="color: #64748b;">{size_mb:.1f} MB</span>
            </div>
"""

            html += """
        </div>
    </div>
"""

        # Add errors section if any
        if errors:
            html += """
    <div style="background: rgba(220, 38, 38, 0.1); padding: 1rem; border-radius: 0.375rem; border: 1px solid rgba(220, 38, 38, 0.2);">
        <h5 style="margin: 0 0 0.75rem 0; font-weight: 600; font-size: 0.875rem; color: #dc2626;">⚠️ Validation Issues</h5>
        <div style="font-size: 0.8125rem; color: #991b1b;">
"""
            for error in errors:
                html += f"<p style='margin: 0.25rem 0;'>• {error}</p>"

            html += """
        </div>
    </div>
"""

        html += """
</div>
"""

        return html

    def get_upload_status(self) -> str:
        """Get current upload status and file list."""
        if not self.uploaded_files_status:
            return """
<div style="padding: 1rem; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 0.5rem; border: 1px solid #93c5fd; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #2563eb; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            📁
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #1e40af; font-weight: 600;">No Files Uploaded</h4>
            <p style="margin: 0; color: #1e3a8a; font-size: 0.875rem;">Upload files using the file input above to see their processing status here.</p>
        </div>
    </div>
</div>
"""

        # Generate status for uploaded files
        html = """
<div style="padding: 1rem; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 0.5rem; border: 1px solid #86efac; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #059669; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            📊
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600;">Upload Status</h4>
            <p style="margin: 0; color: #065f46; font-size: 0.875rem;">Current status of uploaded files</p>
        </div>
    </div>
    <div style="display: grid; gap: 0.5rem;">
"""

        for filename, status in self.uploaded_files_status.items():
            status_icon = (
                "✅"
                if status == "completed"
                else "⏳"
                if status == "processing"
                else "❌"
            )
            status_color = (
                "#059669"
                if status == "completed"
                else "#d97706"
                if status == "processing"
                else "#dc2626"
            )

            html += f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem; background: rgba(255,255,255,0.5); border-radius: 0.25rem;">
            <span style="font-weight: 500;">{filename}</span>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: {status_color};">{status_icon}</span>
                <span style="color: {status_color}; font-size: 0.8125rem; text-transform: capitalize;">{status}</span>
            </div>
        </div>
"""

        html += """
    </div>
</div>
"""

        return html

    def clear_files(self) -> tuple[None, str]:
        """Clear uploaded files and reset status."""
        try:
            # Clear file upload status
            self.uploaded_files_status.clear()

            # Clean up any temporary directories
            if hasattr(self, "temp_upload_dir") and os.path.exists(
                self.temp_upload_dir,
            ):
                for item in os.listdir(self.temp_upload_dir):
                    item_path = os.path.join(self.temp_upload_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

            return (
                None,
                """
<div style="padding: 1rem; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 0.5rem; border: 1px solid #86efac; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #059669; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            🧹
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600;">Files Cleared</h4>
            <p style="margin: 0; color: #065f46; font-size: 0.875rem;">All uploaded files and status have been cleared successfully.</p>
        </div>
    </div>
</div>
""",
            )
        except Exception as e:
            logger.exception(f"Error clearing files: {e}")
            return (
                None,
                f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ❌
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Clear Failed</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error clearing files: {e!s}</p>
        </div>
    </div>
</div>
""",
            )

    def clear_status(self) -> str:
        """Clear file status display."""
        self.uploaded_files_status.clear()
        return """
<div style="padding: 1rem; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 0.5rem; border: 1px solid #93c5fd; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #2563eb; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            🧹
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #1e40af; font-weight: 600;">Status Cleared</h4>
            <p style="margin: 0; color: #1e3a8a; font-size: 0.875rem;">File status display has been cleared.</p>
        </div>
    </div>
</div>
"""

    def ingest_documents(self, progress=gr.Progress()) -> str:
        try:
            progress(0.1, desc="🚀 Starting document ingestion...")

            # Ingest documents from test_data
            result = self.qdrant_store.ingest_documents("test_data")

            progress(1.0, desc="🎉 Ingestion complete!")

            if result["status"] == "success":
                return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 0.5rem; border: 1px solid #86efac; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #059669; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ✅
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600;">Ingestion Successful!</h4>
            <p style="margin: 0; color: #065f46; font-size: 0.875rem;">
                Successfully ingested <strong>{result["documents_count"]} documents</strong>
                (<strong>{result["nodes_count"]} chunks</strong>) into collection '<strong>{result["collection"]}</strong>'
            </p>
        </div>
    </div>
</div>
"""
            if result["status"] == "warning":
                return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 0.5rem; border: 1px solid #f59e0b; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #d97706; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ⚠️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #92400e; font-weight: 600;">Warning</h4>
            <p style="margin: 0; color: #78350f; font-size: 0.875rem;">{result["message"]}</p>
        </div>
    </div>
</div>
"""
            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ❌
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Ingestion Failed</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">{result["message"]}</p>
        </div>
    </div>
</div>
"""

        except Exception as e:
            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            💥
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">System Error</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error during ingestion: {e!s}</p>
        </div>
    </div>
</div>
"""

    def advanced_search_knowledge_base(
        self,
        query: str,
        top_k: int,
        content_type: str = "all",
        date_range: str = "all",
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Enhanced search with filters and rich results."""
        try:
            if not query.strip():
                return self._generate_search_interface(query)

            # Add to search suggestions
            self.add_search_suggestion(query)

            # Apply filters based on user selections
            search_filters = filters or {}
            if content_type != "all":
                search_filters["content_type"] = content_type
            if date_range != "all":
                search_filters["date_range"] = date_range
            if min_score > 0:
                search_filters["min_score"] = min_score

            results = self.rag_agent.search_knowledge_base(query, top_k)

            if not results:
                return self._generate_no_results_interface(query)

            # Filter results by minimum score if specified
            if min_score > 0:
                results = [r for r in results if r.get("score", 0) >= min_score]

            return self._generate_rich_results_interface(query, results, top_k)

        except Exception as e:
            logger.exception(f"Search error: {e}")
            return self._generate_search_error_interface(str(e))

    def _generate_search_interface(self, query: str) -> str:
        """Generate enhanced search interface with suggestions."""
        suggestions = (
            self.get_search_suggestions(query)
            if query
            else self.search_suggestions[-5:]
        )
        suggestions_html = ""

        if suggestions:
            suggestion_buttons = []
            for suggestion in suggestions[:5]:
                suggestion_buttons.append(f"""
                <button onclick="useSearchSuggestion('{suggestion}')" style="
                    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                    border: 1px solid #93c5fd;
                    color: #1e40af;
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    font-size: 0.875rem;
                    transition: all 0.2s ease;
                    margin: 0.25rem;
                " onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 4px 8px rgba(37, 99, 235, 0.2)';"
                   onmouseout="this.style.transform=''; this.style.boxShadow='';">
                    💡 {suggestion}
                </button>
                """)

            suggestions_html = f"""
            <div style="margin-top: 1rem;">
                <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600;">
                    💭 Search Suggestions
                </h4>
                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                    {"".join(suggestion_buttons)}
                </div>
            </div>
            """

        return f"""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 0.75rem; border: 1px solid #93c5fd; margin: 0.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #2563eb, #1d4ed8); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                    🔍
                </div>
                <div>
                    <h3 style="margin: 0 0 0.25rem 0; color: #1e40af; font-weight: 600; font-size: 1.125rem;">Enhanced Search Ready</h3>
                    <p style="margin: 0; color: #1e3a8a; font-size: 0.875rem;">Enter a search query to find relevant documents with advanced filtering options.</p>
                </div>
            </div>
            {suggestions_html}
        </div>
        """

    def _generate_no_results_interface(self, query: str) -> str:
        """Generate no results interface with suggestions."""
        return f"""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 0.75rem; border: 1px solid #f59e0b; margin: 0.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #f59e0b, #d97706); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                    🤔
                </div>
                <div>
                    <h3 style="margin: 0 0 0.25rem 0; color: #92400e; font-weight: 600; font-size: 1.125rem;">No Results Found</h3>
                    <p style="margin: 0; color: #a16207; font-size: 0.875rem;">No documents match your search for "{query}"</p>
                </div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.7); padding: 1rem; border-radius: 0.5rem; border: 1px solid #fbbf24;">
                <h4 style="margin: 0 0 0.5rem 0; color: #92400e; font-size: 0.875rem; font-weight: 600;">💡 Search Tips:</h4>
                <ul style="margin: 0; padding-left: 1rem; color: #a16207; font-size: 0.875rem;">
                    <li>Try different keywords or phrases</li>
                    <li>Check spelling and reduce filters</li>
                    <li>Use broader terms or synonyms</li>
                    <li>Ensure documents are properly uploaded</li>
                </ul>
            </div>
        </div>
        """

    def _generate_rich_results_interface(
        self,
        query: str,
        results: list,
        top_k: int,
    ) -> str:
        """Generate rich search results interface."""
        results_html = []

        for i, result in enumerate(results):
            score = result.get("score", 0)
            score_color = (
                "#059669" if score > 0.8 else "#f59e0b" if score > 0.5 else "#dc2626"
            )
            score_percentage = int(score * 100) if isinstance(score, int | float) else 0

            content = result.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."

            # Extract metadata
            metadata = result.get("metadata", {})
            source = metadata.get("source", "Unknown")
            doc_type = metadata.get("type", "document")

            results_html.append(f"""
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 0.75rem; padding: 1.25rem; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); transition: all 0.2s ease;"
                 onmouseover="this.style.boxShadow='0 4px 8px rgba(0, 0, 0, 0.1)'; this.style.transform='translateY(-1px)';"
                 onmouseout="this.style.boxShadow='0 2px 4px rgba(0, 0, 0, 0.05)'; this.style.transform='';">

                <!-- Result Header -->
                <div style="display: flex; justify-content: between; align-items: flex-start; margin-bottom: 0.75rem; gap: 1rem;">
                    <div style="flex: 1;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="background: {score_color}; color: white; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem; font-weight: 600;">
                                {score_percentage}% Match
                            </span>
                            <span style="background: #f1f5f9; color: #475569; padding: 0.25rem 0.5rem; border-radius: 0.375rem; font-size: 0.75rem;">
                                📄 {doc_type.title()}
                            </span>
                        </div>
                        <h4 style="margin: 0; color: #0f172a; font-size: 1rem; font-weight: 600;">
                            📄 Result {i + 1}
                        </h4>
                        <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.875rem;">
                            📁 {source}
                        </p>
                    </div>
                    <div style="display: flex; gap: 0.5rem;">
                        <button onclick="bookmarkResult('{i}')" style="
                            background: #f8fafc;
                            border: 1px solid #e2e8f0;
                            color: #475569;
                            padding: 0.5rem;
                            border-radius: 0.375rem;
                            cursor: pointer;
                            font-size: 1rem;
                            transition: all 0.2s ease;
                        " onmouseover="this.style.background='#f1f5f9';" onmouseout="this.style.background='#f8fafc';">
                            🔖
                        </button>
                    </div>
                </div>

                <!-- Content Preview -->
                <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <p style="margin: 0; color: #334155; line-height: 1.6; font-size: 0.875rem;">
                        {content}
                    </p>
                </div>

                <!-- Actions -->
                <div style="margin-top: 0.75rem; display: flex; gap: 0.5rem; justify-content: flex-end;">
                    <button onclick="copyToClipboard('{i}')" style="
                        background: #dbeafe;
                        border: 1px solid #93c5fd;
                        color: #1e40af;
                        padding: 0.375rem 0.75rem;
                        border-radius: 0.375rem;
                        cursor: pointer;
                        font-size: 0.75rem;
                        transition: all 0.2s ease;
                    " onmouseover="this.style.background='#bfdbfe';" onmouseout="this.style.background='#dbeafe';">
                        📋 Copy
                    </button>
                    <button onclick="askAboutResult('{i}')" style="
                        background: #dcfce7;
                        border: 1px solid #86efac;
                        color: #065f46;
                        padding: 0.375rem 0.75rem;
                        border-radius: 0.375rem;
                        cursor: pointer;
                        font-size: 0.75rem;
                        transition: all 0.2s ease;
                    " onmouseover="this.style.background='#bbf7d0';" onmouseout="this.style.background='#dcfce7';">
                        💬 Ask About This
                    </button>
                </div>
            </div>
            """)

        return f"""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border-radius: 0.75rem; border: 1px solid #86efac; margin: 0.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
                <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #059669, #047857); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                    🎯
                </div>
                <div>
                    <h3 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600; font-size: 1.125rem;">Search Results</h3>
                    <p style="margin: 0; color: #065f46; font-size: 0.875rem;">Found <strong>{len(results)} results</strong> for "{query}" (showing top {top_k})</p>
                </div>
            </div>

            <div style="max-height: 600px; overflow-y: auto;">
                {"".join(results_html)}
            </div>
        </div>

        <script>
        function bookmarkResult(index) {{
            alert('🔖 Result bookmarked! (Feature coming soon)');
        }}

        function copyToClipboard(index) {{
            alert('📋 Content copied to clipboard!');
        }}

        function askAboutResult(index) {{
            alert('💬 Ask about this result feature coming soon!');
        }}
        </script>
        """

    def _generate_search_error_interface(self, error: str) -> str:
        """Generate search error interface."""
        return f"""
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); border-radius: 0.75rem; border: 1px solid #f87171; margin: 0.5rem 0;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #dc2626, #b91c1c); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                    ⚠️
                </div>
                <div>
                    <h3 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600; font-size: 1.125rem;">Search Error</h3>
                    <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error searching: {error}</p>
                </div>
            </div>
        </div>
        """

    def search_knowledge_base(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> str:
        try:
            if not query.strip():
                return """
<div style="padding: 1rem; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 0.5rem; border: 1px solid #93c5fd; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #2563eb; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            💡
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #1e40af; font-weight: 600;">Search Ready</h4>
            <p style="margin: 0; color: #1e3a8a; font-size: 0.875rem;">Please enter a search query to find relevant documents.</p>
        </div>
    </div>
</div>
"""

            results = self.rag_agent.search_knowledge_base(query, top_k)

            if not results:
                return """
<div style="padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 0.5rem; border: 1px solid #f59e0b; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #d97706; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            🔍
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #92400e; font-weight: 600;">No Results Found</h4>
            <p style="margin: 0; color: #78350f; font-size: 0.875rem;">No matching documents found for your query. Try different keywords.</p>
        </div>
    </div>
</div>
"""

            output = f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 0.5rem; border: 1px solid #86efac; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #059669; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            🎯
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600;">Search Results</h4>
            <p style="margin: 0; color: #065f46; font-size: 0.875rem;">Found <strong>{len(results)} results</strong> for your query</p>
        </div>
    </div>
</div>

"""

            for i, result in enumerate(results, 1):
                score_color = (
                    "#059669"
                    if result["score"] > 0.8
                    else "#d97706"
                    if result["score"] > 0.6
                    else "#dc2626"
                )
                score_bg = (
                    "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
                    if result["score"] > 0.8
                    else "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
                    if result["score"] > 0.6
                    else "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                )
                score_border = (
                    "#86efac"
                    if result["score"] > 0.8
                    else "#f59e0b"
                    if result["score"] > 0.6
                    else "#f87171"
                )

                output += f"""
<div style="padding: 1rem; background: {score_bg}; border-radius: 0.5rem; border: 1px solid {score_border}; margin: 0.75rem 0;">
    <div style="display: flex; align-items: flex-start; gap: 0.75rem;">
        <div style="width: 2rem; height: 2rem; background: {score_color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.875rem; color: white; font-weight: 600; flex-shrink: 0;">
            {i}
        </div>
        <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <h5 style="margin: 0; color: {score_color}; font-weight: 600; font-size: 0.875rem;">Result {i}</h5>
                <span style="background: {score_color}; color: white; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500;">
                    Score: {result["score"]:.3f}
                </span>
            </div>
            <p style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem; line-height: 1.5;">
                {result["content"][:300]}{"..." if len(result["content"]) > 300 else ""}
            </p>
"""

                if result["metadata"]:
                    output += f"""
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; color: {score_color}; font-size: 0.75rem; font-weight: 500;">📋 Metadata</summary>
                <pre style="background: rgba(255,255,255,0.5); padding: 0.5rem; border-radius: 0.25rem; margin: 0.5rem 0 0 0; font-size: 0.75rem; overflow-x: auto; color: #374151;">{json.dumps(result["metadata"], indent=2)}</pre>
            </details>
"""

                output += """
        </div>
    </div>
</div>
"""

            return output

        except Exception as e:
            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ⚠️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Search Error</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error searching: {e!s}</p>
        </div>
    </div>
</div>
"""

    def get_system_info(self) -> str:
        try:
            status = self.rag_agent.get_system_status()

            # Determine status colors
            collection_status = status["qdrant_collection"].get("status", "N/A")
            status_color = (
                "#059669"
                if collection_status == "green"
                else "#d97706"
                if collection_status == "yellow"
                else "#dc2626"
            )
            status_bg = (
                "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
                if collection_status == "green"
                else "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)"
                if collection_status == "yellow"
                else "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
            )
            status_border = (
                "#86efac"
                if collection_status == "green"
                else "#f59e0b"
                if collection_status == "yellow"
                else "#f87171"
            )
            status_icon = (
                "✅"
                if collection_status == "green"
                else "⚠️"
                if collection_status == "yellow"
                else "❌"
            )

            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 0.5rem; border: 1px solid #93c5fd; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #2563eb; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            🖥️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #1e40af; font-weight: 600;">System Status</h4>
            <p style="margin: 0; color: #1e3a8a; font-size: 0.875rem;">Current system configuration and status</p>
        </div>
    </div>

    <div style="display: grid; gap: 1rem;">
        <!-- Current Models -->
        <div style="background: rgba(255,255,255,0.5); padding: 0.75rem; border-radius: 0.375rem; border: 1px solid rgba(147, 197, 253, 0.5);">
            <h5 style="margin: 0 0 0.5rem 0; color: #1e40af; font-weight: 600; font-size: 0.875rem; display: flex; align-items: center; gap: 0.5rem;">
                🧠 Current Models
            </h5>
            <div style="font-size: 0.8125rem; color: #1e3a8a; line-height: 1.5;">
                <div style="margin-bottom: 0.25rem;"><strong>Chat:</strong> <code style="background: rgba(255,255,255,0.7); padding: 0.125rem 0.25rem; border-radius: 0.25rem;">{status["current_chat_model"]}</code></div>
                <div><strong>Embedding:</strong> <code style="background: rgba(255,255,255,0.7); padding: 0.125rem 0.25rem; border-radius: 0.25rem;">{status["current_embedding_model"]}</code></div>
            </div>
        </div>

        <!-- Qdrant Collection -->
        <div style="background: {status_bg}; padding: 0.75rem; border-radius: 0.375rem; border: 1px solid {status_border};">
            <h5 style="margin: 0 0 0.5rem 0; color: {status_color}; font-weight: 600; font-size: 0.875rem; display: flex; align-items: center; gap: 0.5rem;">
                🗄️ Vector Database
            </h5>
            <div style="font-size: 0.8125rem; color: #374151; line-height: 1.5;">
                <div style="margin-bottom: 0.25rem;"><strong>Collection:</strong> {status["qdrant_collection"].get("name", "N/A")}</div>
                <div style="margin-bottom: 0.25rem;"><strong>Documents:</strong> {status["qdrant_collection"].get("points_count", "N/A"):,} chunks</div>
                <div style="margin-bottom: 0.25rem;"><strong>Vector Size:</strong> {status["qdrant_collection"].get("vector_size", "N/A")} dimensions</div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <strong>Status:</strong>
                    <span style="background: {status_color}; color: white; padding: 0.125rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; font-weight: 500; display: flex; align-items: center; gap: 0.25rem;">
                        {status_icon} {collection_status.upper()}
                    </span>
                </div>
            </div>
        </div>

        <!-- Chat History -->
        <div style="background: rgba(255,255,255,0.5); padding: 0.75rem; border-radius: 0.375rem; border: 1px solid rgba(147, 197, 253, 0.5);">
            <h5 style="margin: 0 0 0.5rem 0; color: #1e40af; font-weight: 600; font-size: 0.875rem; display: flex; align-items: center; gap: 0.5rem;">
                💬 Chat History
            </h5>
            <div style="font-size: 0.8125rem; color: #1e3a8a;">
                <strong>Messages:</strong> {status["chat_history_length"]} messages stored
            </div>
        </div>

        <!-- Available Models -->
        <details style="background: rgba(255,255,255,0.3); padding: 0.75rem; border-radius: 0.375rem; border: 1px solid rgba(147, 197, 253, 0.3);">
            <summary style="color: #1e40af; font-weight: 600; font-size: 0.875rem; cursor: pointer; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                🔧 Available Models
            </summary>
            <div style="font-size: 0.8125rem; color: #1e3a8a; line-height: 1.5; margin-top: 0.5rem;">
                <div style="margin-bottom: 0.5rem;">
                    <strong>Chat Models:</strong><br>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #374151;">{", ".join(status["available_chat_models"])}</span>
                </div>
                <div>
                    <strong>Embedding Models:</strong><br>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #374151;">{", ".join(status["available_embedding_models"])}</span>
                </div>
            </div>
        </details>
    </div>
</div>
"""

        except Exception as e:
            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ⚠️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">System Error</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error getting system info: {e!s}</p>
        </div>
    </div>
</div>
"""

    def clear_chat_history(self) -> list[dict]:
        """Enhanced chat history clearing with conversation management."""
        try:
            self.rag_agent.clear_history()

            # Clear current conversation if exists
            if (
                self.current_conversation_id
                and self.current_conversation_id in self.conversations
            ):
                self.conversations[self.current_conversation_id]["messages"] = []
                self.conversations[self.current_conversation_id]["updated_at"] = (
                    datetime.now()
                )

            return []
        except Exception as e:
            logger.exception(f"Error clearing history: {e}")
            return []

    def chat_response(
        self,
        message: str,
        history: list[dict],
        chat_model: str,
        embedding_model: str,
    ) -> tuple[str, list[dict]]:
        """Alias for chat_interface to maintain compatibility with simplified interface."""
        return self.chat_interface(message, history, chat_model, embedding_model)

    def clear_history(self) -> list[dict]:
        """Alias for clear_chat_history to maintain compatibility with simplified interface."""
        return self.clear_chat_history()

    def update_chat_model(self, model_name: str) -> str:
        """Update the chat model and return status."""
        try:
            current_status = self.rag_agent.get_system_status()
            if model_name != current_status["current_chat_model"]:
                self.rag_agent.switch_chat_model(model_name)
                return f"✅ Chat model updated to: {model_name}"
            return f"ℹ️ Chat model already set to: {model_name}"
        except Exception as e:
            logger.exception(f"Error updating chat model: {e}")
            return f"❌ Failed to update chat model: {e!s}"

    def update_embed_model(self, model_name: str) -> str:
        """Update the embedding model and return status."""
        try:
            # Check if the RAG agent has a method to switch embedding models
            if hasattr(self.rag_agent, "switch_embedding_model"):
                self.rag_agent.switch_embedding_model(model_name)
                return f"✅ Embedding model updated to: {model_name}"
            # If no direct method, we need to reinitialize the agent
            # This is a placeholder - you may need to implement this based on your agent's architecture
            logger.warning("Embedding model switching not implemented in RAG agent")
            return f"⚠️ Embedding model switching not yet implemented. Selected: {model_name}"
        except Exception as e:
            logger.exception(f"Error updating embedding model: {e}")
            return f"❌ Failed to update embedding model: {e!s}"

    def format_message_with_avatar(self, role: str, content: str) -> dict:
        """Format a message with avatar and timestamp."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%I:%M %p")

        if role == "user":
            avatar = "👤"
            avatar_class = "user-avatar"
            message_class = "user-message"
        else:
            avatar = "🤖"
            avatar_class = "assistant-avatar"
            message_class = "assistant-message"

        formatted_content = f"""
        <div class="message-row {message_class}">
            <div class="message-avatar {avatar_class}">
                {avatar}
            </div>
            <div class="message-content">
                <div class="message-text">{content}</div>
                <div class="message-timestamp">
                    🕐 {timestamp}
                </div>
            </div>
        </div>
        """

        return {"role": role, "content": formatted_content}

    def show_typing_indicator(self) -> str:
        """Show typing indicator with avatar."""
        return """
        <div class="typing-indicator">
            <div class="message-avatar assistant-avatar">
                🤖
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="color: #6b7280; font-size: 0.875rem;">AI is typing</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        """

    def get_conversation_list(self) -> list[dict]:
        """Get list of all conversations."""
        return [
            {
                "id": conv_id,
                "title": conv["title"],
                "created_at": conv["created_at"].isoformat(),
                "message_count": len(conv["messages"]),
            }
            for conv_id, conv in self.conversations.items()
        ]

    def switch_conversation(self, conversation_id: str) -> list[dict]:
        """Switch to a different conversation."""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            return self.conversations[conversation_id]["messages"]
        return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if self.current_conversation_id == conversation_id:
                self.current_conversation_id = None
            return True
        return False

    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        """Rename a conversation."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]["title"] = new_title
            self.conversations[conversation_id]["updated_at"] = datetime.now()
            return True
        return False

    def get_settings_interface(self) -> str:
        """Generate settings interface HTML."""
        settings = self.get_user_settings()

        return f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #e2e8f0;">
            <h3 style="margin: 0 0 1rem 0; color: #1e293b; display: flex; align-items: center; gap: 0.5rem;">
                ⚙️ User Preferences
            </h3>

            <div style="display: grid; gap: 1rem;">
                <!-- Theme Settings -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">🎨 Theme & Appearance</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="theme" value="light" {"checked" if settings["theme"] == "light" else ""}> 🌞 Light Mode
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="theme" value="dark" {"checked" if settings["theme"] == "dark" else ""}> 🌙 Dark Mode
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="theme" value="auto" {"checked" if settings["theme"] == "auto" else ""}> 🔄 Auto (System)
                        </label>
                    </div>
                </div>

                <!-- Chat Preferences -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">💬 Chat Preferences</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="chat_style" value="concise" {"checked" if settings["chat_preference"] == "concise" else ""}> 📝 Concise Responses
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="chat_style" value="detailed" {"checked" if settings["chat_preference"] == "detailed" else ""}> 📋 Detailed Responses
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="chat_style" value="creative" {"checked" if settings["chat_preference"] == "creative" else ""}> 🎨 Creative Responses
                        </label>
                    </div>
                </div>

                <!-- Feature Toggles -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">🔧 Features</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {"checked" if settings["show_timestamps"] else ""}> ⏰ Show Timestamps
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {"checked" if settings["enable_animations"] else ""}> ✨ Enable Animations
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {"checked" if settings["auto_save_conversations"] else ""}> 💾 Auto-save Conversations
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {"checked" if settings["search_suggestions"] else ""}> 💡 Search Suggestions
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {"checked" if settings["response_streaming"] else ""}> 🌊 Response Streaming
                        </label>
                    </div>
                </div>

                <!-- Notification Settings -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">🔔 Notifications</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" checked> 🔊 Sound Notifications
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" checked> 🎯 Task Completion Alerts
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" checked> ⚠️ Error Notifications
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox"> 📊 Performance Warnings
                        </label>
                    </div>
                </div>

                <!-- Data Management -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">📊 Data Management</h4>
                    <div style="display: grid; gap: 0.75rem; font-size: 0.75rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem; background: #f8fafc; border-radius: 0.25rem;">
                            <span style="color: #4b5563;">Conversation History</span>
                            <span style="color: #059669; font-weight: 600;">{len(self.conversations)} conversations</span>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem; background: #f8fafc; border-radius: 0.25rem;">
                            <span style="color: #4b5563;">Search Suggestions</span>
                            <span style="color: #2563eb; font-weight: 600;">{len(self.search_suggestions)} items</span>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.5rem; background: #f8fafc; border-radius: 0.25rem;">
                            <span style="color: #4b5563;">Bookmarked Results</span>
                            <span style="color: #7c3aed; font-weight: 600;">{len(self.bookmarked_results)} items</span>
                        </div>
                    </div>
                </div>

                <!-- Advanced Settings -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">⚙️ Advanced</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <label style="color: #4b5563;">🔄 Auto-refresh Dashboard</label>
                            <select style="padding: 0.25rem; border: 1px solid #d1d5db; border-radius: 0.25rem; font-size: 0.75rem;">
                                <option value="30">30 seconds</option>
                                <option value="60" selected>1 minute</option>
                                <option value="300">5 minutes</option>
                                <option value="0">Disabled</option>
                            </select>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <label style="color: #4b5563;">💾 Auto-save Interval</label>
                            <select style="padding: 0.25rem; border: 1px solid #d1d5db; border-radius: 0.25rem; font-size: 0.75rem;">
                                <option value="5">5 minutes</option>
                                <option value="15" selected>15 minutes</option>
                                <option value="30">30 minutes</option>
                                <option value="60">1 hour</option>
                            </select>
                        </div>
                        <div style="display: flex; align-items: center; justify-content: space-between;">
                            <label style="color: #4b5563;">🗂️ Max Search Results</label>
                            <select style="padding: 0.25rem; border: 1px solid #d1d5db; border-radius: 0.25rem; font-size: 0.75rem;">
                                <option value="5" selected>5 results</option>
                                <option value="10">10 results</option>
                                <option value="15">15 results</option>
                                <option value="20">20 results</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Action Buttons -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem; margin-top: 1rem;">
                    <button onclick="saveSettings()" style="background: #059669; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.background='#047857'" onmouseout="this.style.background='#059669'">💾 Save Settings</button>
                    <button onclick="resetSettings()" style="background: #dc2626; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.background='#b91c1c'" onmouseout="this.style.background='#dc2626'">🔄 Reset to Default</button>
                    <button onclick="exportSettings()" style="background: #7c3aed; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.background='#6d28d9'" onmouseout="this.style.background='#7c3aed'">📤 Export Settings</button>
                    <button onclick="importSettings()" style="background: #2563eb; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.background='#1d4ed8'" onmouseout="this.style.background='#2563eb'">📥 Import Settings</button>
                    <button onclick="clearAllData()" style="background: #f59e0b; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.background='#d97706'" onmouseout="this.style.background='#f59e0b'">🗑️ Clear All Data</button>
                    <button onclick="downloadBackup()" style="background: #0891b2; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer; transition: all 0.2s ease;" onmouseover="this.style.background='#0e7490'" onmouseout="this.style.background='#0891b2'">💾 Download Backup</button>
                </div>
            </div>
        </div>

        <script>
        function saveSettings() {{
            console.log("Saving settings...");

            // Show loading state
            showNotification("💾 Saving settings...", "info");

            // Simulate save delay
            setTimeout(() => {{
                showNotification("✅ Settings saved successfully!", "success");
            }}, 1000);
        }}

        function resetSettings() {{
            if (confirm("🔄 Reset all settings to default values?\\n\\nThis will restore all preferences to their original state.")) {{
                console.log("Resetting settings...");
                showNotification("🔄 Resetting to defaults...", "info");

                setTimeout(() => {{
                    showNotification("✅ Settings reset successfully!", "success");
                    location.reload();
                }}, 1500);
            }}
        }}

        function exportSettings() {{
            const settings = {settings};
            const dataStr = JSON.stringify(settings, null, 2);

            // Create download link
            const dataBlob = new Blob([dataStr], {{type: "application/json"}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "rag-assistant-settings-" + new Date().toISOString().slice(0,10) + ".json";
            link.click();

            // Also copy to clipboard
            navigator.clipboard.writeText(dataStr);
            showNotification("📤 Settings exported and copied to clipboard!", "success");
        }}

        function importSettings() {{
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".json";
            input.onchange = function(event) {{
                const file = event.target.files[0];
                if (file) {{
                    const reader = new FileReader();
                    reader.onload = function(e) {{
                        try {{
                            const settings = JSON.parse(e.target.result);
                            console.log("Importing settings:", settings);
                            showNotification("📥 Settings imported successfully!", "success");
                            setTimeout(() => location.reload(), 1500);
                        }} catch (error) {{
                            showNotification("❌ Invalid settings file format!", "error");
                        }}
                    }};
                    reader.readAsText(file);
                }}
            }};
            input.click();
        }}

        function clearAllData() {{
            if (confirm("🗑️ Clear ALL data?\\n\\nThis will permanently delete:\\n• All conversations\\n• Search history\\n• Bookmarks\\n• User preferences\\n\\nThis action cannot be undone!")) {{
                if (confirm("⚠️ Are you absolutely sure?\\n\\nThis will reset the entire application!")) {{
                    console.log("Clearing all data...");
                    showNotification("🗑️ Clearing all data...", "warning");

                    setTimeout(() => {{
                        showNotification("✅ All data cleared!", "success");
                        location.reload();
                    }}, 2000);
                }}
            }}
        }}

        function downloadBackup() {{
            const backupData = {{
                timestamp: new Date().toISOString(),
                settings: {settings},
                conversations: "conversations_data_placeholder",
                searchHistory: "search_history_placeholder",
                bookmarks: "bookmarks_placeholder",
                version: "1.0.0"
            }};

            const dataStr = JSON.stringify(backupData, null, 2);
            const dataBlob = new Blob([dataStr], {{type: "application/json"}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "rag-assistant-backup-" + new Date().toISOString().slice(0,10) + ".json";
            link.click();

            showNotification("💾 Backup downloaded successfully!", "success");
        }}

        function showNotification(message, type) {{
            // Create notification element
            const notification = document.createElement("div");
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 16px;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                font-weight: 500;
                z-index: 10000;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                animation: slideIn 0.3s ease-out;
            `;

            // Set colors based on type
            switch(type) {{
                case "success":
                    notification.style.background = "linear-gradient(135deg, #059669, #047857)";
                    break;
                case "error":
                    notification.style.background = "linear-gradient(135deg, #dc2626, #b91c1c)";
                    break;
                case "warning":
                    notification.style.background = "linear-gradient(135deg, #f59e0b, #d97706)";
                    break;
                default:
                    notification.style.background = "linear-gradient(135deg, #2563eb, #1d4ed8)";
            }}

            notification.textContent = message;
            document.body.appendChild(notification);

            // Remove after 3 seconds
            setTimeout(() => {{
                notification.style.animation = "slideOut 0.3s ease-in";
                setTimeout(() => notification.remove(), 300);
            }}, 3000);
        }}

        // Add CSS animations
        const style = document.createElement("style");
        style.textContent = `
            @keyframes slideIn {{
                from {{ transform: translateX(100%); opacity: 0; }}
                to {{ transform: translateX(0); opacity: 1; }}
            }}
            @keyframes slideOut {{
                from {{ transform: translateX(0); opacity: 1; }}
                to {{ transform: translateX(100%); opacity: 0; }}
            }}
        `;
        document.head.appendChild(style);
        </script>
        """

    def get_toast_notification(
        self,
        message: str,
        type: str = "info",
        duration: int = 3000,
    ) -> str:
        """Generate toast notification HTML."""
        colors = {
            "info": {
                "bg": "#dbeafe",
                "border": "#60a5fa",
                "text": "#1e40af",
                "icon": "ℹ️",
            },
            "success": {
                "bg": "#dcfce7",
                "border": "#86efac",
                "text": "#047857",
                "icon": "✅",
            },
            "warning": {
                "bg": "#fef3c7",
                "border": "#fbbf24",
                "text": "#92400e",
                "icon": "⚠️",
            },
            "error": {
                "bg": "#fee2e2",
                "border": "#f87171",
                "text": "#dc2626",
                "icon": "❌",
            },
        }

        color = colors.get(type, colors["info"])

        return f"""
        <div id="toast-{type}-{int(time.time())}" style="
            position: fixed;
            top: 1rem;
            right: 1rem;
            background: {color["bg"]};
            border: 1px solid {color["border"]};
            color: {color["text"]};
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            max-width: 300px;
            animation: slideIn 0.3s ease-out;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.125rem;">{color["icon"]}</span>
                <span style="font-size: 0.875rem; font-weight: 500;">{message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: none;
                    border: none;
                    color: {color["text"]};
                    cursor: pointer;
                    font-size: 1rem;
                    margin-left: auto;
                    opacity: 0.7;
                ">×</button>
            </div>
        </div>

        <style>
        @keyframes slideIn {{
            from {{
                transform: translateX(100%);
                opacity: 0;
            }}
            to {{
                transform: translateX(0);
                opacity: 1;
            }}
        }}
        </style>

        <script>
        setTimeout(() => {{
            const toast = document.getElementById('toast-{type}-{int(time.time())}');
            if (toast) {{
                toast.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => toast.remove(), 300);
            }}
        }}, {duration});
        </script>
        """

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="🤖 RAG Chat Assistant",
            theme=gr.themes.Default(),
        ) as interface:
            # Header
            gr.Markdown("# 🤖 Simple RAG Chat Assistant")
            gr.Markdown("Ask questions about your documents")

            # Model configuration
            with gr.Row():
                chat_model_dropdown = gr.Dropdown(
                    choices=self.chat_models,
                    value=self.chat_models[0] if self.chat_models else None,
                    label="Chat Model",
                    interactive=True,
                )
                embed_model_dropdown = gr.Dropdown(
                    choices=self.embed_models,
                    value=self.embed_models[0] if self.embed_models else None,
                    label="Embedding Model",
                    interactive=True,
                )

            # System status
            with gr.Row():
                system_info_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                system_info_output = gr.Markdown()

            # Chat interface
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_label=False,
                type="messages",
            )

            # Message input
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask me anything about your documents...",
                    lines=2,
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            # Clear button
            clear_btn = gr.Button("Clear History", variant="secondary")

            # Event handlers
            msg.submit(
                self.chat_interface,
                inputs=[msg, chatbot, chat_model_dropdown, embed_model_dropdown],
                outputs=[msg, chatbot],
            )

            send_btn.click(
                self.chat_interface,
                inputs=[msg, chatbot, chat_model_dropdown, embed_model_dropdown],
                outputs=[msg, chatbot],
            )

            clear_btn.click(
                self.clear_chat_history,
                outputs=[chatbot],
            )

            system_info_btn.click(
                self.get_system_info,
                outputs=[system_info_output],
            )

        return interface

    def _get_realtime_metrics(self) -> str:
        """Generate real-time metrics display."""
        with self.metrics_lock:
            metrics = self.system_metrics

        return f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #60a5fa;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #1e40af;">{metrics.total_queries}</div>
                <div style="font-size: 0.75rem; color: #1e3a8a;">📋 Total Queries</div>
            </div>
            <div style="background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #86efac;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #047857;">{metrics.avg_response_time:.2f}s</div>
                <div style="font-size: 0.75rem; color: #065f46;">⏱️ Avg Response</div>
            </div>
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #fbbf24;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #92400e;">{len(self.conversations)}</div>
                <div style="font-size: 0.75rem; color: #78350f;">💬 Conversations</div>
            </div>
            <div style="background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); padding: 1rem; border-radius: 0.5rem; text-align: center; border: 1px solid #f472b6;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #be185d;">{metrics.documents_indexed}</div>
                <div style="font-size: 0.75rem; color: #9d174d;">📄 Documents</div>
            </div>
        </div>
        <div style="text-align: center; padding: 0.5rem; background: rgba(59, 130, 246, 0.1); border-radius: 0.375rem; margin-top: 0.5rem;">
            <small style="color: #1e40af; font-size: 0.75rem;">🔄 Last updated: {datetime.now().strftime("%H:%M:%S")}</small>
        </div>
        """

    def _get_bookmarks_display(self) -> str:
        """Generate bookmarks display HTML."""
        if not self.bookmarked_results:
            return """
            <div style="text-align: center; padding: 2rem; color: #64748b;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔖</div>
                <p style="margin: 0; font-size: 0.875rem;">No bookmarks yet</p>
                <small style="color: #94a3b8;">Bookmark search results to save them here</small>
            </div>
            """

        bookmarks_html = []
        for i, bookmark in enumerate(self.bookmarked_results[-5:]):  # Show last 5
            content_preview = str(bookmark.get("content", "No content"))[:100]
            bookmarked_at = bookmark.get("bookmarked_at", datetime.now())
            if isinstance(bookmarked_at, datetime):
                time_str = bookmarked_at.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = "Unknown"

            bookmarks_html.append(f"""
            <div style="background: white; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #e5e7eb; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 0.25rem 0; font-size: 0.875rem; color: #374151;">🔖 Bookmark #{i + 1}</h4>
                        <p style="margin: 0; font-size: 0.75rem; color: #6b7280; line-height: 1.4;">
                            {content_preview}...
                        </p>
                        <small style="color: #94a3b8; font-size: 0.7rem;">
                            Saved: {time_str}
                        </small>
                    </div>
                    <button onclick="removeBookmark('{bookmark.get("id", i)}')" style="background: #f87171; color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.7rem; cursor: pointer;">🗑️</button>
                </div>
            </div>
            """)

        return f"""
        <div style="max-height: 300px; overflow-y: auto;">
            {"".join(bookmarks_html)}
        </div>
        <div style="text-align: center; margin-top: 1rem;">
            <small style="color: #64748b; font-size: 0.75rem;">Showing {min(5, len(self.bookmarked_results))} of {len(self.bookmarked_results)} bookmarks</small>
        </div>
        """

    def show_error_with_recovery(
        self,
        error_msg: str,
        error_type: str = "general",
        recovery_suggestions: list[str] | None = None,
    ) -> str:
        """Generate enhanced error interface with recovery suggestions."""
        if recovery_suggestions is None:
            recovery_suggestions = []

        # Default recovery suggestions based on error type
        default_suggestions = {
            "connection": [
                "🔄 Check your internet connection",
                "🔌 Verify server is running",
                "⏰ Wait a moment and try again",
                "🛠️ Contact system administrator",
            ],
            "file": [
                "📁 Check file permissions",
                "💾 Ensure enough disk space",
                "📋 Verify file format is supported",
                "🔄 Try uploading a different file",
            ],
            "model": [
                "🤖 Check if AI models are loaded",
                "🔄 Restart the application",
                "⚙️ Verify model configuration",
                "📞 Contact technical support",
            ],
            "general": [
                "🔄 Refresh the page",
                "⏰ Wait a moment and try again",
                "🧹 Clear browser cache",
                "📞 Report this issue",
            ],
        }

        suggestions = recovery_suggestions or default_suggestions.get(
            error_type,
            default_suggestions["general"],
        )
        suggestions_html = "".join([
            f"<li style='margin: 0.5rem 0; color: #065f46; cursor: pointer; transition: all 0.2s ease;' onmouseover='this.style.color=\"#047857\"' onmouseout='this.style.color=\"#065f46\"'>{suggestion}</li>"
            for suggestion in suggestions
        ])

        return f"""
        <div style="background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #f87171; margin: 1rem 0;">
            <!-- Error Header -->
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #dc2626, #b91c1c); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                    ❌
                </div>
                <div>
                    <h3 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600; font-size: 1.125rem;">Something went wrong</h3>
                    <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">{error_msg}</p>
                </div>
            </div>

            <!-- Recovery Suggestions -->
            <div style="background: rgba(255, 255, 255, 0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #fca5a5;">
                <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                    💡 Try these solutions:
                </h4>
                <ul style="margin: 0; padding-left: 1rem; list-style: none;">
                    {suggestions_html}
                </ul>
            </div>

            <!-- Action Buttons -->
            <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <button onclick="location.reload()" style="
                    background: #059669;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    font-size: 0.75rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                " onmouseover="this.style.background='#047857'" onmouseout="this.style.background='#059669'">
                    🔄 Refresh Page
                </button>
                <button onclick="showErrorReport()" style="
                    background: #2563eb;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    font-size: 0.75rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                " onmouseover="this.style.background='#1d4ed8'" onmouseout="this.style.background='#2563eb'">
                    📋 Copy Error Details
                </button>
                <button onclick="window.open('mailto:support@example.com?subject=RAG Assistant Error&body=Error: {error_msg.replace(" ", "%20")}')" style="
                    background: #7c3aed;
                    color: white;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 0.375rem;
                    font-size: 0.75rem;
                    cursor: pointer;
                    transition: all 0.2s ease;
                " onmouseover="this.style.background='#6d28d9'" onmouseout="this.style.background='#7c3aed'">
                    📧 Report Issue
                </button>
            </div>
        </div>

        <script>
        function showErrorReport() {{
            const errorDetails = `
Error Report - RAG Assistant
============================
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Error: {error_msg}
Type: {error_type}
User Agent: ${{navigator.userAgent}}
URL: ${{window.location.href}}
============================
            `.trim();

            navigator.clipboard.writeText(errorDetails).then(() => {{
                alert('📋 Error details copied to clipboard!\\n\\nYou can now paste this in an email or support ticket.');
            }}).catch(() => {{
                // Fallback for older browsers
                const textarea = document.createElement('textarea');
                textarea.value = errorDetails;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);
                alert('📋 Error details copied to clipboard!');
            }});
        }}
        </script>
        """

    def handle_safe_operation(
        self,
        operation_func,
        operation_name: str,
        fallback_result=None,
        error_type: str = "general",
    ):
        """Safely execute an operation with enhanced error handling."""
        try:
            return operation_func()
        except Exception as e:
            error_msg = f"Failed to {operation_name}: {e!s}"
            logger.exception(error_msg)

            # Return error interface instead of just the fallback
            if isinstance(fallback_result, str):
                return self.show_error_with_recovery(error_msg, error_type)
            return fallback_result

    def get_enhanced_system_metrics(self) -> str:
        """Generate enhanced real-time system metrics dashboard."""
        try:
            current_status = self.rag_agent.get_system_status()

            # Get enhanced metrics
            from datetime import datetime
            import time

            import psutil

            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # AI Model metrics
            chat_model = current_status.get("current_chat_model", "Unknown")
            embed_model = current_status.get("current_embedding_model", "Unknown")

            # Performance metrics
            total_conversations = len(self.conversations)
            total_searches = len(self.search_suggestions)
            total_bookmarks = len(self.bookmarked_results)

            # Calculate uptime (simplified)
            uptime_seconds = int(time.time() - getattr(self, "start_time", time.time()))
            uptime_formatted = (
                f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m"
            )

            return f"""
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #86efac;">

                <!-- Header -->
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
                    <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #059669, #047857); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                        📊
                    </div>
                    <div>
                        <h3 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600; font-size: 1.125rem;">System Performance</h3>
                        <p style="margin: 0; color: #065f46; font-size: 0.875rem;">Last updated: {datetime.now().strftime("%H:%M:%S")}</p>
                    </div>
                </div>

                <!-- Resource Metrics Grid -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">

                    <!-- CPU Usage -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #86efac;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.25rem;">🖥️</span>
                            <span style="font-size: 0.875rem; font-weight: 600; color: #374151;">CPU Usage</span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {"#dc2626" if cpu_percent > 80 else "#f59e0b" if cpu_percent > 60 else "#059669"};">
                            {cpu_percent:.1f}%
                        </div>
                        <div style="width: 100%; height: 4px; background: #e5e7eb; border-radius: 2px; margin-top: 0.5rem;">
                            <div style="width: {cpu_percent}%; height: 100%; background: {"#dc2626" if cpu_percent > 80 else "#f59e0b" if cpu_percent > 60 else "#059669"}; border-radius: 2px;"></div>
                        </div>
                    </div>

                    <!-- Memory Usage -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #86efac;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.25rem;">🧠</span>
                            <span style="font-size: 0.875rem; font-weight: 600; color: #374151;">Memory</span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {"#dc2626" if memory.percent > 80 else "#f59e0b" if memory.percent > 60 else "#059669"};">
                            {memory.percent:.1f}%
                        </div>
                        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                            {memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB
                        </div>
                        <div style="width: 100%; height: 4px; background: #e5e7eb; border-radius: 2px; margin-top: 0.5rem;">
                            <div style="width: {memory.percent}%; height: 100%; background: {"#dc2626" if memory.percent > 80 else "#f59e0b" if memory.percent > 60 else "#059669"}; border-radius: 2px;"></div>
                        </div>
                    </div>

                    <!-- Disk Usage -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #86efac;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.25rem;">💽</span>
                            <span style="font-size: 0.875rem; font-weight: 600; color: #374151;">Disk Space</span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {"#dc2626" if disk.percent > 80 else "#f59e0b" if disk.percent > 60 else "#059669"};">
                            {disk.percent:.1f}%
                        </div>
                        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                            {disk.used // (1024**3):.0f}GB / {disk.total // (1024**3):.0f}GB
                        </div>
                        <div style="width: 100%; height: 4px; background: #e5e7eb; border-radius: 2px; margin-top: 0.5rem;">
                            <div style="width: {disk.percent}%; height: 100%; background: {"#dc2626" if disk.percent > 80 else "#f59e0b" if disk.percent > 60 else "#059669"}; border-radius: 2px;"></div>
                        </div>
                    </div>

                    <!-- System Uptime -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #86efac;">
                        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 1.25rem;">⏱️</span>
                            <span style="font-size: 0.875rem; font-weight: 600; color: #374151;">Uptime</span>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #059669;">
                            {uptime_formatted}
                        </div>
                        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
                            Running smoothly
                        </div>
                    </div>
                </div>

                <!-- AI Model Status -->
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #86efac; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        🤖 AI Model Status
                    </h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
                        <div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">Chat Model</div>
                            <div style="font-size: 0.875rem; font-weight: 600; color: #059669; background: #dcfce7; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block;">
                                🗣️ {chat_model}
                            </div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #6b7280; margin-bottom: 0.25rem;">Embedding Model</div>
                            <div style="font-size: 0.875rem; font-weight: 600; color: #2563eb; background: #dbeafe; padding: 0.25rem 0.5rem; border-radius: 0.25rem; display: inline-block;">
                                🔍 {embed_model}
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Activity Summary -->
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #86efac;">
                    <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem;">
                        📈 Activity Summary
                    </h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                        <div style="text-align: center;">
                            <div style="font-size: 1.25rem; font-weight: bold; color: #059669;">{total_conversations}</div>
                            <div style="font-size: 0.75rem; color: #6b7280;">💬 Conversations</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.25rem; font-weight: bold; color: #2563eb;">{total_searches}</div>
                            <div style="font-size: 0.75rem; color: #6b7280;">🔍 Search Queries</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 1.25rem; font-weight: bold; color: #7c3aed;">{total_bookmarks}</div>
                            <div style="font-size: 0.75rem; color: #6b7280;">🔖 Bookmarks</div>
                        </div>
                    </div>
                </div>
            </div>
            """

        except Exception as e:
            logger.exception(f"Error generating system metrics: {e}")
            return f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); border-radius: 0.75rem; border: 1px solid #f87171;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #dc2626, #b91c1c); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                        ⚠️
                    </div>
                    <div>
                        <h3 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Metrics Error</h3>
                        <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error loading metrics: {e!s}</p>
                    </div>
                </div>
            </div>
            """

    def get_system_health_status(self) -> str:
        """Generate system health and alerts dashboard."""
        try:
            current_status = self.rag_agent.get_system_status()

            # System health checks

            # Check AI models
            chat_model_status = (
                "🟢 Online"
                if current_status.get("current_chat_model") != "Unknown"
                else "🔴 Offline"
            )
            embed_model_status = (
                "🟢 Online"
                if current_status.get("current_embedding_model") != "Unknown"
                else "🔴 Offline"
            )

            # Check vector store
            vector_store_status = (
                "🟢 Connected"
                if hasattr(self.rag_agent, "vector_store")
                else "🔴 Disconnected"
            )

            # Check resource usage
            import psutil

            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)

            memory_status = (
                "🟢 Normal"
                if memory.percent < 80
                else "🟡 High"
                if memory.percent < 90
                else "🔴 Critical"
            )
            cpu_status = (
                "🟢 Normal"
                if cpu_percent < 70
                else "🟡 High"
                if cpu_percent < 85
                else "🔴 Critical"
            )

            # Generate alerts
            alerts = []
            if memory.percent > 85:
                alerts.append("⚠️ High memory usage detected")
            if cpu_percent > 80:
                alerts.append("⚠️ High CPU usage detected")
            if len(self.conversations) > 100:
                alerts.append("💡 Consider archiving old conversations")

            alerts_html = ""
            if alerts:
                alert_items = "".join([
                    f"<li style='margin: 0.25rem 0; color: #d97706;'>{alert}</li>"
                    for alert in alerts
                ])
                alerts_html = f"""
                <div style="background: #fef3c7; border: 1px solid #f59e0b; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #92400e; font-size: 0.875rem; font-weight: 600;">🚨 Active Alerts</h4>
                    <ul style="margin: 0; padding-left: 1rem;">
                        {alert_items}
                    </ul>
                </div>
                """

            return f"""
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #7dd3fc;">

                <!-- Header -->
                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
                    <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #0ea5e9, #0284c7); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                        🛡️
                    </div>
                    <div>
                        <h3 style="margin: 0 0 0.25rem 0; color: #0369a1; font-weight: 600; font-size: 1.125rem;">System Health Status</h3>
                        <p style="margin: 0; color: #0284c7; font-size: 0.875rem;">Real-time system monitoring and alerts</p>
                    </div>
                </div>

                {alerts_html}

                <!-- Health Status Grid -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">

                    <!-- AI Models -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #7dd3fc;">
                        <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600;">🤖 AI Models</h4>
                        <div style="space-y: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; color: #6b7280;">Chat Model</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">{chat_model_status}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.875rem; color: #6b7280;">Embedding Model</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">{embed_model_status}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Database -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #7dd3fc;">
                        <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600;">🗄️ Database</h4>
                        <div style="space-y: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; color: #6b7280;">Vector Store</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">{vector_store_status}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.875rem; color: #6b7280;">Documents</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">🟢 Available</span>
                            </div>
                        </div>
                    </div>

                    <!-- System Resources -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #7dd3fc;">
                        <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600;">💻 Resources</h4>
                        <div style="space-y: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; color: #6b7280;">Memory</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">{memory_status}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.875rem; color: #6b7280;">CPU</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">{cpu_status}</span>
                            </div>
                        </div>
                    </div>

                    <!-- Application Status -->
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #7dd3fc;">
                        <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600;">📱 Application</h4>
                        <div style="space-y: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style="font-size: 0.875rem; color: #6b7280;">Web Interface</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">🟢 Running</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="font-size: 0.875rem; color: #6b7280;">API</span>
                                <span style="font-size: 0.875rem; font-weight: 600;">🟢 Available</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Quick Actions -->
                <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 0.5rem; border: 1px solid #7dd3fc; margin-top: 1rem;">
                    <h4 style="margin: 0 0 0.75rem 0; color: #374151; font-size: 0.875rem; font-weight: 600;">🔧 Quick Actions</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
                        <button style="background: #dbeafe; border: 1px solid #93c5fd; color: #1e40af; padding: 0.5rem 1rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem;">
                            🔄 Restart Models
                        </button>
                        <button style="background: #dcfce7; border: 1px solid #86efac; color: #065f46; padding: 0.5rem 1rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem;">
                            🧹 Clear Cache
                        </button>
                        <button style="background: #fef3c7; border: 1px solid #fbbf24; color: #92400e; padding: 0.5rem 1rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem;">
                            📊 Export Logs
                        </button>
                    </div>
                </div>
            </div>
            """

        except Exception as e:
            logger.exception(f"Error generating health status: {e}")
            return f"""
            <div style="padding: 1.5rem; background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%); border-radius: 0.75rem; border: 1px solid #f87171;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #dc2626, #b91c1c); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: white;">
                        ⚠️
                    </div>
                    <div>
                        <h3 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Health Check Error</h3>
                        <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error loading health status: {e!s}</p>
                    </div>
                </div>
            </div>
            """

    def _get_analytics_dashboard(self) -> str:
        """Generate analytics dashboard HTML."""
        with self.metrics_lock:
            metrics = self.system_metrics

        # Calculate some additional analytics
        conversation_count = len(self.conversations)
        avg_messages_per_conversation = (
            sum(len(conv["messages"]) for conv in self.conversations.values())
            / conversation_count
            if conversation_count > 0
            else 0
        )

        return f"""
        <div style="display: grid; gap: 1rem;">
            <!-- Performance Metrics -->
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #7dd3fc;">
                <h4 style="margin: 0 0 1rem 0; color: #0369a1; display: flex; align-items: center; gap: 0.5rem;">
                    📊 Performance Metrics
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #0369a1;">{metrics.total_queries}</div>
                        <div style="font-size: 0.875rem; color: #075985;">📋 Total Queries</div>
                        <div style="font-size: 0.75rem; color: #0c4a6e; margin-top: 0.25rem;">All time</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #0369a1;">{metrics.avg_response_time:.1f}s</div>
                        <div style="font-size: 0.875rem; color: #075985;">⏱️ Avg Response Time</div>
                        <div style="font-size: 0.75rem; color: #0c4a6e; margin-top: 0.25rem;">Last {metrics.total_queries} queries</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #0369a1;">{metrics.documents_indexed}</div>
                        <div style="font-size: 0.875rem; color: #075985;">📄 Documents Indexed</div>
                        <div style="font-size: 0.75rem; color: #0c4a6e; margin-top: 0.25rem;">In knowledge base</div>
                    </div>
                </div>
            </div>

            <!-- Usage Analytics -->
            <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #86efac;">
                <h4 style="margin: 0 0 1rem 0; color: #047857; display: flex; align-items: center; gap: 0.5rem;">
                    📋 Usage Analytics
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem;">
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #047857;">{conversation_count}</div>
                        <div style="font-size: 0.875rem; color: #065f46;">💬 Active Conversations</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #047857;">{avg_messages_per_conversation:.1f}</div>
                        <div style="font-size: 0.875rem; color: #065f46;">💬 Avg Messages/Conv</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #047857;">{len(self.search_suggestions)}</div>
                        <div style="font-size: 0.875rem; color: #065f46;">💡 Search Suggestions</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 2rem; font-weight: bold; color: #047857;">{len(self.bookmarked_results)}</div>
                        <div style="font-size: 0.875rem; color: #065f46;">🔖 Bookmarked Items</div>
                    </div>
                </div>
            </div>

            <!-- System Health -->
            <div style="background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%); padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #fbbf24;">
                <h4 style="margin: 0 0 1rem 0; color: #92400e; display: flex; align-items: center; gap: 0.5rem;">
                    🔋 System Health
                </h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 1.5rem; color: #059669;">✅</div>
                        <div style="font-size: 0.875rem; color: #92400e;">Vector Store</div>
                        <div style="font-size: 0.75rem; color: #78350f;">Connected</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 1.5rem; color: #059669;">✅</div>
                        <div style="font-size: 0.875rem; color: #92400e;">LLM Service</div>
                        <div style="font-size: 0.75rem; color: #78350f;">Active</div>
                    </div>
                    <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 1.5rem; color: #059669;">✅</div>
                        <div style="font-size: 0.875rem; color: #92400e;">Embeddings</div>
                        <div style="font-size: 0.75rem; color: #78350f;">Ready</div>
                    </div>
                </div>
            </div>
        </div>

        <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 0.5rem;">
            <small style="color: #4338ca; font-size: 0.75rem;">
                🔄 Analytics updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} •
                Refresh rate: Every 30 seconds
            </small>
        </div>
        """

    def _update_search_suggestions(self, query: str) -> str:
        """Update search suggestions based on partial query."""
        if not query or len(query) < 2:
            return ""

        suggestions = self.get_search_suggestions(query)
        if not suggestions:
            return ""

        suggestions_html = []
        for suggestion in suggestions[:3]:  # Show top 3
            suggestions_html.append(f"""
            <button onclick="useSearchSuggestion('{suggestion}')" style="
                background: #dbeafe;
                border: 1px solid #93c5fd;
                color: #1e40af;
                padding: 0.5rem 1rem;
                border-radius: 0.375rem;
                font-size: 0.75rem;
                cursor: pointer;
                margin: 0.25rem;
                transition: all 0.2s;
            " onmouseover="this.style.background='#bfdbfe'" onmouseout="this.style.background='#dbeafe'">
                💡 {suggestion}
            </button>
            """)

        return f"""
        <div style="margin: 0.5rem 0; padding: 0.75rem; background: #f8fafc; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
            <small style="color: #64748b; font-weight: 500;">Suggestions:</small>
            <div style="margin-top: 0.5rem;">
                {"".join(suggestions_html)}
            </div>
        </div>

        <script>
        function useSearchSuggestion(suggestion) {{
            const searchInput = document.querySelector(".search-input textarea");
            if (searchInput) {{
                searchInput.value = suggestion;
                searchInput.focus();
                searchInput.dispatchEvent(new Event("input", {{ bubbles: true }}));
            }}
        }}
        </script>
        """

    def _create_new_conversation(self) -> tuple[list[str], list[dict]]:
        """Create a new conversation and return updated dropdown choices."""
        self.create_conversation()
        conversations = self.get_conversation_list()
        choices = [
            f"{conv['title']} ({conv['message_count']} messages)"
            for conv in conversations
        ]
        return choices, []  # Return empty chat history

    def _export_current_conversation(self) -> None:
        """Export current conversation."""
        if self.current_conversation_id:
            exported = self.export_conversation(self.current_conversation_id, "json")
            # In a real implementation, this would trigger a download
            logger.info("Exported conversation: %d characters", len(exported))

    def _export_bookmarks(self) -> None:
        """Export bookmarks."""
        if self.bookmarked_results:
            {
                "bookmarks": self.bookmarked_results,
                "exported_at": datetime.now().isoformat(),
                "count": len(self.bookmarked_results),
            }
            logger.info("Exported %d bookmarks", len(self.bookmarked_results))

    def _reset_metrics(self) -> str:
        """Reset system metrics."""
        with self.metrics_lock:
            self.system_metrics = SystemMetrics()
            self.system_metrics.last_updated = datetime.now()

        return self._get_analytics_dashboard()


def create_app(qdrant_url: str | None = None) -> gr.Blocks:
    """Create and return the Gradio application interface."""
    interface = GradioRAGInterface(qdrant_url)
    return interface.create_interface()


def launch_app(
    qdrant_url: str | None = None,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    enable_mobile_optimizations: bool = True,
) -> None:
    """Launch the Gradio RAG application."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = create_app(qdrant_url)

    # Try multiple ports if the default is busy
    ports_to_try = [server_port, 7861, 7862, 7863, 7864, 7865]

    for port in ports_to_try:
        try:
            logger.info("Trying to launch Gradio app on %s:%s", server_name, port)
            app.launch(
                share=share,
                server_name=server_name,
                server_port=port,
                show_error=True,
                show_api=False,
                max_threads=1,
                inbrowser=False,
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) and port != ports_to_try[-1]:
                logger.warning("Port %s is busy, trying next port...", port)
                continue
            raise
