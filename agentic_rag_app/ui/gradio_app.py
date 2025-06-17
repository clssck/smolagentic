import gradio as gr
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
import json
import os
import tempfile
from pathlib import Path
import shutil
from datetime import datetime
import time
import threading
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

from agents.rag_agent import get_agentic_rag
from vector_store.qdrant_client import get_qdrant_store
from models.factory import get_model_factory
from utils.config_loader import get_config_loader, ModelType
from utils.docling_integration import get_docling_processor, is_document_supported

logger = logging.getLogger(__name__)

class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"

class ChatPreference(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    CREATIVE = "creative"

@dataclass
class UserSettings:
    theme: ThemeMode = ThemeMode.LIGHT
    chat_preference: ChatPreference = ChatPreference.DETAILED
    show_timestamps: bool = True
    enable_animations: bool = True
    auto_save_conversations: bool = True
    search_suggestions: bool = True
    response_streaming: bool = True
    
@dataclass
class ConversationMessage:
    id: str
    role: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
@dataclass
class SystemMetrics:
    active_connections: int = 0
    total_queries: int = 0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    documents_indexed: int = 0
    last_updated: datetime = None

class GradioRAGInterface:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.config = get_config_loader()
        self.model_factory = get_model_factory()
        self.qdrant_store = get_qdrant_store(qdrant_url)
        self.rag_agent = get_agentic_rag(qdrant_url)
        
        # Get available models
        self.chat_models = self.model_factory.list_available_models(ModelType.CHAT)
        self.embed_models = self.model_factory.list_available_models(ModelType.EMBEDDING)
        
        # File handling configuration
        self.supported_file_types = {
            ".pdf": "Portable Document Format",
            ".docx": "Microsoft Word Document", 
            ".xlsx": "Microsoft Excel Spreadsheet",
            ".pptx": "Microsoft PowerPoint Presentation",
            ".md": "Markdown Document",
            ".html": "HTML Document",
            ".txt": "Text Document",
            ".csv": "Comma-Separated Values"
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
        
        # Initialize metrics
        self._update_system_metrics()
    
    def __del__(self):
        """Cleanup temporary files when the instance is destroyed."""
        try:
            if hasattr(self, 'temp_upload_dir') and os.path.exists(self.temp_upload_dir):
                shutil.rmtree(self.temp_upload_dir)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")
    
    def _update_system_metrics(self):
        """Update system metrics in thread-safe manner."""
        with self.metrics_lock:
            try:
                # Update basic metrics
                self.system_metrics.last_updated = datetime.now()
                self.system_metrics.documents_indexed = len(self.qdrant_store.get_all_documents()) if hasattr(self.qdrant_store, 'get_all_documents') else 0
                # Add more metrics as needed
            except Exception as e:
                logger.warning(f"Failed to update system metrics: {e}")
    
    def get_user_settings(self) -> Dict[str, Any]:
        """Get current user settings."""
        return asdict(self.user_settings)
    
    def update_user_settings(self, settings: Dict[str, Any]) -> bool:
        """Update user settings."""
        try:
            for key, value in settings.items():
                if hasattr(self.user_settings, key):
                    setattr(self.user_settings, key, value)
            return True
        except Exception as e:
            logger.error(f"Failed to update user settings: {e}")
            return False
    
    def create_conversation(self, title: str = None) -> str:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            'id': conversation_id,
            'title': title or f"Conversation {len(self.conversations) + 1}",
            'messages': [],
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        self.current_conversation_id = conversation_id
        return conversation_id
    
    def get_conversation_history(self, conversation_id: str = None) -> List[Dict]:
        """Get conversation history."""
        if conversation_id is None:
            conversation_id = self.current_conversation_id
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]['messages']
        return []
    
    def export_conversation(self, conversation_id: str = None, format: str = 'json') -> str:
        """Export conversation in specified format."""
        if conversation_id is None:
            conversation_id = self.current_conversation_id
        if not conversation_id or conversation_id not in self.conversations:
            return "No conversation to export"
        
        conversation = self.conversations[conversation_id]
        
        if format == 'json':
            return json.dumps(conversation, indent=2, default=str)
        elif format == 'markdown':
            md_content = f"# {conversation['title']}\n\n"
            md_content += f"**Created:** {conversation['created_at']}\n\n"
            for msg in conversation['messages']:
                role = "🤖 Assistant" if msg['role'] == 'assistant' else "👤 User"
                md_content += f"## {role}\n\n{msg['content']}\n\n"
            return md_content
        else:
            return str(conversation)
    
    def add_search_suggestion(self, query: str):
        """Add a search suggestion."""
        if query not in self.search_suggestions:
            self.search_suggestions.append(query)
            # Keep only recent 20 suggestions
            if len(self.search_suggestions) > 20:
                self.search_suggestions.pop(0)
    
    def get_search_suggestions(self, partial_query: str) -> List[str]:
        """Get search suggestions based on partial query."""
        if not partial_query:
            return self.search_suggestions[-5:]  # Return recent suggestions
        return [s for s in self.search_suggestions if partial_query.lower() in s.lower()][:5]
    
    def bookmark_result(self, result: Dict[str, Any]) -> bool:
        """Bookmark a search result."""
        try:
            bookmark = {
                'id': str(uuid.uuid4()),
                'content': result,
                'bookmarked_at': datetime.now(),
                'tags': []
            }
            self.bookmarked_results.append(bookmark)
            return True
        except Exception as e:
            logger.error(f"Failed to bookmark result: {e}")
            return False
    
    def get_bookmarked_results(self) -> List[Dict]:
        """Get all bookmarked results."""
        return self.bookmarked_results
    
    def show_typing_indicator(self, show: bool = True):
        """Show or hide typing indicator."""
        self.typing_indicator = show
    
    def chat_interface(self, message: str, history: List[dict], 
                      chat_model: str, embedding_model: str) -> Tuple[str, List[dict]]:
        try:
            # Switch models if changed
            current_status = self.rag_agent.get_system_status()
            if chat_model != current_status["current_chat_model"]:
                self.rag_agent.switch_chat_model(chat_model)
            
            # Get response from agent
            response = self.rag_agent.chat(message)
            
            # Update history with new messages format
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return "", history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def validate_files(self, files: List[gr.File]) -> Tuple[List[Dict], List[str]]:
        """Validate uploaded files for type, size, and format."""
        valid_files = []
        errors = []
        
        if not files:
            errors.append("No files selected for upload")
            return valid_files, errors
        
        if len(files) > self.max_files_per_batch:
            errors.append(f"Too many files. Maximum {self.max_files_per_batch} files allowed per batch")
            return valid_files, errors
        
        for i, file in enumerate(files):
            if file is None:
                continue
                
            try:
                file_path = Path(file.name)
                file_ext = file_path.suffix.lower()
                
                # Check file extension
                if file_ext not in self.supported_file_types:
                    errors.append(f"{file_path.name}: Unsupported file type '{file_ext}'. Supported types: {', '.join(self.supported_file_types.keys())}")
                    continue
                
                # Check file size
                file_size = os.path.getsize(file.name) if os.path.exists(file.name) else 0
                if file_size > self.max_file_size:
                    size_mb = file_size / (1024 * 1024)
                    max_mb = self.max_file_size / (1024 * 1024)
                    errors.append(f"{file_path.name}: File too large ({size_mb:.1f}MB). Maximum size: {max_mb}MB")
                    continue
                
                # Check if file is supported by docling
                if not is_document_supported(file_path):
                    errors.append(f"{file_path.name}: File format not supported by document processor")
                    continue
                
                valid_files.append({
                    "file": file,
                    "path": file_path,
                    "size": file_size,
                    "type": self.supported_file_types[file_ext],
                    "status": "pending"
                })
                
            except Exception as e:
                errors.append(f"Error validating file {i+1}: {str(e)}")
        
        return valid_files, errors
    
    def process_uploaded_files(self, files: List[gr.File], progress=gr.Progress()) -> str:
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
            batch_dir = Path(self.temp_upload_dir) / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                        "type": file_info["type"]
                    })
                    
                    progress_val = 0.1 + (i + 1) * 0.2 / len(valid_files)
                    progress(progress_val, desc=f"Copied {source_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error copying file {file_info['path'].name}: {e}")
                    continue
            
            if not processed_files:
                return """
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            💥
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">File Processing Error</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Failed to copy files for processing.</p>
        </div>
    </div>
</div>
"""
            
            progress(0.3, desc="Starting document ingestion...")
            
            # Process documents through the existing pipeline
            result = self.qdrant_store.ingest_documents(str(batch_dir))
            
            progress(1.0, desc="Document processing complete!")
            
            # Clean up temporary files
            try:
                shutil.rmtree(batch_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")
            
            # Generate detailed status report
            status_html = self._generate_processing_status(processed_files, result, errors)
            
            return status_html
            
        except Exception as e:
            logger.error(f"Error processing uploaded files: {e}")
            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            💥
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">System Error</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error processing files: {str(e)}</p>
        </div>
    </div>
</div>
"""
    
    def _generate_processing_status(self, processed_files: List[Dict], result: Dict[str, Any], errors: List[str]) -> str:
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
                Created <strong>{result.get('documents_count', 0)} documents</strong> • 
                Generated <strong>{result.get('nodes_count', 0)} chunks</strong>
            </p>
"""
        else:
            html += f"""
            <p style="margin: 0; color: {status_color}; font-size: 0.875rem;">
                {result.get('message', 'Unknown error occurred')}
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
                    <span style="font-weight: 500;">{file_info['original_name']}</span>
                    <span style="color: #64748b;">({file_info['type']})</span>
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
            status_icon = "✅" if status == "completed" else "⏳" if status == "processing" else "❌"
            status_color = "#059669" if status == "completed" else "#d97706" if status == "processing" else "#dc2626"
            
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
    
    def clear_files(self) -> Tuple[None, str]:
        """Clear uploaded files and reset status."""
        try:
            # Clear file upload status
            self.uploaded_files_status.clear()
            
            # Clean up any temporary directories
            if hasattr(self, 'temp_upload_dir') and os.path.exists(self.temp_upload_dir):
                for item in os.listdir(self.temp_upload_dir):
                    item_path = os.path.join(self.temp_upload_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
            
            return None, """
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
"""
        except Exception as e:
            logger.error(f"Error clearing files: {e}")
            return None, f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ❌
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Clear Failed</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error clearing files: {str(e)}</p>
        </div>
    </div>
</div>
"""
    
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
                Successfully ingested <strong>{result['documents_count']} documents</strong> 
                (<strong>{result['nodes_count']} chunks</strong>) into collection '<strong>{result['collection']}</strong>'
            </p>
        </div>
    </div>
</div>
"""
            elif result["status"] == "warning":
                return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 0.5rem; border: 1px solid #f59e0b; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #d97706; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ⚠️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #92400e; font-weight: 600;">Warning</h4>
            <p style="margin: 0; color: #78350f; font-size: 0.875rem;">{result['message']}</p>
        </div>
    </div>
</div>
"""
            else:
                return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ❌
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">Ingestion Failed</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">{result['message']}</p>
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
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error during ingestion: {str(e)}</p>
        </div>
    </div>
</div>
"""
    
    def search_knowledge_base(self, query: str, top_k: int, filters: Dict[str, Any] = None) -> str:
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
                score_color = "#059669" if result['score'] > 0.8 else "#d97706" if result['score'] > 0.6 else "#dc2626"
                score_bg = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)" if result['score'] > 0.8 else "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)" if result['score'] > 0.6 else "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                score_border = "#86efac" if result['score'] > 0.8 else "#f59e0b" if result['score'] > 0.6 else "#f87171"
                
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
                    Score: {result['score']:.3f}
                </span>
            </div>
            <p style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem; line-height: 1.5;">
                {result['content'][:300]}{'...' if len(result['content']) > 300 else ''}
            </p>
"""
                
                if result['metadata']:
                    output += f"""
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; color: {score_color}; font-size: 0.75rem; font-weight: 500;">📋 Metadata</summary>
                <pre style="background: rgba(255,255,255,0.5); padding: 0.5rem; border-radius: 0.25rem; margin: 0.5rem 0 0 0; font-size: 0.75rem; overflow-x: auto; color: #374151;">{json.dumps(result['metadata'], indent=2)}</pre>
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
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error searching: {str(e)}</p>
        </div>
    </div>
</div>
"""
    
    def get_system_info(self) -> str:
        try:
            status = self.rag_agent.get_system_status()
            
            # Determine status colors
            collection_status = status['qdrant_collection'].get('status', 'N/A')
            status_color = "#059669" if collection_status == "green" else "#d97706" if collection_status == "yellow" else "#dc2626"
            status_bg = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)" if collection_status == "green" else "linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)" if collection_status == "yellow" else "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
            status_border = "#86efac" if collection_status == "green" else "#f59e0b" if collection_status == "yellow" else "#f87171"
            status_icon = "✅" if collection_status == "green" else "⚠️" if collection_status == "yellow" else "❌"
            
            info = f"""
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
                <div style="margin-bottom: 0.25rem;"><strong>Chat:</strong> <code style="background: rgba(255,255,255,0.7); padding: 0.125rem 0.25rem; border-radius: 0.25rem;">{status['current_chat_model']}</code></div>
                <div><strong>Embedding:</strong> <code style="background: rgba(255,255,255,0.7); padding: 0.125rem 0.25rem; border-radius: 0.25rem;">{status['current_embedding_model']}</code></div>
            </div>
        </div>
        
        <!-- Qdrant Collection -->
        <div style="background: {status_bg}; padding: 0.75rem; border-radius: 0.375rem; border: 1px solid {status_border};">
            <h5 style="margin: 0 0 0.5rem 0; color: {status_color}; font-weight: 600; font-size: 0.875rem; display: flex; align-items: center; gap: 0.5rem;">
                🗄️ Vector Database
            </h5>
            <div style="font-size: 0.8125rem; color: #374151; line-height: 1.5;">
                <div style="margin-bottom: 0.25rem;"><strong>Collection:</strong> {status['qdrant_collection'].get('name', 'N/A')}</div>
                <div style="margin-bottom: 0.25rem;"><strong>Documents:</strong> {status['qdrant_collection'].get('points_count', 'N/A'):,} chunks</div>
                <div style="margin-bottom: 0.25rem;"><strong>Vector Size:</strong> {status['qdrant_collection'].get('vector_size', 'N/A')} dimensions</div>
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
                <strong>Messages:</strong> {status['chat_history_length']} messages stored
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
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #374151;">{', '.join(status['available_chat_models'])}</span>
                </div>
                <div>
                    <strong>Embedding Models:</strong><br>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #374151;">{', '.join(status['available_embedding_models'])}</span>
                </div>
            </div>
        </details>
    </div>
</div>
"""
            return info
            
        except Exception as e:
            return f"""
<div style="padding: 1rem; background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); border-radius: 0.5rem; border: 1px solid #f87171; margin: 0.5rem 0;">
    <div style="display: flex; align-items: center; gap: 0.75rem;">
        <div style="width: 2.5rem; height: 2.5rem; background: #dc2626; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
            ⚠️
        </div>
        <div>
            <h4 style="margin: 0 0 0.25rem 0; color: #b91c1c; font-weight: 600;">System Error</h4>
            <p style="margin: 0; color: #991b1b; font-size: 0.875rem;">Error getting system info: {str(e)}</p>
        </div>
    </div>
</div>
"""
    
    def clear_chat_history(self) -> List[dict]:
        """Enhanced chat history clearing with conversation management."""
        try:
            self.rag_agent.clear_history()
            
            # Clear current conversation if exists
            if self.current_conversation_id and self.current_conversation_id in self.conversations:
                self.conversations[self.current_conversation_id]['messages'] = []
                self.conversations[self.current_conversation_id]['updated_at'] = datetime.now()
            
            return []
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return []
    
    def get_conversation_list(self) -> List[Dict]:
        """Get list of all conversations."""
        return [
            {
                'id': conv_id,
                'title': conv['title'],
                'created_at': conv['created_at'].isoformat(),
                'message_count': len(conv['messages'])
            }
            for conv_id, conv in self.conversations.items()
        ]
    
    def switch_conversation(self, conversation_id: str) -> List[dict]:
        """Switch to a different conversation."""
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            return self.conversations[conversation_id]['messages']
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
            self.conversations[conversation_id]['title'] = new_title
            self.conversations[conversation_id]['updated_at'] = datetime.now()
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
                            <input type="radio" name="theme" value="light" {'checked' if settings['theme'] == 'light' else ''}> 🌞 Light Mode
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="theme" value="dark" {'checked' if settings['theme'] == 'dark' else ''}> 🌙 Dark Mode
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="theme" value="auto" {'checked' if settings['theme'] == 'auto' else ''}> 🔄 Auto (System)
                        </label>
                    </div>
                </div>
                
                <!-- Chat Preferences -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">💬 Chat Preferences</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="chat_style" value="concise" {'checked' if settings['chat_preference'] == 'concise' else ''}> 📝 Concise Responses
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="chat_style" value="detailed" {'checked' if settings['chat_preference'] == 'detailed' else ''}> 📋 Detailed Responses
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="radio" name="chat_style" value="creative" {'checked' if settings['chat_preference'] == 'creative' else ''}> 🎨 Creative Responses
                        </label>
                    </div>
                </div>
                
                <!-- Feature Toggles -->
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #374151; font-size: 0.875rem;">🔧 Features</h4>
                    <div style="display: grid; gap: 0.5rem; font-size: 0.75rem;">
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {'checked' if settings['show_timestamps'] else ''}> ⏰ Show Timestamps
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {'checked' if settings['enable_animations'] else ''}> ✨ Enable Animations
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {'checked' if settings['auto_save_conversations'] else ''}> 💾 Auto-save Conversations
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {'checked' if settings['search_suggestions'] else ''}> 💡 Search Suggestions
                        </label>
                        <label style="display: flex; align-items: center; gap: 0.5rem; color: #4b5563;">
                            <input type="checkbox" {'checked' if settings['response_streaming'] else ''}> 🌊 Response Streaming
                        </label>
                    </div>
                </div>
                
                <!-- Action Buttons -->
                <div style="display: flex; gap: 0.5rem; justify-content: center;">
                    <button onclick="saveSettings()" style="background: #059669; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer;">💾 Save Settings</button>
                    <button onclick="resetSettings()" style="background: #dc2626; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer;">🔄 Reset to Default</button>
                    <button onclick="exportSettings()" style="background: #7c3aed; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.375rem; font-size: 0.875rem; cursor: pointer;">📤 Export</button>
                </div>
            </div>
        </div>
        
        <script>
        function saveSettings() {
            console.log('Saving settings...');
            alert('⚙️ Settings saved successfully!');
        }
        
        function resetSettings() {
            if (confirm('🔄 Reset all settings to default values?')) {
                console.log('Resetting settings...');
                location.reload();
            }
        }
        
        function exportSettings() {
            const settings = {settings};
            navigator.clipboard.writeText(JSON.stringify(settings, null, 2));
            alert('📋 Settings copied to clipboard!');
        }
        </script>
        """
    
    def get_toast_notification(self, message: str, type: str = "info", duration: int = 3000) -> str:
        """Generate toast notification HTML."""
        colors = {
            "info": {"bg": "#dbeafe", "border": "#60a5fa", "text": "#1e40af", "icon": "ℹ️"},
            "success": {"bg": "#dcfce7", "border": "#86efac", "text": "#047857", "icon": "✅"},
            "warning": {"bg": "#fef3c7", "border": "#fbbf24", "text": "#92400e", "icon": "⚠️"},
            "error": {"bg": "#fee2e2", "border": "#f87171", "text": "#dc2626", "icon": "❌"}
        }
        
        color = colors.get(type, colors["info"])
        
        return f"""
        <div id="toast-{type}-{int(time.time())}" style="
            position: fixed;
            top: 1rem;
            right: 1rem;
            background: {color['bg']};
            border: 1px solid {color['border']};
            color: {color['text']};
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            z-index: 9999;
            max-width: 300px;
            animation: slideIn 0.3s ease-out;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.125rem;">{color['icon']}</span>
                <span style="font-size: 0.875rem; font-weight: 500;">{message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: none;
                    border: none;
                    color: {color['text']};
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
        # Custom CSS for modern, professional design with comprehensive mobile responsiveness
        custom_css = """
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* CSS Variables for Design System - Mobile-First Approach */
        :root {
            /* Color System */
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --primary-light: #dbeafe;
            --secondary-color: #64748b;
            --success-color: #059669;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --background-primary: #ffffff;
            --background-secondary: #f8fafc;
            --background-tertiary: #f1f5f9;
            --text-primary: #0f172a;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --border-hover: #cbd5e1;
            
            /* Shadow System */
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
            
            /* Border Radius System */
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-xl: 1rem;
            
            /* Mobile-First Typography Scale */
            --font-xs: 0.75rem;     /* 12px */
            --font-sm: 0.875rem;    /* 14px */
            --font-base: 1rem;      /* 16px */
            --font-lg: 1.125rem;    /* 18px */
            --font-xl: 1.25rem;     /* 20px */
            --font-2xl: 1.5rem;     /* 24px */
            --font-3xl: 1.875rem;   /* 30px */
            --font-4xl: 2.25rem;    /* 36px */
            --font-5xl: 3rem;       /* 48px */
            
            /* Spacing System - Mobile Optimized */
            --space-xs: 0.25rem;    /* 4px */
            --space-sm: 0.5rem;     /* 8px */
            --space-md: 0.75rem;    /* 12px */
            --space-lg: 1rem;       /* 16px */
            --space-xl: 1.5rem;     /* 24px */
            --space-2xl: 2rem;      /* 32px */
            --space-3xl: 3rem;      /* 48px */
            --space-4xl: 4rem;      /* 64px */
            
            /* Touch Target Sizes */
            --touch-target-min: 44px;
            --touch-target-comfortable: 48px;
            --touch-target-large: 56px;
            
            /* Breakpoints (for reference in calculations) */
            --mobile-max: 480px;
            --tablet-max: 768px;
            --desktop-min: 769px;
            
            /* Z-Index Scale */
            --z-dropdown: 1000;
            --z-sticky: 1020;
            --z-fixed: 1030;
            --z-modal-backdrop: 1040;
            --z-modal: 1050;
            --z-popover: 1060;
            --z-tooltip: 1070;
            --z-toast: 1080;
        }

        /* Global Styles - Mobile-First */
        *, *::before, *::after {
            box-sizing: border-box;
        }
        
        html {
            /* Prevent zoom on iOS when focusing inputs */
            -webkit-text-size-adjust: 100%;
            -ms-text-size-adjust: 100%;
            font-size: 16px; /* Base font size for rem calculations */
        }
        
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            min-height: 100vh !important;
            color: var(--text-primary) !important;
            padding: 0 !important;
            margin: 0 !important;
            width: 100vw !important;
            overflow-x: hidden !important;
        }
        
        /* Main Content Area - Mobile-First Responsive */
        .main {
            background: var(--background-primary) !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
            min-height: 100vh !important;
            width: 100% !important;
            position: relative !important;
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .main {
                border-radius: var(--radius-xl) !important;
                box-shadow: var(--shadow-xl) !important;
                margin: var(--space-2xl) !important;
                min-height: calc(100vh - 4rem) !important;
                width: calc(100vw - 4rem) !important;
            }
        }
        
        /* Header Styling - Mobile-First */
        .app-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-hover) 100%) !important;
            color: white !important;
            padding: var(--space-lg) var(--space-md) !important;
            text-align: center !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .app-header {
                padding: var(--space-xl) var(--space-lg) !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .app-header {
                padding: var(--space-2xl) !important;
            }
        }
        
        .app-header::before {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/><circle cx="25" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') !important;
            opacity: 0.3 !important;
        }
        
        .app-header h1 {
            font-size: var(--font-2xl) !important;
            font-weight: 700 !important;
            margin: 0 0 var(--space-sm) 0 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            position: relative !important;
            z-index: 1 !important;
            line-height: 1.2 !important;
        }
        
        .app-header p {
            font-size: var(--font-sm) !important;
            opacity: 0.9 !important;
            margin: 0 !important;
            position: relative !important;
            z-index: 1 !important;
            line-height: 1.4 !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .app-header h1 {
                font-size: var(--font-3xl) !important;
                margin-bottom: var(--space-md) !important;
            }
            
            .app-header p {
                font-size: var(--font-base) !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .app-header h1 {
                font-size: var(--font-4xl) !important;
                margin-bottom: var(--space-lg) !important;
            }
            
            .app-header p {
                font-size: var(--font-lg) !important;
            }
        }
        
        /* Tab Styling - Mobile-First */
        .tab-nav {
            background: var(--background-secondary) !important;
            border-bottom: 1px solid var(--border-color) !important;
            padding: 0 var(--space-sm) !important;
            overflow-x: auto !important;
            -webkit-overflow-scrolling: touch !important;
            scrollbar-width: none !important;
            -ms-overflow-style: none !important;
        }
        
        .tab-nav::-webkit-scrollbar {
            display: none !important;
        }
        
        .tab-nav > div {
            display: flex !important;
            min-width: max-content !important;
        }
        
        .tab-nav button {
            font-weight: 500 !important;
            padding: var(--space-md) var(--space-lg) !important;
            margin: 0 !important;
            border: none !important;
            border-radius: 0 !important;
            background: transparent !important;
            color: var(--text-secondary) !important;
            border-bottom: 3px solid transparent !important;
            transition: all 0.2s ease !important;
            font-size: var(--font-sm) !important;
            min-height: var(--touch-target-min) !important;
            min-width: var(--touch-target-min) !important;
            white-space: nowrap !important;
            cursor: pointer !important;
            -webkit-tap-highlight-color: transparent !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .tab-nav {
                padding: 0 var(--space-lg) !important;
            }
            
            .tab-nav button {
                padding: var(--space-lg) var(--space-xl) !important;
                font-size: var(--font-base) !important;
                min-height: var(--touch-target-comfortable) !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .tab-nav {
                padding: 0 var(--space-2xl) !important;
                overflow-x: visible !important;
            }
            
            .tab-nav button {
                padding: var(--space-lg) var(--space-xl) !important;
                font-size: var(--font-base) !important;
            }
        }
        
        .tab-nav button:hover,
        .tab-nav button:focus {
            background: var(--background-tertiary) !important;
            color: var(--text-primary) !important;
            outline: none !important;
        }
        
        .tab-nav button:active {
            background: var(--border-color) !important;
            transform: scale(0.98) !important;
        }
        
        .tab-nav button.selected {
            background: var(--background-primary) !important;
            color: var(--primary-color) !important;
            border-bottom-color: var(--primary-color) !important;
        }
        
        /* Touch feedback for mobile */
        @media (max-width: 768px) {
            .tab-nav button:active {
                background: var(--primary-light) !important;
                color: var(--primary-color) !important;
            }
        }
        
        /* Card Component - Mobile-First */
        .card {
            background: var(--background-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            box-shadow: var(--shadow-sm) !important;
            transition: all 0.2s ease !important;
            margin-bottom: var(--space-lg) !important;
        }
        
        .card:hover {
            box-shadow: var(--shadow-md) !important;
            border-color: var(--border-hover) !important;
        }
        
        .card-header {
            padding: var(--space-lg) var(--space-lg) var(--space-md) var(--space-lg) !important;
            border-bottom: 1px solid var(--border-color) !important;
            background: var(--background-secondary) !important;
            border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
        }
        
        .card-content {
            padding: var(--space-lg) !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .card {
                border-radius: var(--radius-lg) !important;
            }
            
            .card-header {
                padding: var(--space-xl) var(--space-xl) var(--space-lg) var(--space-xl) !important;
                border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
            }
            
            .card-content {
                padding: var(--space-xl) !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .card-content {
                padding: var(--space-2xl) !important;
            }
        }
        
        /* Chat Interface - Mobile-First */
        .chatbot {
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            box-shadow: var(--shadow-sm) !important;
            background: var(--background-primary) !important;
            height: 400px !important;
            min-height: 300px !important;
        }
        
        .message {
            margin: var(--space-sm) 0 !important;
            padding: var(--space-md) var(--space-lg) !important;
            border-radius: var(--radius-lg) !important;
            position: relative !important;
            max-width: 90% !important;
            word-wrap: break-word !important;
            animation: slideIn 0.3s ease-out !important;
            font-size: var(--font-sm) !important;
            line-height: 1.5 !important;
        }
        
        .message.user {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)) !important;
            color: white !important;
            margin-left: auto !important;
            margin-right: 0 !important;
        }
        
        .message.bot {
            background: var(--background-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-color) !important;
            margin-left: 0 !important;
            margin-right: auto !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .chatbot {
                border-radius: var(--radius-lg) !important;
                height: 450px !important;
            }
            
            .message {
                margin: var(--space-md) 0 !important;
                padding: var(--space-lg) var(--space-xl) !important;
                max-width: 85% !important;
                font-size: var(--font-base) !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .chatbot {
                height: 500px !important;
            }
            
            .message {
                max-width: 80% !important;
            }
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Input Fields - Mobile-First */
        .textbox input, .textbox textarea {
            border: 2px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            padding: var(--space-md) var(--space-lg) !important;
            font-size: var(--font-base) !important;
            transition: all 0.2s ease !important;
            background: var(--background-primary) !important;
            color: var(--text-primary) !important;
            min-height: var(--touch-target-min) !important;
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            appearance: none !important;
            /* Prevent zoom on iOS */
            font-size: 16px !important;
        }
        
        .textbox input:focus, .textbox textarea:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px var(--primary-light) !important;
            outline: none !important;
            transform: none !important;
        }
        
        .textbox label {
            font-weight: 500 !important;
            color: var(--text-primary) !important;
            margin-bottom: var(--space-sm) !important;
            font-size: var(--font-sm) !important;
            display: block !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .textbox input, .textbox textarea {
                font-size: var(--font-sm) !important;
                min-height: var(--touch-target-comfortable) !important;
            }
            
            .textbox label {
                font-size: var(--font-base) !important;
            }
        }
        
        /* Buttons - Mobile-First with Touch Optimization */
        .btn {
            font-weight: 500 !important;
            padding: var(--space-md) var(--space-xl) !important;
            border-radius: var(--radius-md) !important;
            border: none !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            font-size: var(--font-sm) !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: var(--space-sm) !important;
            min-height: var(--touch-target-min) !important;
            min-width: var(--touch-target-min) !important;
            position: relative !important;
            -webkit-tap-highlight-color: transparent !important;
            user-select: none !important;
            -webkit-user-select: none !important;
            -moz-user-select: none !important;
            text-decoration: none !important;
        }
        
        .btn:active {
            transform: scale(0.98) !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .btn {
                padding: var(--space-lg) var(--space-2xl) !important;
                font-size: var(--font-base) !important;
                min-height: var(--touch-target-comfortable) !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .btn {
                padding: var(--space-lg) var(--space-2xl) !important;
            }
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-hover)) !important;
            color: white !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .btn-primary:hover {
            transform: translateY(-1px) !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        .btn-secondary {
            background: var(--background-secondary) !important;
            color: var(--text-secondary) !important;
            border: 2px solid var(--border-color) !important;
        }
        
        .btn-secondary:hover {
            background: var(--background-tertiary) !important;
            border-color: var(--border-hover) !important;
            color: var(--text-primary) !important;
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success-color), #047857) !important;
            color: white !important;
        }
        
        .btn-warning {
            background: linear-gradient(135deg, var(--warning-color), #b45309) !important;
            color: white !important;
        }
        
        .btn-danger {
            background: linear-gradient(135deg, var(--error-color), #b91c1c) !important;
            color: white !important;
        }
        
        /* Dropdowns - Mobile-First */
        .dropdown select {
            border: 2px solid var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            padding: var(--space-md) var(--space-lg) !important;
            background: var(--background-primary) !important;
            color: var(--text-primary) !important;
            font-size: 16px !important; /* Prevent zoom on iOS */
            transition: all 0.2s ease !important;
            min-height: var(--touch-target-min) !important;
            -webkit-appearance: none !important;
            -moz-appearance: none !important;
            appearance: none !important;
            cursor: pointer !important;
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="%23475569" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6,9 12,15 18,9"></polyline></svg>') !important;
            background-repeat: no-repeat !important;
            background-position: right var(--space-md) center !important;
            background-size: 16px !important;
            padding-right: var(--space-3xl) !important;
        }
        
        .dropdown select:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px var(--primary-light) !important;
            outline: none !important;
        }
        
        .dropdown label {
            font-weight: 500 !important;
            color: var(--text-primary) !important;
            margin-bottom: var(--space-sm) !important;
            font-size: var(--font-sm) !important;
            display: block !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .dropdown select {
                font-size: var(--font-sm) !important;
                min-height: var(--touch-target-comfortable) !important;
            }
            
            .dropdown label {
                font-size: var(--font-base) !important;
            }
        }
        
        /* Sliders - Mobile-First */
        .slider input[type="range"] {
            appearance: none !important;
            background: var(--border-color) !important;
            height: 8px !important;
            border-radius: 4px !important;
            outline: none !important;
            width: 100% !important;
            margin: var(--space-sm) 0 !important;
            cursor: pointer !important;
        }
        
        .slider input[type="range"]::-webkit-slider-thumb {
            appearance: none !important;
            width: var(--touch-target-min) !important;
            height: var(--touch-target-min) !important;
            border-radius: 50% !important;
            background: var(--primary-color) !important;
            cursor: pointer !important;
            box-shadow: var(--shadow-md) !important;
            border: 2px solid white !important;
            transition: all 0.2s ease !important;
        }
        
        .slider input[type="range"]::-webkit-slider-thumb:hover {
            background: var(--primary-hover) !important;
            transform: scale(1.1) !important;
        }
        
        .slider input[type="range"]::-webkit-slider-thumb:active {
            transform: scale(1.2) !important;
        }
        
        .slider input[type="range"]::-moz-range-thumb {
            width: var(--touch-target-min) !important;
            height: var(--touch-target-min) !important;
            border-radius: 50% !important;
            background: var(--primary-color) !important;
            cursor: pointer !important;
            box-shadow: var(--shadow-md) !important;
            border: 2px solid white !important;
            transition: all 0.2s ease !important;
        }
        
        .slider label {
            font-weight: 500 !important;
            color: var(--text-primary) !important;
            margin-bottom: var(--space-sm) !important;
            font-size: var(--font-sm) !important;
            display: block !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .slider input[type="range"]::-webkit-slider-thumb {
                width: 24px !important;
                height: 24px !important;
            }
            
            .slider input[type="range"]::-moz-range-thumb {
                width: 24px !important;
                height: 24px !important;
            }
            
            .slider label {
                font-size: var(--font-base) !important;
            }
        }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.5rem !important;
            padding: 0.5rem 1rem !important;
            border-radius: var(--radius-md) !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
        }
        
        .status-success {
            background: #dcfce7 !important;
            color: var(--success-color) !important;
            border: 1px solid #bbf7d0 !important;
        }
        
        .status-warning {
            background: #fef3c7 !important;
            color: var(--warning-color) !important;
            border: 1px solid #fde68a !important;
        }
        
        .status-error {
            background: #fee2e2 !important;
            color: var(--error-color) !important;
            border: 1px solid #fecaca !important;
        }
        
        .status-info {
            background: var(--primary-light) !important;
            color: var(--primary-color) !important;
            border: 1px solid #bfdbfe !important;
        }
        
        /* Progress Bars */
        .progress-bar {
            background: var(--border-color) !important;
            border-radius: var(--radius-sm) !important;
            overflow: hidden !important;
            height: 8px !important;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, var(--primary-color), var(--primary-hover)) !important;
            height: 100% !important;
            transition: width 0.3s ease !important;
        }
        
        /* Advanced Responsive Layout System */
        
        /* Mobile Layout (≤480px) */
        @media (max-width: 480px) {
            .gradio-container {
                font-size: 14px !important;
            }
            
            /* Single column layout for mobile */
            .gradio-container .block {
                margin-bottom: var(--space-lg) !important;
            }
            
            /* Stack columns vertically */
            .gradio-container .wrap {
                flex-direction: column !important;
            }
            
            .gradio-container .wrap > * {
                width: 100% !important;
                max-width: 100% !important;
                flex: none !important;
            }
            
            /* Optimize sidebar for mobile */
            .gradio-container .block:last-child {
                order: -1 !important;
            }
            
            /* Mobile-optimized spacing */
            .card-content {
                padding: var(--space-md) !important;
            }
            
            /* Mobile button sizing */
            .btn {
                width: 100% !important;
                margin-bottom: var(--space-sm) !important;
            }
            
            /* Mobile input optimization */
            .textbox input, .textbox textarea {
                padding: var(--space-lg) !important;
            }
        }
        
        /* Tablet Layout (481px - 768px) */
        @media (min-width: 481px) and (max-width: 768px) {
            .gradio-container {
                font-size: 15px !important;
            }
            
            /* Two column layout for tablets */
            .gradio-container .wrap {
                gap: var(--space-lg) !important;
            }
            
            /* Tablet-optimized spacing */
            .card-content {
                padding: var(--space-xl) !important;
            }
            
            /* Tablet button layout */
            .btn {
                flex: 1 !important;
                min-width: 120px !important;
            }
        }
        
        /* Desktop Layout (>768px) */
        @media (min-width: 769px) {
            .gradio-container {
                font-size: 16px !important;
            }
            
            /* Restore normal button behavior */
            .btn {
                width: auto !important;
                flex: none !important;
            }
            
            /* Desktop hover effects */
            .btn:hover {
                transform: translateY(-1px) !important;
            }
            
            .card:hover {
                transform: translateY(-2px) !important;
            }
        }
        
        /* Touch Device Optimizations */
        @media (hover: none) and (pointer: coarse) {
            /* Remove hover effects on touch devices */
            .btn:hover,
            .card:hover {
                transform: none !important;
            }
            
            /* Larger touch targets */
            .btn {
                min-height: var(--touch-target-comfortable) !important;
            }
            
            /* Improved touch feedback */
            .btn:active {
                background-color: var(--primary-hover) !important;
                transform: scale(0.96) !important;
            }
        }
        
        /* High DPI Display Optimizations */
        @media (-webkit-min-device-pixel-ratio: 2), 
               (min-resolution: 192dpi) {
            .gradio-container {
                -webkit-font-smoothing: antialiased !important;
                -moz-osx-font-smoothing: grayscale !important;
            }
        }
        
        /* Loading States */
        .loading {
            position: relative !important;
            overflow: hidden !important;
        }
        
        .loading::after {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent) !important;
            animation: loading 1.5s infinite !important;
        }
        
        @keyframes loading {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Scroll Styling */
        ::-webkit-scrollbar {
            width: 8px !important;
            height: 8px !important;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--background-secondary) !important;
            border-radius: var(--radius-sm) !important;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--secondary-color) !important;
            border-radius: var(--radius-sm) !important;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-secondary) !important;
        }
        
        /* Focus Outline */
        *:focus {
            outline: 2px solid var(--primary-color) !important;
            outline-offset: 2px !important;
        }
        
        /* Accessibility */
        .sr-only {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        
        /* Mobile-Optimized UI Enhancements */
        .search-results {
            max-height: 400px !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        .status-output {
            max-height: 300px !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .search-results {
                max-height: 500px !important;
            }
            
            .status-output {
                max-height: 400px !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .search-results {
                max-height: 600px !important;
            }
            
            .status-output {
                max-height: 500px !important;
            }
        }
        
        /* Mobile-Optimized File Upload */
        .file-upload {
            border: 2px dashed var(--border-color) !important;
            border-radius: var(--radius-md) !important;
            background: var(--background-secondary) !important;
            transition: all 0.3s ease !important;
            padding: var(--space-lg) !important;
            min-height: 120px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
        }
        
        .file-upload:hover {
            border-color: var(--primary-color) !important;
            background: var(--primary-light) !important;
        }
        
        .file-upload.drag-over {
            border-color: var(--success-color) !important;
            background: #dcfce7 !important;
            transform: scale(1.01) !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .file-upload {
                border-radius: var(--radius-lg) !important;
                padding: var(--space-xl) !important;
                min-height: 150px !important;
            }
            
            .file-upload.drag-over {
                transform: scale(1.02) !important;
            }
        }
        
        /* Touch device optimization */
        @media (hover: none) and (pointer: coarse) {
            .file-upload {
                border-width: 3px !important;
                min-height: 140px !important;
            }
        }
        
        .upload-status {
            max-height: 250px !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        .file-status {
            max-height: 200px !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        .ingestion-status {
            max-height: 200px !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        /* Tablet Enhancement */
        @media (min-width: 481px) {
            .upload-status {
                max-height: 350px !important;
            }
            
            .file-status {
                max-height: 300px !important;
            }
            
            .ingestion-status {
                max-height: 250px !important;
            }
        }
        
        /* Desktop Enhancement */
        @media (min-width: 769px) {
            .upload-status {
                max-height: 400px !important;
            }
            
            .file-status {
                max-height: 350px !important;
            }
            
            .ingestion-status {
                max-height: 300px !important;
            }
        }
        
        /* File Upload Drag and Drop Animation */
        .file-upload-area {
            position: relative !important;
            overflow: hidden !important;
        }
        
        .file-upload-area::before {
            content: '📁 Drag files here or click to browse' !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            color: var(--text-muted) !important;
            font-size: 1.125rem !important;
            font-weight: 500 !important;
            pointer-events: none !important;
            opacity: 0.7 !important;
            z-index: 1 !important;
        }
        
        /* File List Styling */
        .file-list-item {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            padding: 0.75rem !important;
            background: rgba(255,255,255,0.6) !important;
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--border-color) !important;
            margin-bottom: 0.5rem !important;
            transition: all 0.2s ease !important;
        }
        
        .file-list-item:hover {
            background: rgba(255,255,255,0.8) !important;
            border-color: var(--border-hover) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .file-status-icon {
            width: 1.5rem !important;
            height: 1.5rem !important;
            border-radius: 50% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 0.875rem !important;
            color: white !important;
            font-weight: 600 !important;
        }
        
        .file-status-success {
            background: var(--success-color) !important;
        }
        
        .file-status-warning {
            background: var(--warning-color) !important;
        }
        
        .file-status-error {
            background: var(--error-color) !important;
        }
        
        .file-status-processing {
            background: var(--primary-color) !important;
            animation: pulse 1.5s ease-in-out infinite !important;
        }
        
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.5;
            }
        }
        
        /* Progress Bar Enhancements */
        .upload-progress {
            background: var(--background-tertiary) !important;
            border-radius: var(--radius-sm) !important;
            overflow: hidden !important;
            height: 6px !important;
            margin: 0.5rem 0 !important;
        }
        
        .upload-progress-fill {
            background: linear-gradient(90deg, var(--primary-color), var(--success-color)) !important;
            height: 100% !important;
            transition: width 0.3s ease !important;
            border-radius: var(--radius-sm) !important;
        }
        
        /* Toast Notifications (for future enhancement) */
        .toast {
            position: fixed !important;
            top: 2rem !important;
            right: 2rem !important;
            background: var(--background-primary) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-xl) !important;
            padding: 1rem 1.5rem !important;
            z-index: 1000 !important;
            animation: slideInRight 0.3s ease-out !important;
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Footer Styling */
        .app-footer {
            background: var(--background-secondary) !important;
            border-top: 1px solid var(--border-color) !important;
            padding: 2rem !important;
            text-align: center !important;
            margin-top: 2rem !important;
        }
        
        .app-footer p {
            margin: 0 !important;
            color: var(--text-secondary) !important;
            font-size: 0.875rem !important;
        }
        
        .app-footer a {
            color: var(--primary-color) !important;
            text-decoration: none !important;
            font-weight: 500 !important;
        }
        
        .app-footer a:hover {
            text-decoration: underline !important;
        }
        
        /* Enhanced Markdown Styling */
        .markdown h1, .markdown h2, .markdown h3, .markdown h4, .markdown h5, .markdown h6 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        .markdown p {
            color: var(--text-secondary) !important;
            line-height: 1.6 !important;
            margin-bottom: 1rem !important;
        }
        
        .markdown code {
            background: var(--background-tertiary) !important;
            color: var(--text-primary) !important;
            padding: 0.125rem 0.375rem !important;
            border-radius: var(--radius-sm) !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.875em !important;
        }
        
        .markdown pre {
            background: var(--background-tertiary) !important;
            color: var(--text-primary) !important;
            padding: 1rem !important;
            border-radius: var(--radius-md) !important;
            overflow-x: auto !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.875rem !important;
            border: 1px solid var(--border-color) !important;
        }
        
        .markdown blockquote {
            border-left: 4px solid var(--primary-color) !important;
            background: var(--primary-light) !important;
            padding: 1rem !important;
            margin: 1rem 0 !important;
            border-radius: 0 var(--radius-md) var(--radius-md) 0 !important;
        }
        
        .markdown ul, .markdown ol {
            color: var(--text-secondary) !important;
            padding-left: 1.5rem !important;
        }
        
        .markdown li {
            margin-bottom: 0.25rem !important;
        }
        
        /* Performance Optimizations for Mobile */
        * {
            -webkit-font-smoothing: antialiased !important;
            -moz-osx-font-smoothing: grayscale !important;
        }
        
        /* GPU acceleration for smooth animations */
        .gradio-container * {
            transform: translateZ(0) !important;
        }
        
        /* Optimize scrolling performance */
        .gradio-container {
            -webkit-overflow-scrolling: touch !important;
            scroll-behavior: smooth !important;
        }
        
        /* Reduce motion for users who prefer it */
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        
        /* Optimize for dark mode */
        @media (prefers-color-scheme: dark) {
            :root {
                --background-primary: #1e293b;
                --background-secondary: #0f172a;
                --background-tertiary: #334155;
                --text-primary: #f1f5f9;
                --text-secondary: #cbd5e1;
                --text-muted: #94a3b8;
                --border-color: #475569;
                --border-hover: #64748b;
            }
        }
        
        /* Improve contrast for better accessibility */
        @media (prefers-contrast: high) {
            :root {
                --border-color: #000000;
                --text-secondary: var(--text-primary);
            }
        }
        
        /* Enhanced Mobile-Specific Styles */
        
        /* iOS Safari Specific Fixes */
        @supports (-webkit-touch-callout: none) {
            .textbox input,
            .textbox textarea,
            .dropdown select {
                font-size: 16px !important;
                transform: translateZ(0) !important;
            }
        }
        
        /* Android Chrome Specific Fixes */
        @supports ((-webkit-appearance: none) and (not (-moz-appearance: none))) {
            .btn {
                -webkit-appearance: none !important;
                -webkit-touch-callout: none !important;
            }
        }
        
        /* Focus Management for Mobile */
        @media (max-width: 768px) {
            .textbox input:focus,
            .textbox textarea:focus,
            .dropdown select:focus {
                /* Prevent page zoom on focus */
                transform: translateZ(0) !important;
                -webkit-user-select: text !important;
                user-select: text !important;
            }
        }
        
        /* Landscape Orientation Optimizations */
        @media (orientation: landscape) and (max-height: 600px) {
            .app-header {
                padding: var(--space-md) !important;
            }
            
            .app-header h1 {
                font-size: var(--font-xl) !important;
            }
            
            .app-header p {
                font-size: var(--font-xs) !important;
            }
            
            .chatbot {
                height: 250px !important;
            }
        }
        
        /* Portrait Orientation Optimizations */
        @media (orientation: portrait) and (max-width: 480px) {
            .chatbot {
                height: 350px !important;
            }
            
            .card-content {
                padding: var(--space-sm) !important;
            }
        }
        
        /* PWA and Fullscreen Optimizations */
        @media (display-mode: standalone) {
            .gradio-container {
                padding-top: env(safe-area-inset-top) !important;
                padding-bottom: env(safe-area-inset-bottom) !important;
            }
        }
        
        /* Notch and Safe Area Support */
        @supports (padding: max(0px)) {
            .gradio-container {
                padding-left: max(var(--space-md), env(safe-area-inset-left)) !important;
                padding-right: max(var(--space-md), env(safe-area-inset-right)) !important;
            }
        }
        
        /* Enhanced Touch Feedback */
        .btn:active,
        .tab-nav button:active {
            background-color: var(--primary-hover) !important;
            transform: scale(0.96) !important;
            transition: transform 0.1s ease !important;
        }
        
        /* Improved Scrollbar for Mobile */
        @media (max-width: 768px) {
            ::-webkit-scrollbar {
                width: 4px !important;
                height: 4px !important;
            }
            
            ::-webkit-scrollbar-track {
                background: transparent !important;
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--border-color) !important;
                border-radius: 2px !important;
            }
        }
        
        /* Loading States for Mobile */
        .loading-mobile {
            position: relative !important;
            overflow: hidden !important;
        }
        
        .loading-mobile::after {
            content: '' !important;
            position: absolute !important;
            top: 0 !important;
            left: -100% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(90deg, transparent, rgba(37, 99, 235, 0.3), transparent) !important;
            animation: shimmer 1.5s infinite !important;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Enhanced File Upload for Mobile */
        @media (max-width: 768px) {
            .file-upload {
                position: relative !important;
            }
            
            .file-upload::before {
                content: '\1F4F1 Tap to upload files' !important;
                position: absolute !important;
                top: 50% !important;
                left: 50% !important;
                transform: translate(-50%, -50%) !important;
                color: var(--text-muted) !important;
                font-size: var(--font-sm) !important;
                font-weight: 500 !important;
                pointer-events: none !important;
                z-index: 1 !important;
            }
        }
        
        /* Utility Classes for Mobile */
        .mobile-only {
            display: block !important;
        }
        
        .desktop-only {
            display: none !important;
        }
        
        @media (min-width: 769px) {
            .mobile-only {
                display: none !important;
            }
            
            .desktop-only {
                display: block !important;
            }
        }
        
        /* Responsive Grid System */
        .mobile-grid {
            display: flex !important;
            flex-direction: column !important;
            gap: var(--space-lg) !important;
        }
        
        .mobile-sidebar {
            order: 1 !important;
            width: 100% !important;
        }
        
        .mobile-full-width {
            width: 100% !important;
        }
        
        /* Tablet Layout */
        @media (min-width: 481px) {
            .mobile-grid {
                flex-direction: row !important;
                align-items: flex-start !important;
            }
            
            .mobile-sidebar {
                order: 2 !important;
                width: 300px !important;
                flex-shrink: 0 !important;
            }
        }
        
        /* Desktop Layout */
        @media (min-width: 769px) {
            .mobile-grid {
                gap: var(--space-2xl) !important;
            }
            
            .mobile-sidebar {
                width: 350px !important;
            }
        }
        
        /* Enhanced Mobile Navigation */
        .mobile-nav-toggle {
            display: none !important;
            position: fixed !important;
            top: var(--space-lg) !important;
            right: var(--space-lg) !important;
            z-index: var(--z-fixed) !important;
            background: var(--primary-color) !important;
            color: white !important;
            border: none !important;
            border-radius: 50% !important;
            width: var(--touch-target-large) !important;
            height: var(--touch-target-large) !important;
            box-shadow: var(--shadow-lg) !important;
            cursor: pointer !important;
        }
        
        @media (max-width: 480px) {
            .mobile-nav-toggle {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
        }
        
        /* Mobile-Optimized Gradio Components */
        @media (max-width: 768px) {
            .gradio-container .block {
                margin: var(--space-sm) 0 !important;
            }
            
            .gradio-container .form {
                gap: var(--space-sm) !important;
            }
            
            .gradio-container .wrap {
                gap: var(--space-sm) !important;
            }
            
            /* Make file upload more prominent on mobile */
            .gradio-container input[type="file"] {
                padding: var(--space-lg) !important;
                border: 3px dashed var(--primary-color) !important;
                border-radius: var(--radius-lg) !important;
                background: var(--primary-light) !important;
                min-height: var(--touch-target-large) !important;
            }
        }
        
        /* Enhanced Button Groups for Mobile */
        .button-group {
            display: flex !important;
            flex-direction: column !important;
            gap: var(--space-sm) !important;
            width: 100% !important;
        }
        
        .button-group .btn {
            width: 100% !important;
        }
        
        @media (min-width: 481px) {
            .button-group {
                flex-direction: row !important;
            }
            
            .button-group .btn {
                flex: 1 !important;
                width: auto !important;
            }
        }
        
        /* Mobile-Specific Form Enhancements */
        @media (max-width: 768px) {
            .gradio-container label {
                font-size: var(--font-base) !important;
                margin-bottom: var(--space-sm) !important;
            }
            
            .gradio-container .input-wrap {
                margin-bottom: var(--space-lg) !important;
            }
            
            /* Larger touch targets for mobile */
            .gradio-container button,
            .gradio-container select,
            .gradio-container input[type="range"] {
                min-height: var(--touch-target-min) !important;
            }
        }
        
        /* Mobile Loading States */
        .mobile-loading {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            right: 0 !important;
            bottom: 0 !important;
            background: rgba(0, 0, 0, 0.5) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            z-index: var(--z-modal) !important;
        }
        
        .mobile-loading-spinner {
            width: 40px !important;
            height: 40px !important;
            border: 4px solid var(--primary-light) !important;
            border-top: 4px solid var(--primary-color) !important;
            border-radius: 50% !important;
            animation: spin 1s linear infinite !important;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Mobile Accessibility Enhancements */
        @media (max-width: 768px) {
            /* Skip to main content link */
            .skip-to-main {
                position: absolute !important;
                top: -40px !important;
                left: 6px !important;
                background: var(--primary-color) !important;
                color: white !important;
                padding: 8px !important;
                text-decoration: none !important;
                z-index: var(--z-tooltip) !important;
                border-radius: var(--radius-sm) !important;
            }
            
            .skip-to-main:focus {
                top: 6px !important;
            }
            
            /* Ensure focus indicators are visible on mobile */
            *:focus {
                outline: 3px solid var(--primary-color) !important;
                outline-offset: 2px !important;
            }
        }
        
        /* Mobile Error States */
        .mobile-error {
            position: fixed !important;
            bottom: var(--space-lg) !important;
            left: var(--space-md) !important;
            right: var(--space-md) !important;
            background: var(--error-color) !important;
            color: white !important;
            padding: var(--space-lg) !important;
            border-radius: var(--radius-lg) !important;
            box-shadow: var(--shadow-xl) !important;
            z-index: var(--z-toast) !important;
            animation: slideInUp 0.3s ease-out !important;
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(100%);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Final Mobile Optimizations */
        
        /* Improve tap performance */
        .btn, .tab-nav button {
            will-change: transform !important;
        }
        
        /* Optimize rendering for mobile */
        .gradio-container {
            contain: layout style paint !important;
        }
        
        /* Mobile-specific scroll optimizations */
        @media (max-width: 768px) {
            .gradio-container {
                scroll-snap-type: y proximity !important;
            }
            
            .card {
                scroll-snap-align: start !important;
            }
        }
        
        /* Responsive images and media */
        img, video {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Ensure proper text wrapping */
        .gradio-container * {
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        
        /* Mobile keyboard optimizations */
        @media (max-width: 768px) {
            .textbox input:focus,
            .textbox textarea:focus {
                /* Prevent viewport jumping when keyboard appears */
                scroll-margin-top: 100px !important;
            }
        }
        
        /* Final accessibility improvements */
        @media (prefers-reduced-motion: no-preference) {
            .btn {
                transition: all 0.2s ease-in-out !important;
            }
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .btn {
                border: 2px solid currentColor !important;
            }
        }
        """
        
        with gr.Blocks(
            title="Agentic RAG Chat Assistant",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate",
                font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
                font_mono=["JetBrains Mono", "ui-monospace", "monospace"]
            ),
            css=custom_css,
            head="""
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
            <meta name="mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="default">
            <meta name="theme-color" content="#2563eb">
            <meta name="apple-mobile-web-app-title" content="RAG Assistant">
            <link rel="apple-touch-icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🤖%3C/text%3E%3C/svg%3E">
            """
        ) as interface:
            
            # Viewport and mobile optimization meta tags
            gr.HTML("""
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
            <meta name="mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="default">
            <meta name="theme-color" content="#2563eb">
            <meta name="apple-mobile-web-app-title" content="RAG Assistant">
            <style>
                /* Prevent text size adjust and zoom on orientation change */
                html {
                    -webkit-text-size-adjust: none;
                    -moz-text-size-adjust: none;
                    -ms-text-size-adjust: none;
                    text-size-adjust: none;
                }
                
                /* Prevent highlight on tap */
                * {
                    -webkit-tap-highlight-color: transparent;
                    -webkit-touch-callout: none;
                    -webkit-user-select: none;
                    user-select: none;
                }
                
                /* Allow text selection in inputs */
                input, textarea, [contenteditable] {
                    -webkit-user-select: text;
                    user-select: text;
                }
            </style>
            """)
            
            # Modern Header Section
            with gr.Row(elem_classes="app-header"):
                with gr.Column():
                    gr.HTML("""
                    <div style="position: relative; z-index: 1;">
                        <h1 style="margin: 0 0 0.5rem 0; font-size: 2.5rem; font-weight: 700; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            🤖 Agentic RAG Assistant
                        </h1>
                        <p style="margin: 0; font-size: 1.125rem; opacity: 0.9; color: white;">
                            Intelligent document search and conversation powered by LlamaIndex, Qdrant, and LiteLLM
                        </p>
                    </div>
                    """)
            
            with gr.Tabs(elem_classes="tab-nav"):
                # Main Chat Tab
                with gr.TabItem("💬 Chat"):
                    with gr.Row(elem_classes="card-content mobile-grid"):
                        with gr.Column(scale=3, elem_classes="card"):
                            with gr.Group(elem_classes="card-content"):
                                gr.HTML("""
                                <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 0.75rem; border: 1px solid #e2e8f0;">
                                    <div style="width: 3rem; height: 3rem; background: linear-gradient(135deg, #2563eb, #1d4ed8); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem;">
                                        🤖
                                    </div>
                                    <div>
                                        <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a;">AI Assistant</h3>
                                        <p style="margin: 0; font-size: 0.875rem; color: #64748b;">Ready to help with your documents</p>
                                    </div>
                                </div>
                                """)
                                
                                chatbot = gr.Chatbot(
                                    label="Conversation",
                                    height=500,
                                    show_label=False,
                                    container=True,
                                    bubble_full_width=False,
                                    type="messages",
                                    elem_classes="chatbot"
                                )
                                
                                msg = gr.Textbox(
                                    label="Your Message",
                                    placeholder="Ask me anything about your documents... 💭",
                                    lines=2,
                                    max_lines=5,
                                    elem_classes="textbox"
                                )
                                
                                with gr.Row(elem_classes="button-group"):
                                    send_btn = gr.Button(
                                        "📤 Send Message", 
                                        variant="primary", 
                                        elem_classes="btn btn-primary"
                                    )
                                    clear_btn = gr.Button(
                                        "🗑️ Clear History", 
                                        variant="secondary",
                                        elem_classes="btn btn-secondary"
                                    )
                        
                        with gr.Column(scale=1, elem_classes="mobile-sidebar"):
                            # Model Configuration Card
                            with gr.Group(elem_classes="card"):
                                gr.HTML("""
                                <div style="padding: 1rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0.75rem 0.75rem 0 0; border-bottom: 1px solid #e2e8f0; margin: -1px -1px 1rem -1px;">
                                    <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                        ⚙️ Model Configuration
                                    </h3>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="card-content"):
                                    chat_model_dropdown = gr.Dropdown(
                                        choices=self.chat_models,
                                        value=self.chat_models[0] if self.chat_models else None,
                                        label="🧠 Chat Model",
                                        interactive=True,
                                        elem_classes="dropdown"
                                    )
                                    
                                    embed_model_dropdown = gr.Dropdown(
                                        choices=self.embed_models,
                                        value=self.embed_models[0] if self.embed_models else None,
                                        label="🔍 Embedding Model",
                                        interactive=True,
                                        elem_classes="dropdown"
                                    )
                            
                            # System Status Card
                            with gr.Group(elem_classes="card"):
                                gr.HTML("""
                                <div style="padding: 1rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0.75rem 0.75rem 0 0; border-bottom: 1px solid #e2e8f0; margin: -1px -1px 1rem -1px;">
                                    <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                        📊 System Status
                                    </h3>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="card-content"):
                                    system_info_btn = gr.Button(
                                        "🔄 Refresh Status", 
                                        variant="secondary",
                                        elem_classes="btn btn-secondary",
                                        size="sm"
                                    )
                                    system_info_output = gr.Markdown(
                                        elem_classes="status-output"
                                    )
                
                # Document Management Tab
                with gr.TabItem("📚 Documents"):
                    with gr.Row(elem_classes="card-content mobile-grid"):
                        with gr.Column(elem_classes="mobile-full-width"):
                            # File Upload Card
                            with gr.Group(elem_classes="card"):
                                gr.HTML("""
                                <div style="padding: 1rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0.75rem 0.75rem 0 0; border-bottom: 1px solid #e2e8f0; margin: -1px -1px 1rem -1px;">
                                    <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                        📤 Upload Documents
                                    </h3>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="card-content"):
                                    gr.HTML("""
                                    <div style="padding: 1rem; background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid #86efac;">
                                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                                            <div style="width: 2.5rem; height: 2.5rem; background: #059669; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
                                                🎯
                                            </div>
                                            <div>
                                                <h4 style="margin: 0 0 0.25rem 0; color: #047857; font-weight: 600;">Modern File Upload</h4>
                                                <p style="margin: 0; font-size: 0.875rem; color: #065f46;">
                                                    Upload multiple documents directly with drag-and-drop support. Supported formats: 
                                                    <strong>PDF, DOCX, XLSX, PPTX, MD, HTML, TXT, CSV</strong>
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                    """)
                                    
                                    file_upload = gr.File(
                                        label="📁 Select Documents",
                                        file_count="multiple",
                                        file_types=[".pdf", ".docx", ".xlsx", ".pptx", ".md", ".html", ".txt", ".csv"],
                                        height=150,
                                        elem_classes="file-upload"
                                    )
                                    
                                    with gr.Row(elem_classes="button-group"):
                                        upload_btn = gr.Button(
                                            "🚀 Process Files", 
                                            variant="primary", 
                                            size="lg",
                                            elem_classes="btn btn-primary"
                                        )
                                        
                                        clear_files_btn = gr.Button(
                                            "🗑️ Clear Files", 
                                            variant="secondary",
                                            elem_classes="btn btn-secondary"
                                        )
                                    
                                    upload_output = gr.HTML(
                                        label="📋 Processing Status",
                                        elem_classes="upload-status"
                                    )
                            
                            # Legacy Document Ingestion Card (for backward compatibility)
                            with gr.Group(elem_classes="card"):
                                gr.HTML("""
                                <div style="padding: 1rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0.75rem 0.75rem 0 0; border-bottom: 1px solid #e2e8f0; margin: -1px -1px 1rem -1px;">
                                    <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                        📂 Legacy Folder Ingestion
                                    </h3>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="card-content"):
                                    gr.HTML("""
                                    <div style="padding: 1rem; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 0.5rem; margin-bottom: 1rem; border: 1px solid #f59e0b;">
                                        <div style="display: flex; align-items: center; gap: 0.75rem;">
                                            <div style="width: 2.5rem; height: 2.5rem; background: #d97706; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.125rem; color: white;">
                                                ℹ️
                                            </div>
                                            <div>
                                                <h4 style="margin: 0 0 0.25rem 0; color: #92400e; font-weight: 600;">Alternative Method</h4>
                                                <p style="margin: 0; font-size: 0.875rem; color: #78350f;">
                                                    You can still manually place documents in the <code style="background: rgba(255,255,255,0.7); padding: 0.125rem 0.25rem; border-radius: 0.25rem;">test_data/</code> folder and use this button.
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                    """)
                                    
                                    ingest_btn = gr.Button(
                                        "📁 Ingest from test_data/", 
                                        variant="secondary", 
                                        size="sm",
                                        elem_classes="btn btn-secondary"
                                    )
                                    
                                    ingest_output = gr.HTML(
                                        label="📋 Legacy Ingestion Status",
                                        elem_classes="ingestion-status"
                                    )
                            
                            # File Management Status Card
                            with gr.Group(elem_classes="card"):
                                gr.HTML("""
                                <div style="padding: 1rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0.75rem 0.75rem 0 0; border-bottom: 1px solid #e2e8f0; margin: -1px -1px 1rem -1px;">
                                    <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                        📊 File Management
                                    </h3>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="card-content"):
                                    with gr.Row(elem_classes="button-group"):
                                        status_refresh_btn = gr.Button(
                                            "🔄 Refresh Status", 
                                            variant="secondary",
                                            size="sm",
                                            elem_classes="btn btn-secondary"
                                        )
                                        
                                        clear_status_btn = gr.Button(
                                            "🧹 Clear Status", 
                                            variant="secondary",
                                            size="sm",
                                            elem_classes="btn btn-secondary"
                                        )
                                    
                                    file_status_output = gr.HTML(
                                        label="📋 File Status",
                                        elem_classes="file-status"
                                    )
                            
                            # Knowledge Base Search Card
                            with gr.Group(elem_classes="card"):
                                gr.HTML("""
                                <div style="padding: 1rem; background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-radius: 0.75rem 0.75rem 0 0; border-bottom: 1px solid #e2e8f0; margin: -1px -1px 1rem -1px;">
                                    <h3 style="margin: 0; font-size: 1.125rem; font-weight: 600; color: #0f172a; display: flex; align-items: center; gap: 0.5rem;">
                                        🔍 Knowledge Base Search
                                    </h3>
                                </div>
                                """)
                                
                                with gr.Group(elem_classes="card-content"):
                                    with gr.Row(elem_classes="mobile-grid"):
                                        search_query = gr.Textbox(
                                            label="🔎 Search Query",
                                            placeholder="Enter your search query... 🌟",
                                            lines=2,
                                            elem_classes="textbox"
                                        )
                                        
                                        with gr.Column(scale=0, min_width=200, elem_classes="mobile-full-width"):
                                            search_top_k = gr.Slider(
                                                minimum=1,
                                                maximum=20,
                                                value=5,
                                                step=1,
                                                label="📊 Number of Results",
                                                elem_classes="slider"
                                            )
                                    
                                    search_btn = gr.Button(
                                        "🎯 Search Knowledge Base", 
                                        variant="primary",
                                        elem_classes="btn btn-primary"
                                    )
                                    
                                    search_output = gr.Markdown(
                                        label="📖 Search Results",
                                        elem_classes="search-results"
                                    )
            
            # Footer Section
            with gr.Row(elem_classes="app-footer"):
                gr.HTML("""
                <div style="text-align: center; padding: 1rem;">
                    <p style="margin: 0 0 0.5rem 0; color: #64748b; font-size: 0.875rem;">
                        Powered by <strong>LlamaIndex</strong>, <strong>Qdrant</strong>, and <strong>LiteLLM</strong>
                    </p>
                    <p style="margin: 0; color: #64748b; font-size: 0.75rem;">
                        🚀 Agentic RAG Application • Built with modern AI technologies
                    </p>
                    <div class="mobile-only" style="margin-top: 1rem; font-size: 0.75rem; color: #94a3b8;">
                        📱 Optimized for mobile devices
                    </div>
                </div>
                """)
            
            # Event handlers
            msg.submit(
                self.chat_interface,
                inputs=[msg, chatbot, chat_model_dropdown, embed_model_dropdown],
                outputs=[msg, chatbot]
            )
            
            send_btn.click(
                self.chat_interface,
                inputs=[msg, chatbot, chat_model_dropdown, embed_model_dropdown],
                outputs=[msg, chatbot]
            )
            
            clear_btn.click(
                self.clear_chat_history,
                outputs=[chatbot]
            )
            
            ingest_btn.click(
                self.ingest_documents,
                outputs=[ingest_output]
            )
            
            search_btn.click(
                self.search_knowledge_base,
                inputs=[search_query, search_top_k],
                outputs=[search_output]
            )
            
            system_info_btn.click(
                self.get_system_info,
                outputs=[system_info_output]
            )
            
            # New file upload event handlers
            upload_btn.click(
                self.process_uploaded_files,
                inputs=[file_upload],
                outputs=[upload_output]
            )
            
            clear_files_btn.click(
                self.clear_files,
                outputs=[file_upload, upload_output]
            )
            
            status_refresh_btn.click(
                self.get_upload_status,
                outputs=[file_status_output]
            )
            
            clear_status_btn.click(
                self.clear_status,
                outputs=[file_status_output]
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
            <small style="color: #1e40af; font-size: 0.75rem;">🔄 Last updated: {datetime.now().strftime('%H:%M:%S')}</small>
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
            content_preview = str(bookmark.get('content', 'No content'))[:100]
            bookmarked_at = bookmark.get('bookmarked_at', datetime.now())
            if isinstance(bookmarked_at, datetime):
                time_str = bookmarked_at.strftime('%Y-%m-%d %H:%M')
            else:
                time_str = 'Unknown'
                
            bookmarks_html.append(f"""
            <div style="background: white; padding: 0.75rem; border-radius: 0.5rem; border: 1px solid #e5e7eb; margin-bottom: 0.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <h4 style="margin: 0 0 0.25rem 0; font-size: 0.875rem; color: #374151;">🔖 Bookmark #{i+1}</h4>
                        <p style="margin: 0; font-size: 0.75rem; color: #6b7280; line-height: 1.4;">
                            {content_preview}...
                        </p>
                        <small style="color: #94a3b8; font-size: 0.7rem;">
                            Saved: {time_str}
                        </small>
                    </div>
                    <button onclick="removeBookmark('{bookmark.get('id', i)}')" style="background: #f87171; color: white; border: none; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.7rem; cursor: pointer;">🗑️</button>
                </div>
            </div>
            """)
        
        return f"""
        <div style="max-height: 300px; overflow-y: auto;">
            {''.join(bookmarks_html)}
        </div>
        <div style="text-align: center; margin-top: 1rem;">
            <small style="color: #64748b; font-size: 0.75rem;">Showing {min(5, len(self.bookmarked_results))} of {len(self.bookmarked_results)} bookmarks</small>
        </div>
        """
    
    def _get_analytics_dashboard(self) -> str:
        """Generate analytics dashboard HTML."""
        with self.metrics_lock:
            metrics = self.system_metrics
        
        # Calculate some additional analytics
        conversation_count = len(self.conversations)
        avg_messages_per_conversation = (
            sum(len(conv['messages']) for conv in self.conversations.values()) / conversation_count
            if conversation_count > 0 else 0
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
                🔄 Analytics updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • 
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
                {''.join(suggestions_html)}
            </div>
        </div>
        
        <script>
        function useSearchSuggestion(suggestion) {
            const searchInput = document.querySelector('.search-input textarea');
            if (searchInput) {
                searchInput.value = suggestion;
                searchInput.focus();
                searchInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }
        }
        </script>
        """
    
    def _create_new_conversation(self) -> Tuple[List[str], List[dict]]:
        """Create a new conversation and return updated dropdown choices."""
        conversation_id = self.create_conversation()
        conversations = self.get_conversation_list()
        choices = [f"{conv['title']} ({conv['message_count']} messages)" for conv in conversations]
        return choices, []  # Return empty chat history
    
    def _export_current_conversation(self) -> None:
        """Export current conversation."""
        if self.current_conversation_id:
            exported = self.export_conversation(self.current_conversation_id, 'json')
            # In a real implementation, this would trigger a download
            logger.info(f"Exported conversation: {len(exported)} characters")
    
    def _export_bookmarks(self) -> None:
        """Export bookmarks."""
        if self.bookmarked_results:
            export_data = {
                'bookmarks': self.bookmarked_results,
                'exported_at': datetime.now().isoformat(),
                'count': len(self.bookmarked_results)
            }
            logger.info(f"Exported {len(self.bookmarked_results)} bookmarks")
    
    def _reset_metrics(self) -> str:
        """Reset system metrics."""
        with self.metrics_lock:
            self.system_metrics = SystemMetrics()
            self.system_metrics.last_updated = datetime.now()
        
        return self._get_analytics_dashboard()

def create_app(qdrant_url: str = "http://localhost:6333") -> gr.Blocks:
    interface = GradioRAGInterface(qdrant_url)
    return interface.create_interface()

def launch_app(qdrant_url: str = "http://localhost:6333", 
               share: bool = False, 
               server_name: str = "0.0.0.0",
               server_port: int = 7860,
               enable_mobile_optimizations: bool = True):
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    app = create_app(qdrant_url)
    
    # Try multiple ports if the default is busy
    ports_to_try = [server_port, 7861, 7862, 7863, 7864, 7865]
    
    for port in ports_to_try:
        try:
            logger.info(f"Trying to launch Gradio app on {server_name}:{port}")
            app.launch(
                share=share,
                server_name=server_name,
                server_port=port,
                show_error=True,
                show_api=False,
                max_threads=1,
                inbrowser=False
            )
            break
        except OSError as e:
            if "Cannot find empty port" in str(e) and port != ports_to_try[-1]:
                logger.warning(f"Port {port} is busy, trying next port...")
                continue
            else:
                raise