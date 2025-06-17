"""
Simplified Docling integration for RAG document processing.

Extracts the core functionality from docling_converter.py for use in the RAG pipeline.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

# Fix transformers deprecation warning
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated")

logger = logging.getLogger(__name__)

# Try to import Docling with fallback
try:
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    
    DOCLING_AVAILABLE = True
    logger.info("Docling library available")
except ImportError as e:
    DOCLING_AVAILABLE = False
    logger.warning(f"Docling not available: {e}")
    logger.warning("Document conversion will use fallback text extraction")


class DoclingProcessor:
    """Simplified Docling processor for RAG document ingestion."""
    
    def __init__(self):
        """Initialize the Docling processor."""
        self.converter = None
        self.supported_extensions = {
            ".pdf", ".docx", ".xlsx", ".pptx", 
            ".md", ".html", ".png", ".jpg", ".jpeg",
            ".txt", ".csv"
        }
        
        if DOCLING_AVAILABLE:
            self._setup_converter()
    
    def _setup_converter(self) -> None:
        """Setup DocumentConverter with basic configuration."""
        try:
            # Define supported formats
            allowed_formats = [
                InputFormat.PDF,
                InputFormat.DOCX, 
                InputFormat.XLSX,
                InputFormat.PPTX,
                InputFormat.MD,
                InputFormat.HTML,
                InputFormat.IMAGE,
                InputFormat.CSV
            ]
            
            # Configure PDF processing
            pdf_options = PdfPipelineOptions(
                do_ocr=True,
                do_table_structure=True,
                force_full_page_ocr=False
            )
            
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
            }
            
            self.converter = DocumentConverter(
                allowed_formats=allowed_formats,
                format_options=format_options
            )
            
            logger.info("DocumentConverter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize DocumentConverter: {e}")
            # Try fallback with minimal formats
            try:
                self.converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF, InputFormat.MD]
                )
                logger.warning("Using fallback converter with limited formats")
            except Exception as e2:
                logger.error(f"Fallback converter failed: {e2}")
                self.converter = None
    
    def is_supported(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def convert_to_markdown(self, file_path: Path) -> Optional[str]:
        """
        Convert document to markdown text.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Markdown content as string, or None if conversion failed
        """
        if not DOCLING_AVAILABLE or not self.converter:
            return self._fallback_text_extraction(file_path)
        
        try:
            logger.info(f"Converting {file_path.name} to markdown")
            
            result = self.converter.convert(file_path)
            
            if result.status == ConversionStatus.SUCCESS:
                markdown_content = result.document.export_to_markdown()
                logger.info(f"Successfully converted {file_path.name}")
                return markdown_content
            else:
                logger.error(f"Conversion failed for {file_path}: {result.errors}")
                return self._fallback_text_extraction(file_path)
                
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")
            return self._fallback_text_extraction(file_path)
    
    def _fallback_text_extraction(self, file_path: Path) -> Optional[str]:
        """Fallback text extraction for when Docling is not available."""
        try:
            if file_path.suffix.lower() in {".txt", ".md"}:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                logger.info(f"Extracted text from {file_path.name} using fallback")
                return content
            else:
                logger.warning(f"No fallback available for {file_path.suffix} files")
                return None
        except Exception as e:
            logger.error(f"Fallback extraction failed for {file_path}: {e}")
            return None
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from the file."""
        try:
            stat = file_path.stat()
            return {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": stat.st_size,
                "file_extension": file_path.suffix.lower(),
                "modified_time": stat.st_mtime
            }
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return {"filename": file_path.name, "file_path": str(file_path)}


def get_docling_processor() -> DoclingProcessor:
    """Get a singleton DoclingProcessor instance."""
    if not hasattr(get_docling_processor, '_instance'):
        get_docling_processor._instance = DoclingProcessor()
    return get_docling_processor._instance


def convert_document_to_text(file_path: Path) -> Optional[str]:
    """
    Convenience function to convert a document to text.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Text content or None if conversion failed
    """
    processor = get_docling_processor()
    return processor.convert_to_markdown(file_path)


def is_document_supported(file_path: Path) -> bool:
    """
    Check if a document format is supported.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        True if supported, False otherwise
    """
    processor = get_docling_processor()
    return processor.is_supported(file_path)