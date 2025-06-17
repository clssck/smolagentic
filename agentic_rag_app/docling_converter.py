"""Document conversion tool using Docling library with Rich UI enhancements.

This tool converts various document formats (PDF, DOCX, XLSX, PPTX, MD, HTML, images)
to structured Markdown using the Docling library. It supports both single file and
batch directory processing, with optional table extraction to CSV and HTML formats.
"""

# Apply comprehensive SSL bypass BEFORE any other imports
try:
    from comprehensive_ssl_bypass import apply_comprehensive_ssl_bypass

    apply_comprehensive_ssl_bypass()
except ImportError:
    # Note: Logger not yet initialized, so we'll use print here
    print(
        "WARNING: Comprehensive SSL bypass not available - you may encounter certificate errors",
    )

import argparse
from enum import Enum, auto
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

# Fix transformers deprecation warning
import warnings

if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]

# Suppress the specific transformers deprecation warning
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated")

# Optional imports
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

__version__ = "1.1.0"

# Rich imports
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

# Local imports
from space_toggle_menu import SpaceToggleMenu

# Initialize rich console
console = Console()

# Initialize logger with Rich handler
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)
_log.addHandler(RichHandler(console=console, show_time=False))

# Try to import Docling with fallback
try:
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice as PipelineAcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        VlmPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling_config import (
        AcceleratorDevice,
        DoclingConfig,
        get_fast_config,
        get_quality_config,
        get_vlm_config,
    )
    from docling_core.types.doc.base import ImageRefMode

    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    _log.error(f"Docling import error: {e!s}")
    _log.error("Please install Docling first: pip install docling")


class InteractiveArgs:
    """Wrapper for interactive mode arguments to match argparse.Namespace interface."""

    def __init__(self, input_path: str, output_path: str, export_tables: bool) -> None:
        """Initialize interactive arguments.

        Args:
            input_path: Path to input file/directory
            output_path: Path to output directory
            export_tables: Whether to export tables
        """
        self.input = Path(input_path)
        self.output = Path(output_path)
        self.export_tables = export_tables

        # Add missing CLI argument attributes with default values
        self.config = None
        self.preset = None
        self.save_config = None
        self.no_ocr = False
        self.ocr_engine = "easyocr"
        self.ocr_lang = ["en"]
        self.force_full_page_ocr = False
        self.accelerator = None
        self.threads = None
        self.image_scale = 2.0
        self.use_vlm = False
        self.vlm_model = "smoldocling"
        self.vlm_api_url = None
        self.vlm_response_format = "markdown"
        self.export_json = False
        self.export_html = False
        self.export_figures = False
        self.export_yaml = False
        self.export_doctags = False

    def is_file(self) -> bool:
        """Check if input path is a file."""
        return self.input.is_file()

    def is_dir(self) -> bool:
        """Check if input path is a directory."""
        return self.input.is_dir()


def setup_converter(
    config: DoclingConfig = None,
    args: Any = None,
) -> DocumentConverter:
    """Configure DocumentConverter with format-specific backends and advanced options."""
    if not DOCLING_AVAILABLE:
        error_msg = "Docling is not available. Please install it first."
        _log.error(error_msg)
        raise ImportError(error_msg)

    if config is None:
        config = DoclingConfig()

    # Convert format strings to InputFormat enums
    format_mapping = {
        "PDF": InputFormat.PDF,
        "DOCX": InputFormat.DOCX,
        "XLSX": InputFormat.XLSX,
        "PPTX": InputFormat.PPTX,
        "MD": InputFormat.MD,
        "HTML": InputFormat.HTML,
        "IMAGE": InputFormat.IMAGE,
        "ASCIIDOC": InputFormat.ASCIIDOC,
        "CSV": InputFormat.CSV,
    }

    allowed_formats = [
        format_mapping[fmt] for fmt in config.allowed_formats if fmt in format_mapping
    ]

    try:
        # Create format options with advanced pipeline configuration
        format_options = {}

        # Configure PDF processing with advanced options
        if InputFormat.PDF in allowed_formats:
            format_options[InputFormat.PDF] = create_pdf_format_option(args)

        # Use enhanced setup with advanced pipeline options
        converter = DocumentConverter(
            allowed_formats=allowed_formats,
            format_options=format_options if format_options else None,
        )
        _log.info("✓ DocumentConverter initialized successfully with advanced options")
        return converter
    except Exception as e:
        _log.error(f"Failed to initialize DocumentConverter: {e}")
        # Try with minimal formats as fallback
        try:
            converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.MD],
            )
            _log.warning("Using fallback converter with limited formats")
            return converter
        except Exception as e2:
            _log.error(f"Fallback converter also failed: {e2}")
            raise


def create_pdf_format_option(args: Any = None) -> Any:
    """Create PDF format option with advanced pipeline configuration."""
    try:
        # Prepare pipeline options parameters
        pipeline_kwargs = {}

        # Configure OCR settings
        if hasattr(args, "no_ocr") and args.no_ocr:
            pipeline_kwargs["do_ocr"] = False
        else:
            pipeline_kwargs["do_ocr"] = True

        # Configure full page OCR (correct API usage from examples)
        if hasattr(args, "force_full_page_ocr") and args.force_full_page_ocr:
            pipeline_kwargs["force_full_page_ocr"] = True

        # Create pipeline options with parameters
        pipeline_options = PdfPipelineOptions(**pipeline_kwargs)

        # Configure OCR language after creation
        if hasattr(args, "ocr_lang") and args.ocr_lang:
            try:
                pipeline_options.ocr_options.lang = args.ocr_lang
            except AttributeError:
                _log.debug("OCR language configuration not available")

        # Configure image scaling
        if hasattr(args, "image_scale") and args.image_scale:
            try:
                pipeline_options.images_scale = args.image_scale
                pipeline_options.generate_page_images = True
                pipeline_options.generate_picture_images = True
            except AttributeError:
                _log.debug("Image scaling configuration not available")

        # Configure table processing
        pipeline_options.do_table_structure = True
        try:
            pipeline_options.table_structure_options.do_cell_matching = True
        except AttributeError:
            _log.debug("Table cell matching not available")

        # Configure accelerator options
        if hasattr(args, "accelerator") and args.accelerator:
            try:
                device_map = {
                    "auto": PipelineAcceleratorDevice.AUTO,
                    "cpu": PipelineAcceleratorDevice.CPU,
                    "cuda": PipelineAcceleratorDevice.CUDA,
                    "mps": PipelineAcceleratorDevice.MPS,
                }
                device = device_map.get(
                    args.accelerator,
                    PipelineAcceleratorDevice.AUTO,
                )
                threads = getattr(args, "threads", 4)
                pipeline_options.accelerator_options = AcceleratorOptions(
                    num_threads=threads,
                    device=device,
                )
            except (ImportError, AttributeError):
                _log.debug("Accelerator configuration not available")

        # Handle VLM pipeline
        if hasattr(args, "use_vlm") and args.use_vlm:
            try:
                vlm_options = VlmPipelineOptions()
                vlm_options.generate_page_images = True

                return PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=vlm_options,
                )
            except ImportError:
                _log.warning("VLM pipeline not available, using standard pipeline")

        return PdfFormatOption(pipeline_options=pipeline_options)

    except ImportError:
        _log.debug("Advanced pipeline options not available, using basic setup")
        return None


def export_tables(
    document: Any,
    output_dir: Path,
    doc_filename: str,
    config: DoclingConfig | None = None,
) -> None:
    """Export tables from document to CSV and HTML formats."""
    if config is None:
        config = DoclingConfig()

    for table_ix, table in enumerate(document.tables):
        try:
            table_df = table.export_to_dataframe()

            # Save CSV (with UTF-8 encoding to handle Unicode characters)
            if config.export.export_table_csv:
                csv_path = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
                table_df.to_csv(csv_path, encoding="utf-8")
                _log.info(f"✓ Saved table {table_ix + 1} to {csv_path}")

            # Save individual HTML table files (with UTF-8 encoding)
            if config.export.export_table_html:
                html_path = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
                with html_path.open("w", encoding="utf-8") as f:
                    f.write(table.export_to_html(doc=document))
                _log.info(f"✓ Saved table {table_ix + 1} to {html_path}")

        except (OSError, ValueError, AttributeError) as e:
            _log.error(f"✗ Failed to export table {table_ix + 1}: {e!s}")


def export_figures_and_images(
    result: ConversionResult,
    output_dir: Path,
    doc_filename: str,
    config: DoclingConfig | None = None,
) -> None:
    """Export page images, table images, and picture images."""
    if config is None:
        config = DoclingConfig()

    try:
        # Save page images if available
        if hasattr(result.document, "pages"):
            for page_no, page in result.document.pages.items():
                if (
                    hasattr(page, "image")
                    and page.image
                    and hasattr(page.image, "pil_image")
                ):
                    page_image_filename = (
                        output_dir / f"{doc_filename}-page-{page_no}.png"
                    )
                    with page_image_filename.open("wb") as fp:
                        if hasattr(page.image.pil_image, "save"):
                            page.image.pil_image.save(fp, format="PNG")
                        else:
                            _log.debug("PIL image save method not available")
                    _log.info(f"✓ Saved page {page_no} image to {page_image_filename}")

        # Save images of figures and tables
        table_counter = 0
        picture_counter = 0

        if hasattr(result.document, "iterate_items"):
            for element, _ in result.document.iterate_items():
                try:
                    if (
                        hasattr(element, "__class__")
                        and "Table" in element.__class__.__name__
                    ):
                        table_counter += 1
                        element_image_filename = (
                            output_dir / f"{doc_filename}-table-{table_counter}.png"
                        )
                        if hasattr(element, "get_image"):
                            try:
                                image = element.get_image(result.document)
                                if image and hasattr(image, "save"):
                                    with element_image_filename.open("wb") as fp:
                                        image.save(fp, "PNG")
                                else:
                                    _log.debug(
                                        "Image object or save method not available",
                                    )
                                    _log.info(
                                        f"✓ Saved table {table_counter} image to {element_image_filename}",
                                    )
                            except Exception as img_e:
                                _log.debug(f"Could not get image: {img_e}")

                    if (
                        hasattr(element, "__class__")
                        and "Picture" in element.__class__.__name__
                    ):
                        picture_counter += 1
                        element_image_filename = (
                            output_dir / f"{doc_filename}-picture-{picture_counter}.png"
                        )
                        if hasattr(element, "get_image"):
                            try:
                                image = element.get_image(result.document)
                                if image and hasattr(image, "save"):
                                    with element_image_filename.open("wb") as fp:
                                        image.save(fp, "PNG")
                                    _log.info(
                                        f"✓ Saved picture {picture_counter} to {element_image_filename}",
                                    )
                                else:
                                    _log.debug(
                                        "Image object or save method not available",
                                    )
                            except Exception as img_e:
                                _log.debug(f"Could not get image: {img_e}")
                except Exception as e:
                    _log.debug(f"Could not export image for element: {e}")

        # Export markdown with different image reference modes if supported
        if hasattr(result.document, "save_as_markdown"):
            try:
                # Embedded images
                md_embedded = output_dir / f"{doc_filename}-with-images.md"
                result.document.save_as_markdown(
                    md_embedded,
                    image_mode=ImageRefMode.EMBEDDED,
                )
                _log.info(f"✓ Saved markdown with embedded images to {md_embedded}")

                # Referenced images
                md_referenced = output_dir / f"{doc_filename}-with-image-refs.md"
                result.document.save_as_markdown(
                    md_referenced,
                    image_mode=ImageRefMode.REFERENCED,
                )
                _log.info(f"✓ Saved markdown with image references to {md_referenced}")
            except Exception as e:
                _log.debug(f"Could not save markdown with image modes: {e}")

    except (OSError, ValueError, AttributeError) as e:
        _log.error(f"✗ Failed to export figures and images: {e!s}")


def process_file(
    converter: DocumentConverter,
    input_path: Path,
    output_dir: Path,
    export_tables_flag: bool = True,
    config: DoclingConfig | None = None,
    export_yaml_flag: bool = False,
    export_doctags_flag: bool = False,
) -> ConversionResult | None:
    """Convert single file and save results."""
    try:
        _log.info(f"Processing {input_path.name}")
        start_time = time.time()

        result = converter.convert(input_path)
        doc_filename = input_path.stem

        if result.status != ConversionStatus.SUCCESS:
            _log.error(f"✗ Failed to convert {input_path}: {result.errors}")
            return None

        output_dir.mkdir(parents=True, exist_ok=True)

        # Use default config if none provided
        if config is None:
            config = DoclingConfig()

        # Save markdown (always enabled by default)
        if config.export.export_markdown:
            md_path = output_dir / f"{doc_filename}.md"
            with md_path.open("w", encoding="utf-8") as f:
                f.write(result.document.export_to_markdown())
            _log.info(f"✓ Saved markdown to {md_path}")

        # Export JSON
        if config.export.export_json:
            json_path = output_dir / f"{doc_filename}.json"
            with json_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps(result.document.export_to_dict(), indent=2))
            _log.info(f"✓ Saved JSON to {json_path}")

        # Export HTML
        if config.export.export_html:
            html_path = output_dir / f"{doc_filename}.html"
            with html_path.open("w", encoding="utf-8") as f:
                f.write(result.document.export_to_html())
            _log.info(f"✓ Saved HTML to {html_path}")

        # Export text
        if config.export.export_text:
            txt_path = output_dir / f"{doc_filename}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(
                    result.document.export_to_markdown(
                        strict_text=config.export.strict_text,
                    ),
                )
            _log.info(f"✓ Saved text to {txt_path}")

        # Export YAML format
        if export_yaml_flag:
            if YAML_AVAILABLE:
                yaml_path = output_dir / f"{doc_filename}.yaml"
                with yaml_path.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(result.document.export_to_dict(), f)
                _log.info(f"✓ Saved YAML to {yaml_path}")
            else:
                _log.warning("PyYAML not available, skipping YAML export")

        # Export document tokens format
        if export_doctags_flag:
            doctags_path = output_dir / f"{doc_filename}.doctags.txt"
            with doctags_path.open("w", encoding="utf-8") as f:
                f.write(result.document.export_to_doctags())
            _log.info(f"✓ Saved document tokens to {doctags_path}")

        # Export tables if requested
        if (
            (export_tables_flag and config.export.export_tables)
            and hasattr(result.document, "tables")
            and result.document.tables
        ):
            export_tables(result.document, output_dir, doc_filename, config)

        # Export figures and images if requested
        if hasattr(config.export, "export_figures") and config.export.export_figures:
            export_figures_and_images(result, output_dir, doc_filename, config)

        elapsed = time.time() - start_time
        _log.info(f"✓ Completed {input_path.name} in {elapsed:.2f} seconds")
        return result

    except (OSError, ValueError, AttributeError) as e:
        _log.error(f"✗ Error processing {input_path}: {e!s}")
        return None


def process_directory(
    converter: DocumentConverter,
    input_dir: Path,
    output_dir: Path,
    export_tables: bool = True,
    config: DoclingConfig | None = None,
    export_yaml_flag: bool = False,
    export_doctags_flag: bool = False,
) -> dict[str, int]:
    """Batch convert all supported files in directory."""
    results = {
        "success": 0,
        "partial": 0,
        "failed": 0,
    }
    supported_extensions = {
        ".pdf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".md",
        ".html",
        ".png",
        ".jpg",
        ".jpeg",
        ".asciidoc",
        ".adoc",
        ".csv",
    }

    files = [f for f in input_dir.glob("*") if f.suffix.lower() in supported_extensions]
    total_files = len(files)

    if not files:
        _log.warning("⚠ No supported files found in directory")
        return results

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting files...", total=total_files)

        for input_path in files:
            result = process_file(
                converter,
                input_path,
                output_dir,
                export_tables,
                config,
                export_yaml_flag,
                export_doctags_flag,
            )
            progress.update(task, advance=1)

            if result:
                if result.status == ConversionStatus.SUCCESS:
                    results["success"] += 1
                elif result.status == ConversionStatus.PARTIAL_SUCCESS:
                    results["partial"] += 1
            else:
                results["failed"] += 1

    return results


def show_summary(results: dict[str, int]) -> None:
    """Display conversion summary as a rich table."""
    table = Table(title="Conversion Summary", show_header=True)
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="magenta")

    for status, count in results.items():
        table.add_row(status.title(), str(count))

    console.print()
    console.print(Panel(table, title="Results"))


def get_interactive_input() -> InteractiveArgs:
    """Get input parameters interactively using Rich prompts."""
    console.print("[bold]Document Conversion Tool[/bold]", style="blue")
    console.print(Panel("Select input file or directory", style="green"))

    input_path = console.input("📁 Input path: ")
    output_path = console.input("📂 Output directory: ")
    export_tables = console.input("Export tables? (y/n): ").lower() == "y"

    return InteractiveArgs(input_path, output_path, export_tables)


class ConversionMode(Enum):
    """Conversion mode enum (CLI, interactive, or config menu)."""

    CLI = auto()
    INTERACTIVE = auto()
    CONFIG_MENU = auto()


def main() -> None:
    """Main entry point for document conversion tool."""
    # Determine mode based on arguments
    interactive_flags = {"--interactive", "-i"}
    min_args_for_interactive = 2

    if len(sys.argv) == 1:
        mode = ConversionMode.CONFIG_MENU  # Show config menu when no args
    elif len(sys.argv) == min_args_for_interactive and sys.argv[1] in interactive_flags:
        mode = ConversionMode.INTERACTIVE  # Old interactive mode
    else:
        mode = ConversionMode.CLI

    if mode == ConversionMode.CLI:
        parser = argparse.ArgumentParser(
            description="Convert documents to structured Markdown using Docling",
            epilog="""
Examples:
  %(prog)s --input document.pdf --output ./output
  %(prog)s --input ./documents --output ./converted --no-tables
  %(prog)s --input https://arxiv.org/pdf/2408.09869 --output ./papers

Supported formats: PDF, DOCX, XLSX, PPTX, MD, HTML, PNG, JPG, JPEG, ASCIIDOC, CSV
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--input",
            type=Path,
            required=True,
            help="Input file or directory path",
        )
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Output directory path",
        )
        parser.add_argument(
            "--no-tables",
            action="store_false",
            dest="export_tables",
            help="Disable table exports",
        )
        parser.add_argument(
            "--version",
            action="version",
            version=f"%(prog)s {__version__}",
        )

        # Configuration options
        config_group = parser.add_argument_group("Configuration Options")
        config_group.add_argument(
            "--config",
            type=Path,
            help="Path to configuration file (JSON format)",
        )
        config_group.add_argument(
            "--save-config",
            type=Path,
            help="Save current configuration to file and exit",
        )
        config_group.add_argument(
            "--preset",
            choices=["fast", "quality", "vlm"],
            help="Use predefined configuration preset",
        )

        # Processing options
        processing_group = parser.add_argument_group("Processing Options")
        processing_group.add_argument(
            "--no-ocr",
            action="store_true",
            help="Disable OCR processing",
        )
        processing_group.add_argument(
            "--ocr-engine",
            choices=["easyocr", "tesseract", "tesseract-cli", "rapidocr", "ocrmac"],
            default="easyocr",
            help="OCR engine to use",
        )
        processing_group.add_argument(
            "--ocr-lang",
            nargs="+",
            default=["en"],
            help="OCR language(s) to use (e.g., en es fr de auto)",
        )
        processing_group.add_argument(
            "--force-full-page-ocr",
            action="store_true",
            help="Force OCR on entire pages",
        )
        processing_group.add_argument(
            "--accelerator",
            choices=["auto", "cpu", "cuda", "mps"],
            default="auto",
            help="Accelerator device to use",
        )
        processing_group.add_argument(
            "--threads",
            type=int,
            default=4,
            help="Number of threads to use",
        )
        processing_group.add_argument(
            "--image-scale",
            type=float,
            default=2.0,
            help="Image resolution scale factor",
        )

        # VLM (Vision-Language Model) options
        vlm_group = parser.add_argument_group("VLM Options")
        vlm_group.add_argument(
            "--use-vlm",
            action="store_true",
            help="Use Vision-Language Model pipeline",
        )
        vlm_group.add_argument(
            "--vlm-model",
            choices=["smoldocling", "granite-vision", "qwen", "pixtral"],
            default="smoldocling",
            help="VLM model to use",
        )
        vlm_group.add_argument(
            "--vlm-api-url",
            help="API URL for VLM service (e.g., LM Studio, Ollama)",
        )
        vlm_group.add_argument(
            "--vlm-response-format",
            choices=["markdown", "doctags"],
            default="markdown",
            help="VLM response format",
        )

        # Export options
        export_group = parser.add_argument_group("Export Options")
        export_group.add_argument(
            "--export-json",
            action="store_true",
            help="Export to JSON format",
        )
        export_group.add_argument(
            "--export-html",
            action="store_true",
            help="Export to HTML format",
        )
        export_group.add_argument(
            "--export-figures",
            action="store_true",
            help="Export figures and images",
        )
        export_group.add_argument(
            "--export-yaml",
            action="store_true",
            help="Export to YAML format",
        )
        export_group.add_argument(
            "--export-doctags",
            action="store_true",
            help="Export document tokens format",
        )
        args = parser.parse_args()
    elif mode == ConversionMode.CONFIG_MENU:
        # Show interactive configuration menu
        menu = SpaceToggleMenu()
        config = menu.run()

        # Get input/output from user after configuration
        console.print("\n" + "=" * 60)
        console.print("📁 [bold]Input & Output Settings[/bold]")
        console.print("=" * 60)

        input_path = console.input("📁 Input file or directory: ")
        output_path = console.input("📂 Output directory: ")

        # Create a simple args object
        class ConfigMenuArgs:
            def __init__(self, input_path: str, output_path: str) -> None:
                self.input = Path(input_path)
                self.output = Path(output_path)
                self.export_tables = True  # Will be overridden by config

                # Add missing CLI argument attributes with default values
                self.config = None
                self.preset = None
                self.save_config = None
                self.no_ocr = False
                self.ocr_engine = "easyocr"
                self.ocr_lang = ["en"]
                self.force_full_page_ocr = False
                self.accelerator = None
                self.threads = None
                self.image_scale = 2.0
                self.use_vlm = False
                self.vlm_model = "smoldocling"
                self.vlm_api_url = None
                self.vlm_response_format = "markdown"
                self.export_json = False
                self.export_html = False
                self.export_figures = False
                self.export_yaml = False
                self.export_doctags = False

        args = ConfigMenuArgs(input_path, output_path)
    else:
        args = get_interactive_input()
        config = None

    try:
        # Handle configuration (skip if already set by config menu)
        if "config" not in locals() or config is None:
            config = DoclingConfig()

            if hasattr(args, "config") and args.config:
                config = DoclingConfig.load_from_file(args.config)
                _log.info(f"Loaded configuration from {args.config}")
            elif hasattr(args, "preset") and args.preset:
                if args.preset == "fast":
                    config = get_fast_config()
                elif args.preset == "quality":
                    config = get_quality_config()
                elif args.preset == "vlm":
                    config = get_vlm_config()
                _log.info(f"Using {args.preset} preset configuration")

        # Ensure config is not None before applying overrides
        if config is None:
            config = DoclingConfig()

        # Apply command line overrides
        if hasattr(args, "no_ocr") and args.no_ocr:
            config.pipeline.do_ocr = False
        if hasattr(args, "accelerator") and args.accelerator:
            config.accelerator.device = AcceleratorDevice(args.accelerator)
        if hasattr(args, "threads") and args.threads:
            config.accelerator.num_threads = args.threads
        if hasattr(args, "export_json") and args.export_json:
            config.export.export_json = True
        if hasattr(args, "export_html") and args.export_html:
            config.export.export_html = True
        if hasattr(args, "export_figures") and args.export_figures:
            config.export.export_figures = True
        # Store custom export flags for later use
        export_yaml_flag = hasattr(args, "export_yaml") and args.export_yaml
        export_doctags_flag = hasattr(args, "export_doctags") and args.export_doctags

        # Handle save config and exit
        if hasattr(args, "save_config") and args.save_config:
            config.save_to_file(args.save_config)
            _log.info(f"Configuration saved to {args.save_config}")
            return

        converter = setup_converter(config, args)

        if args.input.is_file():
            result = process_file(
                converter,
                args.input,
                args.output,
                args.export_tables,
                config,
                export_yaml_flag,
                export_doctags_flag,
            )
            if result:
                show_summary({
                    "success": 1 if result.status == ConversionStatus.SUCCESS else 0,
                    "partial": 1
                    if result.status == ConversionStatus.PARTIAL_SUCCESS
                    else 0,
                })
        elif args.input.is_dir():
            results = process_directory(
                converter,
                args.input,
                args.output,
                args.export_tables,
                config,
                export_yaml_flag,
                export_doctags_flag,
            )
            show_summary(results)
        else:
            _log.error(f"✗ Input path does not exist: {args.input}")
            sys.exit(1)
    except ImportError as e:
        _log.error(str(e))
        _log.error("Please install Docling first: pip install docling")
        sys.exit(1)
    except (OSError, ValueError, RuntimeError) as e:
        _log.error(f"✗ Error during conversion: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
