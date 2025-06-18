#!/usr/bin/env python3
"""Test script to demonstrate the enhanced file upload functionality.

This script creates some sample files and shows how the validation works.
"""

from pathlib import Path
import tempfile

from ui.gradio_app import GradioRAGInterface


def create_sample_files():
    """Create some sample files for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_files_")
    print(f"Creating sample files in: {temp_dir}")

    # Create a valid text file
    txt_file = Path(temp_dir) / "sample.txt"
    with open(txt_file, "w") as f:
        f.write("This is a sample text document for testing the RAG application.")

    # Create a valid markdown file
    md_file = Path(temp_dir) / "sample.md"
    with open(md_file, "w") as f:
        f.write("""# Sample Markdown Document

This is a **sample** markdown document for testing.

## Features
- File validation
- Progress tracking
- Error handling
""")

    # Create an invalid file type
    invalid_file = Path(temp_dir) / "sample.xyz"
    with open(invalid_file, "w") as f:
        f.write("This file has an unsupported extension.")

    # Create a large file (for size testing)
    large_file = Path(temp_dir) / "large_sample.txt"
    with open(large_file, "w") as f:
        # Write 1MB of data
        f.write("Large file content. " * 50000)

    return temp_dir, [txt_file, md_file, invalid_file, large_file]

def test_file_validation():
    """Test the file validation functionality."""
    print("Testing file validation functionality...")

    # Create sample files
    temp_dir, files = create_sample_files()

    try:
        # Initialize the interface
        interface = GradioRAGInterface()

        # Create mock file objects
        class MockFile:
            def __init__(self, file_path):
                self.name = str(file_path)

        mock_files = [MockFile(f) for f in files]

        # Test validation
        valid_files, errors = interface.validate_files(mock_files)

        print("\nValidation Results:")
        print(f"Valid files: {len(valid_files)}")
        print(f"Errors: {len(errors)}")

        for i, file_info in enumerate(valid_files):
            print(f"  {i+1}. {file_info['path'].name} ({file_info['type']}) - {file_info['size']} bytes")

        for i, error in enumerate(errors):
            print(f"  Error {i+1}: {error}")

        # Test supported file types
        print("\nSupported file types:")
        for ext, desc in interface.supported_file_types.items():
            print(f"  {ext}: {desc}")

        print("\nConfiguration:")
        print(f"  Max file size: {interface.max_file_size / (1024*1024):.1f} MB")
        print(f"  Max files per batch: {interface.max_files_per_batch}")

    except (ImportError, RuntimeError) as e:
        print(f"Error during testing: {e}")

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up test files from: {temp_dir}")

if __name__ == "__main__":
    print("🧪 Testing Enhanced File Upload Functionality")
    print("=" * 50)
    test_file_validation()
    print("\n✅ Testing completed!")
