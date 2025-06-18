"""Main entry point for the Agentic RAG application.

This module sets up logging and launches the Gradio web interface
for the RAG application with reasoning capabilities.
"""

import logging
import os

from dotenv import load_dotenv
from ui.gradio_app import launch_app

# Load environment variables
load_dotenv()

def main() -> None:
    """Initialize and launch the Agentic RAG application.

    Sets up logging, loads environment variables, and starts the
    Gradio web interface with the configured settings.
    """
    # Get settings from environment
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    host = "0.0.0.0"
    port = 7860

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Agentic RAG Application...")
    logger.info("Qdrant URL: %s", qdrant_url)
    logger.info("Server: %s:%s", host, port)

    # Launch the Gradio app
    launch_app(
        qdrant_url=qdrant_url,
        share=False,
        server_name=host,
        server_port=port,
    )

if __name__ == "__main__":
    main()
