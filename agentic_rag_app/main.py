import os
from dotenv import load_dotenv
import logging

from ui.gradio_app import launch_app

# Load environment variables
load_dotenv()

def main():
    # Get settings from environment
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    host = "0.0.0.0"
    port = 7860
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Agentic RAG Application...")
    logger.info(f"Qdrant URL: {qdrant_url}")
    logger.info(f"Server: {host}:{port}")
    
    # Launch the Gradio app
    launch_app(
        qdrant_url=qdrant_url,
        share=False,
        server_name=host,
        server_port=port
    )

if __name__ == "__main__":
    main()