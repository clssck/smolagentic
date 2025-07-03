"""Web UI for the RAG system using smolagents and manager agent system."""

import socket

try:
    from smolagents import GradioUI

    GRADIO_UI_AVAILABLE = True
except ImportError:
    GRADIO_UI_AVAILABLE = False

# Try new refactored system first, fallback to old system
try:
    from src.core.refactored_manager_system import RefactoredManagerSystem as ManagerSystem
    REFACTORED_SYSTEM = True
except ImportError:
    from src.core.manager_agent_system import ManagerAgentSystem as ManagerSystem
    REFACTORED_SYSTEM = False
from src.utils.config import Config


def create_manager_agent():
    """Create a manager agent system for the web UI."""
    Config.validate()

    # Create manager agent system
    system = ManagerSystem()
    
    if REFACTORED_SYSTEM:
        print("✅ Using refactored modular agent system for web UI")
        if not system.manager_agent:
            raise RuntimeError("Failed to initialize refactored manager agent system")
        return system.manager_agent
    else:
        print("✅ Using current manager agent system for web UI")
        if not system.manager_agent:
            raise RuntimeError("Failed to initialize manager agent system")
        return system.manager_agent


def find_available_port(start_port: int, end_port: int = None) -> int:
    """Find an available port in the given range."""
    if end_port is None:
        end_port = start_port + 10

    for port in range(start_port, end_port + 1):
        try:
            # Try to bind to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            continue

    raise OSError(f"No available ports found in range {start_port}-{end_port}")


def launch_web_ui(
    server_port: int = None,
    server_name: str = None,
    share: bool = False,
    file_upload_folder: str = "./uploads",
):
    """Launch the web UI.

    Args:
        server_port: Port for web server (default from config)
        server_name: Host for web server (default from config)
        share: Create public URL
        file_upload_folder: Folder for file uploads
    """
    # Create manager agent
    print("🔧 Initializing Manager Agent System...")
    agent = create_manager_agent()

    # Create smolagents UI
    if not GRADIO_UI_AVAILABLE:
        raise ImportError(
            "smolagents GradioUI not available - please install smolagents"
        )

    print("🎨 Creating Enhanced Gradio UI...")
    ui = GradioUI(
        agent=agent, file_upload_folder=file_upload_folder, reset_agent_memory=False
    )

    # Configure server settings
    server_name = server_name or Config.GRADIO_SERVER_NAME
    default_port = server_port or Config.GRADIO_SERVER_PORT

    # Try to find an available port if the default is in use
    try:
        server_port = find_available_port(default_port)
        if server_port != default_port:
            print(f"⚠️  Port {default_port} is in use, using port {server_port} instead")
    except OSError as e:
        print(f"❌ {e}")
        raise

    print(f"🚀 Launching Enhanced RAG Web UI at http://{server_name}:{server_port}")
    print(
        "✨ Features: Manager agent system, natural routing, file uploads, conversation memory"
    )

    # Launch the UI
    ui.launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_api=False,
        show_error=True,
    )


if __name__ == "__main__":
    launch_web_ui()
