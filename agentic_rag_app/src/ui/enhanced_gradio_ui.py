#!/usr/bin/env python3
"""
Enhanced Gradio UI with Manager Agent System
Based on smolagents examples - clean, fast, and user-friendly
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from smolagents import GradioUI

    GRADIO_UI_AVAILABLE = True
except ImportError:
    GRADIO_UI_AVAILABLE = False
    print("⚠️  smolagents GradioUI not available")

from src.core.manager_agent_system import ManagerAgentSystem


def create_enhanced_gradio_ui(config_path=None, **kwargs):
    """Create enhanced Gradio UI with manager agent system"""

    if not GRADIO_UI_AVAILABLE:
        print("❌ GradioUI not available - falling back to basic web UI")
        return None

    try:
        # Create manager agent system
        print("🚀 Creating manager agent system...")
        system = ManagerAgentSystem(config_path)

        # Get the manager agent
        if not system.manager_agent:
            print("❌ Manager agent not available")
            return None

        # Create GradioUI with the manager agent
        print("🎨 Creating enhanced Gradio interface...")
        ui = GradioUI(
            agent=system.manager_agent, file_upload_folder="./uploads", **kwargs
        )

        print("✅ Enhanced Gradio UI created successfully")
        return ui

    except Exception as e:
        print(f"❌ Failed to create enhanced Gradio UI: {e}")
        return None


def launch_enhanced_ui(
    config_path=None, server_name="127.0.0.1", server_port=7860, share=False, **kwargs
):
    """Launch enhanced Gradio UI"""

    print("🚀 ENHANCED AGENTIC RAG UI")
    print("=" * 50)
    print("🤖 Manager agent with natural routing")
    print("🔍 Research + RAG + Simple QA agents")
    print("⚡ Fast, free Qwen 30B model")
    print("🎨 Enhanced Gradio interface")
    print("=" * 50)

    ui = create_enhanced_gradio_ui(config_path, **kwargs)

    if ui:
        print(f"🌐 Launching on http://{server_name}:{server_port}")
        if share:
            print("🔗 Creating public URL for sharing...")

        ui.launch(server_name=server_name, server_port=server_port, share=share)
    else:
        print("❌ Failed to create UI - check dependencies")


if __name__ == "__main__":
    # Launch with default settings
    launch_enhanced_ui()
