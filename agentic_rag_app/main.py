#!/usr/bin/env python3
"""Main entry point for the Agentic RAG application.
Usage:
  python main.py          # Launch web UI (recommended)
  python main.py --cli    # CLI mode
  python main.py --help   # Show help
"""

import argparse
import sys


def main():
    """Main entry point with simplified options."""
    parser = argparse.ArgumentParser(
        description="Modular Agentic RAG System - Intelligent assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              # Launch web UI (recommended)
  python main.py --cli        # Use command line interface
  python main.py --chat "what is UFDF?"  # Ask single question and exit
  python main.py --port 8080  # Launch on custom port
  python main.py --config my_config.json  # Use custom config

The modular system features:
  🔧 Swappable models, embedders, retrievers
  🎯 Intelligent component selection
  🔍 Multiple agent types (RAG, Research, Code)
  🧠 Conversation memory
        """,
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use CLI instead of web UI",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI (default: 7860)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for web UI (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public URL for sharing",
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom component configuration file",
    )

    parser.add_argument(
        "--agent",
        type=str,
        help="Specific agent to use (rag_agent, simple_qa, research_agent, etc.)",
    )

    parser.add_argument(
        "--chat",
        type=str,
        help="Ask a single question and exit (e.g., --chat 'what is UFDF?')",
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        from src.utils.config import Config

        Config.validate()
        print("✅ Configuration validated")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("💡 Please check your environment variables (.env file)")
        sys.exit(1)

    if args.chat:
        # Handle single chat query
        print(f"🤔 Question: {args.chat}")
        print("=" * 60)

        try:
            # Import and create debug-enabled manager system
            try:
                from src.core.refactored_manager_system import RefactoredManagerSystem
                from src.core.debug_agent_wrapper import create_debug_manager

                config_path = args.config if args.config else None
                base_system = RefactoredManagerSystem(config_path)
                system = create_debug_manager(base_system)
                print("✅ Using refactored system with debug output")
            except ImportError:
                from src.core.manager_agent_system import ManagerAgentSystem
                from src.core.debug_agent_wrapper import create_debug_manager

                config_path = args.config if args.config else None
                base_system = ManagerAgentSystem(config_path)
                system = create_debug_manager(base_system)
                print("✅ Using manager system with debug output")

            # Use specific agent if requested, otherwise use default
            agent_name = args.agent if args.agent else None

            print("🤖 Assistant: ", end="", flush=True)
            response = system.run_query(args.chat, agent_name=agent_name)
            print(response)

        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif args.cli:
        # Launch CLI
        print("🚀 Starting Modular RAG CLI...")

        try:
            # Import and create debug-enabled manager system for CLI
            try:
                from src.core.refactored_manager_system import RefactoredManagerSystem
                from src.core.debug_agent_wrapper import create_debug_manager

                config_path = args.config if args.config else None
                base_system = RefactoredManagerSystem(config_path)
                system = create_debug_manager(base_system)
                print("✅ Using refactored system with debug output")
            except ImportError:
                from src.core.manager_agent_system import ManagerAgentSystem
                from src.core.debug_agent_wrapper import create_debug_manager

                config_path = args.config if args.config else None
                base_system = ManagerAgentSystem(config_path)
                system = create_debug_manager(base_system)
                print("✅ Using manager system with debug output")

            # Show available components
            components = system.list_available_components()
            print("📋 Available components:")
            for comp_type, names in components.items():
                print(f"  {comp_type.title()}: {', '.join(names)}")

            # Show manager agent info
            if args.agent:
                print("🤖 Manager agent system (specific agent routing not needed)")
            else:
                print("🤖 Manager agent system ready")

            print("\n💬 Type your questions (or 'quit' to exit)")
            print("📊 Type 'info' to see system status")
            print("🤖 Manager agent automatically routes to specialized agents")
            print("=" * 60)

            while True:
                query = input("\n🤔 You: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 Goodbye!")
                    break

                if query.lower() == "info":
                    # Show system status
                    try:
                        status = system.get_status()
                        print("\n📊 Manager Agent System Status:")
                        for key, value in status.items():
                            if key != "config":  # Skip detailed config
                                print(f"  {key}: {value}")
                    except Exception as e:
                        print(f"❌ Error getting system info: {e}")
                    continue

                if query.lower().startswith("switch "):
                    # Manager agent system doesn't need manual switching
                    print("💡 Manager agent system automatically handles routing!")
                    continue

                if query:
                    print("\n🤖 Assistant: ", end="", flush=True)
                    try:
                        # Use specific agent if set, otherwise use default
                        agent_name = args.agent if args.agent else None
                        response = system.run_query(query, agent_name=agent_name)
                        print(response)
                    except Exception as e:
                        print(f"❌ Error: {e}")

        except Exception as e:
            print(f"❌ Failed to start CLI: {e}")
            print("💡 Make sure all dependencies and API keys are configured")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        # Launch web UI (default behavior)
        print("🚀 Starting Modular RAG Web UI...")
        print(f"📡 Server: http://{args.host}:{args.port}")
        print("🔧 Modular system with swappable components")
        print("💡 Use --chat 'question' for single queries or --cli for CLI")

        try:
            from src.ui.enhanced_gradio_ui import launch_enhanced_ui

            launch_enhanced_ui(
                server_port=args.port,
                server_name=args.host,
                share=args.share,
            )

        except Exception as e:
            print(f"❌ Failed to start web UI: {e}")
            print("💡 Make sure all dependencies and API keys are configured")
            import traceback

            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
