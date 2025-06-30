#!/usr/bin/env python3
"""
Demo Script for Modular Agent System

This script demonstrates the new modular agent architecture with individual agent files.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def demo_individual_agents():
    """Demonstrate individual agent capabilities"""
    print("🔧 INDIVIDUAL AGENT DEMONSTRATIONS")
    print("=" * 60)
    
    try:
        from agents.agent_factory import AgentFactory
        from agents.research_agent import ResearchAgent
        from agents.rag_agent import RAGAgent
        from agents.code_agent import CodeAgent
        from agents.simple_agent import SimpleAgent
        from agents.vision_agent import VisionAgent
        
        # Create agent factory
        factory = AgentFactory()
        print("✅ Agent factory created")
        
        # Demo 1: Simple Agent
        print("\n1️⃣  SIMPLE AGENT DEMO")
        print("-" * 30)
        simple_agent = factory.create_agent("simple", name="demo_simple")
        
        simple_queries = ["hello", "what is 5 + 3?", "thank you"]
        for query in simple_queries:
            print(f"Query: {query}")
            result = simple_agent.run(query)
            print(f"Response: {result['response'][:100]}...")
            print(f"Agent: {result['agent_name']}, Time: {result['execution_time']:.2f}s")
            print()
        
        # Demo 2: Research Agent
        print("\n2️⃣  RESEARCH AGENT DEMO")
        print("-" * 30)
        research_agent = factory.create_agent("research", name="demo_research")
        
        research_query = "current AI developments"
        print(f"Query: {research_query}")
        result = research_agent.run(research_query)
        print(f"Response: {result['response'][:200]}...")
        print(f"Agent: {result['agent_name']}, Time: {result['execution_time']:.2f}s")
        print()
        
        # Demo 3: Code Agent
        print("\n3️⃣  CODE AGENT DEMO")
        print("-" * 30)
        code_agent = factory.create_agent("code", name="demo_code")
        
        code_query = "write a Python function to calculate factorial"
        print(f"Query: {code_query}")
        result = code_agent.run(code_query)
        print(f"Response: {result['response'][:300]}...")
        print(f"Agent: {result['agent_name']}, Time: {result['execution_time']:.2f}s")
        if 'code_analysis' in result:
            print(f"Code Analysis: {result['code_analysis']}")
        print()
        
        # Demo 4: RAG Agent (if vector store available)
        print("\n4️⃣  RAG AGENT DEMO")
        print("-" * 30)
        try:
            from vector_store.qdrant_client import QdrantVectorStore
            from utils.config import Config
            
            config = Config()
            vector_store = QdrantVectorStore("agentic_rag", config)
            rag_agent = factory.create_agent("rag", name="demo_rag", vector_store=vector_store)
            
            rag_query = "explain ultrafiltration process"
            print(f"Query: {rag_query}")
            result = rag_agent.run(rag_query)
            print(f"Response: {result['response'][:300]}...")
            print(f"Agent: {result['agent_name']}, Time: {result['execution_time']:.2f}s")
            print()
        except Exception as e:
            print(f"⚠️  RAG agent demo skipped: {e}")
        
        # Demo 5: Vision Agent
        print("\n5️⃣  VISION AGENT DEMO")
        print("-" * 30)
        vision_agent = factory.create_agent("vision", name="demo_vision")
        
        vision_query = "analyze image capabilities"
        print(f"Query: {vision_query}")
        result = vision_agent.run(vision_query)
        print(f"Response: {result['response'][:200]}...")
        print(f"Agent: {result['agent_name']}, Time: {result['execution_time']:.2f}s")
        print()
        
        # Show factory statistics
        print("\n📊 FACTORY STATISTICS")
        print("-" * 30)
        stats = factory.get_factory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        print(f"❌ Individual agent demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_manager_system():
    """Demonstrate manager system with specialized agents"""
    print("\n\n🤖 MANAGER SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    try:
        from core.refactored_manager_system import RefactoredManagerSystem
        
        # Create manager system
        system = RefactoredManagerSystem()
        print("✅ Manager system created")
        
        # Show system status
        status = system.get_status()
        print(f"\nSystem Status:")
        for key, value in status.items():
            if key not in ["factory_stats"]:
                print(f"  {key}: {value}")
        
        # Demo queries for different agents
        demo_queries = [
            ("Simple greeting", "hello there"),
            ("Math calculation", "what is 15 * 7?"),
            ("Code request", "create a Python function for prime numbers"),
            ("Research query", "latest developments in machine learning"),
            ("Knowledge base", "explain diafiltration process"),
            ("Vision task", "describe image analysis capabilities")
        ]
        
        print(f"\n🧪 Testing {len(demo_queries)} queries through manager:")
        print("-" * 50)
        
        for i, (category, query) in enumerate(demo_queries, 1):
            print(f"\n{i}. {category}")
            print(f"   Query: {query}")
            
            try:
                result = system.run_query(query)
                
                if result.get("success", True):
                    response = result["response"]
                    agent_used = result.get("agent_name", "unknown")
                    routing = result.get("routing_decision", "N/A")
                    execution_time = result.get("execution_time", 0.0)
                    
                    print(f"   ✅ Routed to: {routing}")
                    print(f"   🤖 Agent: {agent_used}")
                    print(f"   ⏱️  Time: {execution_time:.2f}s")
                    print(f"   📝 Response: {response[:150]}...")
                else:
                    print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
        
        # Test direct agent access
        print(f"\n🎯 DIRECT AGENT ACCESS DEMO")
        print("-" * 40)
        
        agents = system.list_agents()
        print(f"Available agents: {agents}")
        
        if agents:
            test_agent = agents[0]
            test_query = "direct agent test"
            print(f"\nTesting direct access to '{test_agent}' agent:")
            print(f"Query: {test_query}")
            
            result = system.run_agent_directly(test_agent, test_query)
            print(f"Response: {result.get('response', 'No response')[:100]}...")
        
    except Exception as e:
        print(f"❌ Manager system demo failed: {e}")
        import traceback
        traceback.print_exc()


def demo_agent_factory():
    """Demonstrate agent factory capabilities"""
    print("\n\n🏭 AGENT FACTORY DEMONSTRATION")
    print("=" * 60)
    
    try:
        from agents.agent_factory import AgentFactory, create_manager_system, create_agent
        
        # Demo 1: Basic factory usage
        print("1️⃣  Basic Factory Usage")
        print("-" * 30)
        
        factory = AgentFactory()
        
        # Show available agent types
        available_types = factory.get_available_agent_types()
        print(f"Available agent types: {available_types}")
        
        # Create agents of different types
        for agent_type in available_types[:3]:  # Limit to first 3 for demo
            try:
                agent = factory.create_agent(agent_type, name=f"demo_{agent_type}")
                info = agent.get_info()
                print(f"✅ Created {agent_type}: {info['name']} using {info['model_id']}")
            except Exception as e:
                print(f"❌ Failed to create {agent_type}: {e}")
        
        # Demo 2: Convenience functions
        print(f"\n2️⃣  Convenience Functions")
        print("-" * 30)
        
        # Single agent creation
        simple_agent = create_agent("simple", name="convenience_simple")
        print(f"✅ Created agent via convenience function: {simple_agent.name}")
        
        # Manager system creation
        try:
            manager_system = create_manager_system()
            agents = manager_system.get_available_agents()
            print(f"✅ Created manager system with agents: {agents}")
        except Exception as e:
            print(f"⚠️  Manager system creation failed: {e}")
        
        # Demo 3: Factory statistics
        print(f"\n3️⃣  Factory Statistics")
        print("-" * 30)
        
        stats = factory.get_factory_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Demo 4: Agent information
        print(f"\n4️⃣  Agent Information")
        print("-" * 30)
        
        created_agents = factory.list_agents()
        for agent_name in created_agents[:3]:  # Show first 3
            info = factory.get_agent_info(agent_name)
            if info:
                print(f"Agent: {agent_name}")
                print(f"  Type: {info['type']}")
                print(f"  Class: {info['class']}")
                print(f"  Created: {info['created_at']}")
        
    except Exception as e:
        print(f"❌ Agent factory demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all demonstrations"""
    print("🚀 MODULAR AGENT SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This demo showcases the new modular agent architecture:")
    print("• Individual agent files for better organization")
    print("• Agent factory for easy creation and management")
    print("• Manager system for intelligent routing")
    print("• Backward compatibility with existing interfaces")
    print("=" * 80)
    
    # Run demonstrations
    demo_individual_agents()
    demo_manager_system()
    demo_agent_factory()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETED")
    print("The modular agent system is now properly organized with:")
    print("• src/agents/ - Individual agent implementations")
    print("• Agent factory for centralized management")
    print("• Manager system for intelligent routing")
    print("• Backward compatibility maintained")
    print("=" * 80)


if __name__ == "__main__":
    main()