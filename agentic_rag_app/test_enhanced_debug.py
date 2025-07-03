#!/usr/bin/env python3
"""
Test Enhanced Debug and Citation System

This script tests the new debug output and enhanced citation system.
"""

import sys
sys.path.append('.')

from src.core.manager_agent_system import ManagerAgentSystem
from src.core.debug_agent_wrapper import create_debug_manager


def test_enhanced_system():
    """Test the enhanced system with debug output."""
    print("🧪 Testing Enhanced Debug and Citation System")
    print("=" * 60)
    
    try:
        # Create debug-enabled system
        base_system = ManagerAgentSystem()
        debug_system = create_debug_manager(base_system)
        
        print("\n✅ Debug system created successfully")
        
        # Test query
        test_query = "what is the donnan effect"
        print(f"\n🔍 Testing query: {test_query}")
        print("-" * 40)
        
        response = debug_system.run_query(test_query)
        
        print("\n📄 Response received:")
        print("=" * 40)
        print(response)
        print("=" * 40)
        
        print("\n✅ Enhanced debug test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_enhanced_system()