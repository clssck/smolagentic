#!/usr/bin/env python3
"""
Quick Agent Health Check
"""

import sys
import time
import traceback

# Add src to path
sys.path.append('src')

try:
    from core.manager_agent_system import ManagerAgentSystem
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def quick_agent_test():
    """Quick test of the complete agent system"""
    print("🚀 Quick Agent Health Check")
    print("=" * 40)
    
    try:
        # Test manager system initialization
        print("\n1. 🔧 Initializing Manager System...")
        start = time.time()
        manager_system = ManagerAgentSystem()
        init_time = time.time() - start
        print(f"   ✅ Initialized in {init_time:.2f}s")
        
        # Test simple queries
        print("\n2. 🧪 Testing Core Functionality...")
        
        test_cases = [
            ("hello", "greeting", 10),
            ("2 + 2", "math", 10),
            ("what is Python", "knowledge", 30),
            ("latest AI news", "research", 60)
        ]
        
        results = {}
        
        for query, test_type, timeout in test_cases:
            print(f"\n   🔍 Testing {test_type}: '{query}'")
            try:
                start = time.time()
                response = manager_system.run_query(query)
                query_time = time.time() - start
                
                if response and len(response) > 5 and query_time < timeout:
                    print(f"      ✅ Success: {len(response)} chars, {query_time:.2f}s")
                    results[test_type] = "✅ PASS"
                else:
                    print(f"      ❌ Failed: {len(response)} chars, {query_time:.2f}s")
                    results[test_type] = "❌ FAIL"
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                results[test_type] = "❌ FAIL"
        
        # Test system status
        print("\n3. 📊 System Status...")
        try:
            status = manager_system.get_status()
            print(f"   📋 Manager Agent: {'✅' if status['manager_agent'] else '❌'}")
            print(f"   📋 Vector Store: {'✅' if status['vector_store'] else '❌'}")
            print(f"   📋 Conversation History: {status['conversation_history_length']} messages")
            results["system_status"] = "✅ PASS"
        except Exception as e:
            print(f"   ❌ Status check failed: {e}")
            results["system_status"] = "❌ FAIL"
        
        # Summary
        print("\n" + "=" * 40)
        print("📊 QUICK TEST RESULTS")
        print("=" * 40)
        
        for test_name, result in results.items():
            print(f"{result} {test_name.upper().replace('_', ' ')}")
        
        passed = sum(1 for r in results.values() if "✅" in r)
        total = len(results)
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({passed/total:.1%})")
        
        if passed == total:
            print("🎉 All agents are HEALTHY! 🎉")
        elif passed >= total * 0.75:
            print("✅ System is in good condition")
        else:
            print("⚠️ System needs attention")
        
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    quick_agent_test()