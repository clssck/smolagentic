#!/usr/bin/env python3
"""
Agent Health Test Suite
Test each agent individually for proper functionality
"""

import sys
import time
import traceback
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from core.manager_agent_system import ManagerAgentSystem
    from agents.research_agent import ResearchAgent
    from agents.rag_agent import RAGAgent
    from agents.simple_agent import SimpleAgent
    from utils.config import Config
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class AgentHealthTester:
    """Test suite for individual agent health"""
    
    def __init__(self):
        self.results = {}
        
    def test_simple_agent(self):
        """Test simple agent functionality"""
        print("\n🤖 Testing Simple Agent...")
        
        try:
            # Initialize simple agent
            simple_agent = SimpleAgent()
            
            test_cases = [
                ("hello", "greeting"),
                ("2 + 2", "math"),
                ("thanks", "gratitude"),
                ("What's the weather?", "simple question")
            ]
            
            successes = 0
            for query, test_type in test_cases:
                try:
                    result = simple_agent.run(query)
                    response = result.get("response", "") if isinstance(result, dict) else str(result)
                    
                    if response and len(response) > 5:
                        successes += 1
                        print(f"   ✅ {test_type}: '{response[:50]}...'")
                    else:
                        print(f"   ❌ {test_type}: Empty or short response")
                        
                except Exception as e:
                    print(f"   ❌ {test_type}: Error - {e}")
            
            success_rate = successes / len(test_cases)
            status = "✅ PASS" if success_rate >= 0.75 else "❌ FAIL"
            
            print(f"   📊 Simple Agent: {successes}/{len(test_cases)} tests passed ({success_rate:.1%})")
            self.results["simple_agent"] = {
                "status": status,
                "success_rate": success_rate,
                "tests_passed": successes,
                "total_tests": len(test_cases)
            }
            
        except Exception as e:
            print(f"   ❌ Simple Agent failed to initialize: {e}")
            self.results["simple_agent"] = {"status": "❌ FAIL", "error": str(e)}
    
    def test_rag_agent(self):
        """Test RAG agent functionality"""
        print("\n📚 Testing RAG Agent...")
        
        try:
            # Initialize RAG agent
            rag_agent = RAGAgent()
            
            test_cases = [
                ("Python programming", "technical query"),
                ("machine learning", "ML query"),
                ("data structures", "CS fundamentals"),
                ("what is recursion", "basic concept")
            ]
            
            successes = 0
            for query, test_type in test_cases:
                try:
                    start_time = time.time()
                    result = rag_agent.run(query)
                    search_time = time.time() - start_time
                    
                    response = result.get("response", "") if isinstance(result, dict) else str(result)
                    
                    # Check response quality
                    if response and len(response) > 20 and search_time < 5.0:
                        successes += 1
                        print(f"   ✅ {test_type}: Response length {len(response)} chars ({search_time:.2f}s)")
                    else:
                        print(f"   ❌ {test_type}: Poor response (len: {len(response)}, time: {search_time:.2f}s)")
                        
                except Exception as e:
                    print(f"   ❌ {test_type}: Error - {e}")
            
            success_rate = successes / len(test_cases)
            status = "✅ PASS" if success_rate >= 0.75 else "❌ FAIL"
            
            print(f"   📊 RAG Agent: {successes}/{len(test_cases)} tests passed ({success_rate:.1%})")
            self.results["rag_agent"] = {
                "status": status,
                "success_rate": success_rate,
                "tests_passed": successes,
                "total_tests": len(test_cases)
            }
            
        except Exception as e:
            print(f"   ❌ RAG Agent failed to initialize: {e}")
            traceback.print_exc()
            self.results["rag_agent"] = {"status": "❌ FAIL", "error": str(e)}
    
    def test_research_agent(self):
        """Test research agent functionality"""
        print("\n🔍 Testing Research Agent...")
        
        try:
            # Initialize research agent
            research_agent = ResearchAgent()
            
            test_cases = [
                ("latest AI news", "current events"),
                ("Python 3.12 features", "technology updates"),
                ("climate change 2024", "recent developments")
            ]
            
            successes = 0
            for query, test_type in test_cases:
                try:
                    start_time = time.time()
                    result = research_agent.run(query)
                    search_time = time.time() - start_time
                    
                    response = result.get("response", "") if isinstance(result, dict) else str(result)
                    
                    # Check response quality
                    if response and len(response) > 50 and search_time < 30.0:
                        successes += 1
                        print(f"   ✅ {test_type}: Response length {len(response)} chars ({search_time:.2f}s)")
                    else:
                        print(f"   ❌ {test_type}: Poor response (len: {len(response)}, time: {search_time:.2f}s)")
                        
                except Exception as e:
                    print(f"   ❌ {test_type}: Error - {e}")
            
            success_rate = successes / len(test_cases)
            status = "✅ PASS" if success_rate >= 0.66 else "❌ FAIL"  # Lower threshold for web searches
            
            print(f"   📊 Research Agent: {successes}/{len(test_cases)} tests passed ({success_rate:.1%})")
            self.results["research_agent"] = {
                "status": status,
                "success_rate": success_rate,
                "tests_passed": successes,
                "total_tests": len(test_cases)
            }
            
        except Exception as e:
            print(f"   ❌ Research Agent failed to initialize: {e}")
            traceback.print_exc()
            self.results["research_agent"] = {"status": "❌ FAIL", "error": str(e)}
    
    def test_manager_system(self):
        """Test manager agent system functionality"""
        print("\n🎯 Testing Manager Agent System...")
        
        try:
            # Initialize manager system
            manager_system = ManagerAgentSystem()
            
            test_cases = [
                ("hello", "simple greeting"),
                ("explain Python functions", "knowledge query"),
                ("latest tech news", "research query"),
                ("2 + 2", "math query")
            ]
            
            successes = 0
            for query, test_type in test_cases:
                try:
                    start_time = time.time()
                    response = manager_system.run_query(query)
                    query_time = time.time() - start_time
                    
                    # Check response quality
                    if response and len(response) > 10 and query_time < 60.0:
                        successes += 1
                        print(f"   ✅ {test_type}: Response length {len(response)} chars ({query_time:.2f}s)")
                    else:
                        print(f"   ❌ {test_type}: Poor response (len: {len(response)}, time: {query_time:.2f}s)")
                        
                except Exception as e:
                    print(f"   ❌ {test_type}: Error - {e}")
                    traceback.print_exc()
            
            success_rate = successes / len(test_cases)
            status = "✅ PASS" if success_rate >= 0.75 else "❌ FAIL"
            
            print(f"   📊 Manager System: {successes}/{len(test_cases)} tests passed ({success_rate:.1%})")
            self.results["manager_system"] = {
                "status": status,
                "success_rate": success_rate,
                "tests_passed": successes,
                "total_tests": len(test_cases)
            }
            
        except Exception as e:
            print(f"   ❌ Manager System failed to initialize: {e}")
            traceback.print_exc()
            self.results["manager_system"] = {"status": "❌ FAIL", "error": str(e)}
    
    def test_system_performance(self):
        """Test overall system performance metrics"""
        print("\n⚡ Testing System Performance...")
        
        try:
            manager_system = ManagerAgentSystem()
            
            # Performance test queries
            queries = [
                "hello",
                "what is Python",
                "latest AI developments",
                "2 + 2",
                "explain neural networks"
            ]
            
            times = []
            successes = 0
            
            for query in queries:
                try:
                    start = time.time()
                    response = manager_system.run_query(query)
                    query_time = time.time() - start
                    times.append(query_time)
                    
                    if response and len(response) > 5:
                        successes += 1
                        
                except Exception as e:
                    print(f"   ⚠️ Performance test failed for '{query}': {e}")
            
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                
                print(f"   📊 Performance Results:")
                print(f"       Average response time: {avg_time:.2f}s")
                print(f"       Min response time: {min_time:.2f}s")
                print(f"       Max response time: {max_time:.2f}s")
                print(f"       Success rate: {successes}/{len(queries)} ({successes/len(queries):.1%})")
                
                # Performance criteria
                perf_pass = avg_time < 10.0 and max_time < 30.0 and successes/len(queries) >= 0.8
                status = "✅ PASS" if perf_pass else "⚠️ MARGINAL"
                
                self.results["performance"] = {
                    "status": status,
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time,
                    "success_rate": successes/len(queries)
                }
            else:
                self.results["performance"] = {"status": "❌ FAIL", "error": "No successful queries"}
                
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
            self.results["performance"] = {"status": "❌ FAIL", "error": str(e)}
    
    def generate_health_report(self):
        """Generate comprehensive health report"""
        print("\n" + "="*60)
        print("🏥 AGENT HEALTH TEST REPORT")
        print("="*60)
        
        # Count results
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() 
                          if isinstance(result, dict) and "✅" in result.get("status", ""))
        
        print(f"\n🎯 OVERALL HEALTH:")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   System health: {passed_tests/total_tests:.1%}" if total_tests > 0 else "   System health: 0%")
        
        # Detailed results
        for test_name, result in self.results.items():
            print(f"\n🔍 {test_name.upper().replace('_', ' ')}:")
            
            if isinstance(result, dict):
                print(f"   Status: {result.get('status', 'Unknown')}")
                
                if "success_rate" in result:
                    print(f"   Success Rate: {result['success_rate']:.1%}")
                    print(f"   Tests: {result.get('tests_passed', 0)}/{result.get('total_tests', 0)}")
                
                if "avg_time" in result:
                    print(f"   Avg Response Time: {result['avg_time']:.2f}s")
                    print(f"   Max Response Time: {result['max_time']:.2f}s")
                
                if "error" in result:
                    print(f"   Error: {result['error']}")
        
        # Health verdict
        if passed_tests / total_tests >= 0.8:
            print(f"\n🎉 VERDICT: System is HEALTHY! 🎉")
            print(f"   All critical components are functioning properly.")
        elif passed_tests / total_tests >= 0.6:
            print(f"\n⚠️ VERDICT: System needs attention")
            print(f"   Some components may need optimization.")
        else:
            print(f"\n❌ VERDICT: System has issues")
            print(f"   Critical components are failing.")
        
        return self.results
    
    def run_all_tests(self):
        """Run all agent health tests"""
        print("🚀 STARTING AGENT HEALTH TEST SUITE")
        print("="*50)
        
        # Run all tests
        test_methods = [
            self.test_simple_agent,
            self.test_rag_agent,
            self.test_research_agent,
            self.test_manager_system,
            self.test_system_performance
        ]
        
        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                print(f"❌ Test {test_method.__name__} crashed: {e}")
                traceback.print_exc()
        
        # Generate report
        return self.generate_health_report()


def main():
    """Run agent health tests"""
    tester = AgentHealthTester()
    return tester.run_all_tests()


if __name__ == "__main__":
    main()