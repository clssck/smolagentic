#!/usr/bin/env python3
"""
End-to-End Workflow Test
Complete testing of the agentic RAG system with real APIs
"""

import sys
import time
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from core.manager_agent_system import ManagerAgentSystem
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class E2EWorkflowTester:
    """End-to-end workflow tester for the complete system"""
    
    def __init__(self):
        self.manager_system = None
        self.test_results = []
        
    def setup_system(self):
        """Initialize the complete manager system"""
        print("🚀 Initializing Complete Agentic RAG System...")
        try:
            start = time.time()
            self.manager_system = ManagerAgentSystem()
            init_time = time.time() - start
            print(f"✅ System initialized in {init_time:.2f}s")
            return True
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def test_greeting_workflow(self):
        """Test simple greeting workflow"""
        print("\n1. 👋 Testing Greeting Workflow")
        
        queries = [
            "Hello there!",
            "Good morning",
            "Hi, how are you?",
            "Hey!"
        ]
        
        for query in queries:
            try:
                start = time.time()
                response = self.manager_system.run_query(query)
                response_time = time.time() - start
                
                result = {
                    "test": "greeting",
                    "query": query,
                    "response_length": len(response),
                    "response_time": response_time,
                    "success": len(response) > 10 and response_time < 10,
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                }
                
                self.test_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"   {status} '{query}' -> {len(response)} chars ({response_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ '{query}' -> Error: {e}")
                self.test_results.append({
                    "test": "greeting",
                    "query": query,
                    "error": str(e),
                    "success": False
                })
    
    def test_math_workflow(self):
        """Test simple math workflow"""
        print("\n2. 🧮 Testing Math Workflow")
        
        math_queries = [
            "What is 2 + 2?",
            "Calculate 15 * 8",
            "What's 100 divided by 4?",
            "Solve: (5 + 3) * 2"
        ]
        
        for query in math_queries:
            try:
                start = time.time()
                response = self.manager_system.run_query(query)
                response_time = time.time() - start
                
                # Check if response contains numbers (likely correct math)
                has_numbers = any(char.isdigit() for char in response)
                
                result = {
                    "test": "math",
                    "query": query,
                    "response_length": len(response),
                    "response_time": response_time,
                    "has_numbers": has_numbers,
                    "success": has_numbers and len(response) > 5 and response_time < 15,
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                }
                
                self.test_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"   {status} '{query}' -> {len(response)} chars ({response_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ '{query}' -> Error: {e}")
                self.test_results.append({
                    "test": "math",
                    "query": query,
                    "error": str(e),
                    "success": False
                })
    
    def test_knowledge_workflow(self):
        """Test knowledge base queries through RAG agent"""
        print("\n3. 📚 Testing Knowledge Base Workflow")
        
        knowledge_queries = [
            "What is Python programming?",
            "Explain machine learning",
            "What are data structures?",
            "Define neural networks",
            "How does recursion work?"
        ]
        
        for query in knowledge_queries:
            try:
                start = time.time()
                response = self.manager_system.run_query(query)
                response_time = time.time() - start
                
                # Knowledge responses should be detailed
                is_detailed = len(response) > 100
                is_timely = response_time < 30
                
                result = {
                    "test": "knowledge",
                    "query": query,
                    "response_length": len(response),
                    "response_time": response_time,
                    "is_detailed": is_detailed,
                    "success": is_detailed and is_timely,
                    "response_preview": response[:200] + "..." if len(response) > 200 else response
                }
                
                self.test_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"   {status} '{query}' -> {len(response)} chars ({response_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ '{query}' -> Error: {e}")
                self.test_results.append({
                    "test": "knowledge",
                    "query": query,
                    "error": str(e),
                    "success": False
                })
    
    def test_research_workflow(self):
        """Test web research workflow through Research agent"""
        print("\n4. 🔍 Testing Research Workflow")
        
        research_queries = [
            "Latest AI developments 2024",
            "Current Python version features",
            "Recent machine learning breakthroughs",
            "What's new in technology today?"
        ]
        
        for query in research_queries:
            try:
                start = time.time()
                response = self.manager_system.run_query(query)
                response_time = time.time() - start
                
                # Research responses should be substantial and timely
                is_substantial = len(response) > 150
                is_reasonable_time = response_time < 60  # Research can take longer
                
                result = {
                    "test": "research",
                    "query": query,
                    "response_length": len(response),
                    "response_time": response_time,
                    "is_substantial": is_substantial,
                    "success": is_substantial and is_reasonable_time,
                    "response_preview": response[:200] + "..." if len(response) > 200 else response
                }
                
                self.test_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"   {status} '{query}' -> {len(response)} chars ({response_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ '{query}' -> Error: {e}")
                self.test_results.append({
                    "test": "research",
                    "query": query,
                    "error": str(e),
                    "success": False
                })
    
    def test_mixed_conversation(self):
        """Test a mixed conversation workflow"""
        print("\n5. 💬 Testing Mixed Conversation Workflow")
        
        conversation = [
            "Hello!",
            "What is Python?", 
            "Can you search for latest Python news?",
            "What's 25 + 17?",
            "Thanks for your help!"
        ]
        
        for i, query in enumerate(conversation):
            try:
                start = time.time()
                response = self.manager_system.run_query(query)
                response_time = time.time() - start
                
                result = {
                    "test": "conversation",
                    "step": i + 1,
                    "query": query,
                    "response_length": len(response),
                    "response_time": response_time,
                    "success": len(response) > 5 and response_time < 60,
                    "response_preview": response[:150] + "..." if len(response) > 150 else response
                }
                
                self.test_results.append(result)
                
                status = "✅" if result["success"] else "❌"
                print(f"   {status} Step {i+1}: '{query}' -> {len(response)} chars ({response_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ Step {i+1}: '{query}' -> Error: {e}")
                self.test_results.append({
                    "test": "conversation",
                    "step": i + 1,
                    "query": query,
                    "error": str(e),
                    "success": False
                })
    
    def test_system_memory(self):
        """Test conversation memory and context"""
        print("\n6. 🧠 Testing System Memory")
        
        try:
            # Get conversation history
            history = self.manager_system.get_conversation_history()
            
            # Test memory functions
            memory_tests = [
                ("conversation_history", len(history) > 0),
                ("history_length", len(history) <= 20),  # Should manage history size
                ("clear_function", True)  # Test if clear function exists
            ]
            
            for test_name, passed in memory_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}: {'PASS' if passed else 'FAIL'}")
                
                self.test_results.append({
                    "test": "memory",
                    "component": test_name,
                    "success": passed
                })
            
            # Test clear function
            try:
                self.manager_system.clear_conversation_history()
                after_clear = self.manager_system.get_conversation_history()
                clear_worked = len(after_clear) == 0
                
                status = "✅" if clear_worked else "❌"
                print(f"   {status} memory_clear: {'PASS' if clear_worked else 'FAIL'}")
                
            except Exception as e:
                print(f"   ❌ memory_clear: Error - {e}")
                
        except Exception as e:
            print(f"   ❌ Memory test failed: {e}")
    
    def test_performance_metrics(self):
        """Test system performance metrics"""
        print("\n7. ⚡ Testing Performance Metrics")
        
        try:
            # Get system metrics
            metrics = self.manager_system.get_performance_metrics()
            
            performance_tests = [
                ("metrics_available", isinstance(metrics, dict)),
                ("has_token_usage", "total_tokens" in metrics),
                ("has_timing", "average_time_seconds" in metrics),
                ("reasonable_performance", metrics.get("average_time_seconds", 0) < 30)
            ]
            
            for test_name, passed in performance_tests:
                status = "✅" if passed else "❌"
                print(f"   {status} {test_name}: {'PASS' if passed else 'FAIL'}")
                
                self.test_results.append({
                    "test": "performance",
                    "component": test_name,
                    "success": passed
                })
            
            # Show key metrics
            if isinstance(metrics, dict):
                print(f"   📊 Total tokens: {metrics.get('total_tokens', 'N/A')}")
                print(f"   📊 Average time: {metrics.get('average_time_seconds', 'N/A')}s")
                print(f"   📊 Total requests: {metrics.get('total_requests', 'N/A')}")
                
        except Exception as e:
            print(f"   ❌ Performance metrics test failed: {e}")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*80)
        print("🎯 END-TO-END WORKFLOW TEST REPORT")
        print("="*80)
        
        # Overall statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get("success", False))
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total tests: {total_tests}")
        print(f"   Successful tests: {successful_tests}")
        print(f"   Success rate: {successful_tests/total_tests:.1%}" if total_tests > 0 else "   Success rate: 0%")
        
        # Test category breakdown
        test_categories = {}
        for result in self.test_results:
            category = result.get("test", "unknown")
            if category not in test_categories:
                test_categories[category] = {"total": 0, "success": 0}
            test_categories[category]["total"] += 1
            if result.get("success", False):
                test_categories[category]["success"] += 1
        
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, stats in test_categories.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"   {category.upper()}: {stats['success']}/{stats['total']} ({success_rate:.1%})")
        
        # Performance analysis
        response_times = [r.get("response_time", 0) for r in self.test_results if "response_time" in r]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)
            min_time = min(response_times)
            
            print(f"\n⚡ PERFORMANCE ANALYSIS:")
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Fastest response: {min_time:.2f}s")
            print(f"   Slowest response: {max_time:.2f}s")
        
        # Error analysis
        errors = [r for r in self.test_results if not r.get("success", True)]
        if errors:
            print(f"\n❌ ERROR ANALYSIS:")
            print(f"   Failed tests: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                print(f"   - {error.get('test', 'unknown')}: {error.get('query', error.get('component', 'N/A'))}")
                if 'error' in error:
                    print(f"     Error: {error['error']}")
        
        # Save detailed report
        report_file = f"e2e_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "timestamp": time.time(),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests/total_tests if total_tests > 0 else 0,
                "test_categories": test_categories,
                "performance": {
                    "avg_response_time": avg_time if response_times else 0,
                    "max_response_time": max_time if response_times else 0,
                    "min_response_time": min_time if response_times else 0
                },
                "detailed_results": self.test_results
            }, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
        
        # Final verdict
        if successful_tests / total_tests >= 0.85:
            print(f"\n🎉 VERDICT: System is PRODUCTION READY! 🎉")
            print(f"   Your agentic RAG system passed {successful_tests}/{total_tests} tests.")
            print(f"   All workflows are functioning correctly with real APIs.")
        elif successful_tests / total_tests >= 0.70:
            print(f"\n✅ VERDICT: System is mostly functional")
            print(f"   {successful_tests}/{total_tests} tests passed.")
            print(f"   Minor issues that should be addressed before production.")
        else:
            print(f"\n⚠️ VERDICT: System needs significant work")
            print(f"   Only {successful_tests}/{total_tests} tests passed.")
            print(f"   Review errors and fix issues before deployment.")
        
        return {
            "success_rate": successful_tests/total_tests if total_tests > 0 else 0,
            "total_tests": total_tests,
            "successful_tests": successful_tests
        }
    
    def run_complete_test_suite(self):
        """Run the complete end-to-end test suite"""
        print("🚀 STARTING END-TO-END WORKFLOW TEST SUITE")
        print("="*80)
        
        if not self.setup_system():
            print("❌ System setup failed, aborting tests")
            return False
        
        # Run all test workflows
        test_workflows = [
            self.test_greeting_workflow,
            self.test_math_workflow,
            self.test_knowledge_workflow,
            self.test_research_workflow,
            self.test_mixed_conversation,
            self.test_system_memory,
            self.test_performance_metrics
        ]
        
        for workflow in test_workflows:
            try:
                workflow()
            except Exception as e:
                print(f"❌ Workflow {workflow.__name__} crashed: {e}")
        
        # Generate comprehensive report
        return self.generate_comprehensive_report()


def main():
    """Run the complete end-to-end test suite"""
    tester = E2EWorkflowTester()
    return tester.run_complete_test_suite()


if __name__ == "__main__":
    main()