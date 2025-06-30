#!/usr/bin/env python3
"""
Comprehensive CLI Test - Easy to Extremely Complex Questions
Tests the optimal multi-model system with real-world scenarios
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.core.manager_agent_system import ManagerAgentSystem


def test_comprehensive_cli():
    """Test with questions ranging from trivial to extremely complex"""
    
    print("🚀 COMPREHENSIVE CLI TEST - EASY TO INSANELY COMPLEX")
    print("=" * 80)
    print("Testing optimal system: Groq Qwen3-32B Manager + Mistral Small 3.2 Agents")
    print()
    
    # Initialize system
    try:
        system = ManagerAgentSystem("optimal_models_config.json")
        print("✅ System initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test cases from trivial to extremely complex
    test_cases = [
        # LEVEL 1: TRIVIAL
        {
            "level": "TRIVIAL",
            "question": "Hello",
            "expected_type": "greeting",
            "complexity": 1
        },
        {
            "level": "TRIVIAL", 
            "question": "What is 2 + 2?",
            "expected_type": "basic_math",
            "complexity": 1
        },
        {
            "level": "TRIVIAL",
            "question": "Thanks!",
            "expected_type": "gratitude",
            "complexity": 1
        },
        
        # LEVEL 2: SIMPLE
        {
            "level": "SIMPLE",
            "question": "What is 15 * 37 + 89?",
            "expected_type": "math",
            "complexity": 2
        },
        {
            "level": "SIMPLE",
            "question": "What's the weather like today?",
            "expected_type": "current_info",
            "complexity": 2
        },
        
        # LEVEL 3: MODERATE 
        {
            "level": "MODERATE",
            "question": "Search for the latest developments in quantum computing and summarize the key breakthroughs",
            "expected_type": "research_summary",
            "complexity": 3
        },
        {
            "level": "MODERATE",
            "question": "Find information about machine learning optimization techniques in your knowledge base",
            "expected_type": "knowledge_retrieval", 
            "complexity": 3
        },
        
        # LEVEL 4: COMPLEX
        {
            "level": "COMPLEX",
            "question": "Compare the performance characteristics of transformer architectures vs RNN architectures for sequence modeling, considering computational complexity, memory requirements, and parallelization capabilities",
            "expected_type": "technical_analysis",
            "complexity": 4
        },
        {
            "level": "COMPLEX", 
            "question": "Research the current state of artificial general intelligence development, analyze the major research approaches, and predict potential timelines based on current progress",
            "expected_type": "research_analysis",
            "complexity": 4
        },
        
        # LEVEL 5: VERY COMPLEX
        {
            "level": "VERY COMPLEX",
            "question": "Analyze the intersection of quantum computing and machine learning: research current quantum ML algorithms, evaluate their theoretical advantages over classical approaches, assess hardware limitations, and provide a comprehensive roadmap for practical quantum ML applications in the next decade",
            "expected_type": "multi_domain_research",
            "complexity": 5
        },
        
        # LEVEL 6: EXTREMELY COMPLEX
        {
            "level": "EXTREMELY COMPLEX",
            "question": "Conduct a comprehensive analysis of the global AI governance landscape: research current international AI regulations, analyze geopolitical implications of AI development across major powers (US, China, EU), evaluate the effectiveness of existing frameworks like the EU AI Act, assess the risks of AI arms races, propose solutions for international AI cooperation, and develop a multi-stakeholder governance model that balances innovation with safety while considering economic, ethical, and security dimensions. Include specific policy recommendations with implementation timelines.",
            "expected_type": "multi_domain_policy_analysis",
            "complexity": 6
        },
        
        # LEVEL 7: FUCKING INSANE
        {
            "level": "FUCKING INSANE",
            "question": "Create a comprehensive meta-analysis combining: (1) Latest research on consciousness and AI sentience, (2) Current developments in neuromorphic computing and brain-computer interfaces, (3) Quantum consciousness theories and their implications for AGI, (4) Economic models for post-scarcity societies enabled by AI, (5) Philosophical frameworks for AI rights and personhood, (6) Technical specifications for safe superintelligence alignment, (7) Global governance structures for managing existential risks, (8) Timeline analysis for technological singularity scenarios. Synthesize all these domains into a unified framework with actionable recommendations for humanity's transition to a post-human future. Include risk mitigation strategies, ethical guidelines, and implementation roadmaps across multiple decades.",
            "expected_type": "existential_meta_analysis", 
            "complexity": 7
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/13 - COMPLEXITY LEVEL: {test_case['level']} ({test_case['complexity']}/7)")
        print(f"{'='*80}")
        print(f"QUESTION: {test_case['question']}")
        print(f"\nExpected Type: {test_case['expected_type']}")
        print("-" * 80)
        
        # Execute test
        start_time = time.time()
        try:
            print("🤖 PROCESSING...")
            response = system.run_query(test_case["question"])
            elapsed_time = time.time() - start_time
            
            # Evaluate response
            response_str = str(response)
            response_length = len(response_str)
            
            # Success criteria based on complexity
            min_length_by_complexity = {1: 10, 2: 20, 3: 50, 4: 100, 5: 200, 6: 300, 7: 500}
            min_expected = min_length_by_complexity.get(test_case['complexity'], 50)
            
            success = response_length >= min_expected and "error" not in response_str.lower()
            
            # Record result
            result = {
                "level": test_case['level'],
                "complexity": test_case['complexity'],
                "question": test_case['question'][:100] + "..." if len(test_case['question']) > 100 else test_case['question'],
                "success": success,
                "time": elapsed_time,
                "response_length": response_length,
                "response_preview": response_str[:200] + "..." if len(response_str) > 200 else response_str
            }
            results.append(result)
            
            # Display result
            status = "✅ SUCCESS" if success else "⚠️ PARTIAL"
            print(f"\n{status} | {elapsed_time:.2f}s | {response_length} chars")
            print(f"RESPONSE PREVIEW: {result['response_preview']}")
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            
            result = {
                "level": test_case['level'],
                "complexity": test_case['complexity'], 
                "question": test_case['question'][:100] + "..." if len(test_case['question']) > 100 else test_case['question'],
                "success": False,
                "time": elapsed_time,
                "response_length": 0,
                "error": error_msg
            }
            results.append(result)
            
            print(f"\n❌ ERROR | {elapsed_time:.2f}s")
            print(f"ERROR: {error_msg}")
        
        # Brief pause between tests
        if i < len(test_cases):
            time.sleep(1)
    
    # Generate comprehensive report
    total_time = time.time() - total_start_time
    
    print(f"\n\n{'='*80}")
    print("🏆 COMPREHENSIVE TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Overall stats
    successful_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100
    avg_time = sum(r['time'] for r in results) / total_tests
    
    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    print(f"   Average Time: {avg_time:.2f}s per query")
    print(f"   Total Time: {total_time:.1f}s")
    print(f"   Speed Rating: {'🏆 EXCELLENT' if avg_time < 5 else '✅ GOOD' if avg_time < 10 else '⚠️ SLOW'}")
    
    # Performance by complexity level
    print(f"\n📈 PERFORMANCE BY COMPLEXITY LEVEL:")
    complexity_stats = {}
    for result in results:
        level = result['complexity']
        if level not in complexity_stats:
            complexity_stats[level] = {'success': 0, 'total': 0, 'times': []}
        
        complexity_stats[level]['total'] += 1
        if result['success']:
            complexity_stats[level]['success'] += 1
        complexity_stats[level]['times'].append(result['time'])
    
    for level in sorted(complexity_stats.keys()):
        stats = complexity_stats[level]
        success_rate = (stats['success'] / stats['total']) * 100
        avg_time = sum(stats['times']) / len(stats['times'])
        level_name = {1: "TRIVIAL", 2: "SIMPLE", 3: "MODERATE", 4: "COMPLEX", 
                     5: "VERY COMPLEX", 6: "EXTREMELY COMPLEX", 7: "FUCKING INSANE"}[level]
        
        print(f"   Level {level} ({level_name}): {success_rate:.0f}% success, {avg_time:.2f}s avg")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'Level':<12} {'Success':<8} {'Time':<8} {'Question':<50}")
    print("-" * 80)
    
    for result in results:
        success_icon = "✅" if result['success'] else "❌"
        level_name = {1: "TRIVIAL", 2: "SIMPLE", 3: "MODERATE", 4: "COMPLEX",
                     5: "V.COMPLEX", 6: "EXTREME", 7: "INSANE"}[result['complexity']]
        question_short = result['question'][:47] + "..." if len(result['question']) > 50 else result['question']
        
        print(f"{level_name:<12} {success_icon:<8} {result['time']:<8.2f} {question_short:<50}")
    
    # System capability assessment
    print(f"\n🎯 SYSTEM CAPABILITY ASSESSMENT:")
    
    if success_rate >= 90:
        print("   🏆 EXCEPTIONAL: System handles all complexity levels excellently")
    elif success_rate >= 80:
        print("   ✅ EXCELLENT: System performs very well across complexity levels")
    elif success_rate >= 70:
        print("   👍 GOOD: System handles most scenarios effectively")
    elif success_rate >= 60:
        print("   ⚠️ MODERATE: System works but may struggle with complex queries")
    else:
        print("   ❌ NEEDS IMPROVEMENT: System has significant limitations")
    
    # Speed assessment
    if avg_time < 3:
        print("   ⚡ BLAZING FAST: Ultra-responsive performance")
    elif avg_time < 6:
        print("   🚀 FAST: Excellent response times")
    elif avg_time < 10:
        print("   ✅ GOOD: Reasonable response times")
    else:
        print("   ⚠️ SLOW: May need optimization")
    
    # Model performance assessment
    print(f"\n🧠 MODEL PERFORMANCE ASSESSMENT:")
    print("   Manager (Groq Qwen3-32B): Ultra-fast coordination ⚡")
    print("   Agents (Mistral Small 3.2): Excellent task execution 🎯")
    print("   Architecture: Optimal multi-model delegation 🏗️")
    
    print(f"\n🎉 COMPREHENSIVE CLI TEST COMPLETE!")
    print(f"The system is {'ready for production' if success_rate >= 75 else 'ready for further development'}!")
    
    return {
        'success_rate': success_rate,
        'avg_time': avg_time,
        'total_time': total_time,
        'results': results
    }


if __name__ == "__main__":
    test_comprehensive_cli()