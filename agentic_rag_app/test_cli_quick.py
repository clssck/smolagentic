#!/usr/bin/env python3
"""
Quick CLI Test - Key Complexity Levels
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.core.manager_agent_system import ManagerAgentSystem


def test_quick_cli():
    """Test key complexity levels quickly"""
    
    print("🚀 QUICK CLI TEST - KEY COMPLEXITY LEVELS")
    print("=" * 60)
    
    # Initialize system
    try:
        system = ManagerAgentSystem("optimal_models_config.json")
        print("✅ System ready")
        print()
    except Exception as e:
        print(f"❌ System failed: {e}")
        return
    
    # Key test cases
    test_cases = [
        # TRIVIAL
        {
            "level": "TRIVIAL",
            "question": "Hello!",
            "timeout": 10
        },
        # SIMPLE
        {
            "level": "SIMPLE", 
            "question": "What is 25 * 16?",
            "timeout": 15
        },
        # MODERATE
        {
            "level": "MODERATE",
            "question": "Search for latest AI news and give me 3 key points",
            "timeout": 45
        },
        # COMPLEX
        {
            "level": "COMPLEX",
            "question": "Analyze the pros and cons of transformer vs RNN architectures for NLP tasks",
            "timeout": 60
        },
        # VERY COMPLEX  
        {
            "level": "EXTREME",
            "question": "Research quantum computing progress in 2024, analyze commercial viability, and predict next breakthrough areas",
            "timeout": 90
        }
    ]
    
    results = []
    start_time = time.time()
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}/5 - {test['level']}")
        print(f"{'='*60}")
        print(f"Q: {test['question']}")
        print("-" * 60)
        
        test_start = time.time()
        try:
            print("🤖 Processing...")
            
            # Set a reasonable timeout for each test
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Query timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(test['timeout'])
            
            try:
                response = system.run_query(test["question"])
                signal.alarm(0)  # Cancel timeout
                
                elapsed = time.time() - test_start
                response_str = str(response)
                
                # Quick quality check
                success = len(response_str) > 20 and "error" not in response_str.lower()
                
                results.append({
                    "level": test['level'],
                    "success": success,
                    "time": elapsed,
                    "length": len(response_str)
                })
                
                status = "✅" if success else "⚠️"
                print(f"\n{status} {elapsed:.1f}s | {len(response_str)} chars")
                
                # Show preview
                preview = response_str[:150] + "..." if len(response_str) > 150 else response_str
                print(f"PREVIEW: {preview}")
                
            except TimeoutError:
                signal.alarm(0)
                elapsed = time.time() - test_start
                results.append({
                    "level": test['level'],
                    "success": False,
                    "time": elapsed,
                    "length": 0,
                    "error": "timeout"
                })
                print(f"\n❌ TIMEOUT after {elapsed:.1f}s")
                
        except Exception as e:
            elapsed = time.time() - test_start
            results.append({
                "level": test['level'],
                "success": False,
                "time": elapsed,
                "length": 0,
                "error": str(e)[:50]
            })
            print(f"\n❌ ERROR {elapsed:.1f}s: {str(e)[:50]}")
    
    # Results summary
    total_time = time.time() - start_time
    successes = sum(1 for r in results if r['success'])
    success_rate = (successes / len(results)) * 100
    avg_time = sum(r['time'] for r in results) / len(results)
    
    print(f"\n{'='*60}")
    print("🏆 QUICK TEST RESULTS")
    print(f"{'='*60}")
    print(f"Success Rate: {success_rate:.0f}% ({successes}/{len(results)})")
    print(f"Average Time: {avg_time:.1f}s")
    print(f"Total Time: {total_time:.1f}s")
    
    print(f"\nBy Level:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {result['level']:<12} {status} {result['time']:<6.1f}s")
    
    # Assessment
    if success_rate >= 80:
        print(f"\n🏆 EXCELLENT: System handles complexity levels very well!")
    elif success_rate >= 60:
        print(f"\n✅ GOOD: System performs well across most levels!")
    else:
        print(f"\n⚠️ NEEDS WORK: System struggling with complexity!")
    
    if avg_time < 15:
        print(f"⚡ FAST: Excellent response times!")
    elif avg_time < 30:
        print(f"🚀 GOOD: Reasonable response times!")
    else:
        print(f"⚠️ SLOW: Response times need optimization!")
    
    return results


if __name__ == "__main__":
    test_quick_cli()