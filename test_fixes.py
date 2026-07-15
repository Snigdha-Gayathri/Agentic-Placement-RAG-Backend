"""
Quick test script to verify the query routing and generation fixes.
Run this after restarting the backend server.
"""

import asyncio
import sys
sys.path.insert(0, 'backend')

from backend.core.retrieval.router import QueryRouter


def test_company_detection():
    """Test that all companies are now properly detected."""
    router = QueryRouter()
    
    test_cases = [
        ("What system design questions does Netflix ask?", "Netflix", "system_design"),
        ("What ML questions does Meta ask?", "Meta", "ml"),
        ("Google DSA interview questions", "Google", "dsa"),
        ("TCS coding round questions", "Tcs", "dsa"),
        ("Infosys SQL questions", "Infosys", "sql"),
        ("NVIDIA machine learning interviews", "Nvidia", "machine_learning"),
        ("Flipkart system design round", "Flipkart", "system_design"),
    ]
    
    print("=" * 80)
    print("COMPANY DETECTION TESTS")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for query, expected_company, expected_topic in test_cases:
        result = router.route(query)
        detected_company = result.metadata_filters.get("company", "NOT DETECTED")
        detected_topic = result.metadata_filters.get("topic", "NOT DETECTED")
        
        company_match = detected_company.lower() == expected_company.lower()
        topic_match = detected_topic.lower().replace("_", " ") == expected_topic.lower().replace("_", " ")
        
        if company_match:
            passed += 1
            status = "✅ PASS"
        else:
            failed += 1
            status = "❌ FAIL"
        
        print(f"\n{status}")
        print(f"Query: {query}")
        print(f"Expected: company={expected_company}, topic={expected_topic}")
        print(f"Detected: company={detected_company}, topic={detected_topic}")
        print(f"Reasoning: {result.reasoning}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


def test_no_google_default():
    """Verify that Google is NOT the default when company isn't mentioned."""
    router = QueryRouter()
    
    print("\n" + "=" * 80)
    print("NO DEFAULT COMPANY TEST")
    print("=" * 80)
    
    # Queries that don't mention any company
    test_cases = [
        "What are common system design questions?",
        "Best DSA practice problems",
        "How to prepare for ML interviews?",
    ]
    
    for query in test_cases:
        result = router.route(query)
        detected_company = result.metadata_filters.get("company", "NONE")
        
        if detected_company == "NONE":
            print(f"\n✅ PASS: '{query}'")
            print(f"   No company detected (correct behavior)")
        else:
            print(f"\n❌ FAIL: '{query}'")
            print(f"   Incorrectly detected: {detected_company}")
            return False
    
    print("\n" + "=" * 80)
    print("All queries correctly returned NO default company")
    print("=" * 80)
    return True


def test_expanded_topics():
    """Test that new topics are detected."""
    router = QueryRouter()
    
    print("\n" + "=" * 80)
    print("EXPANDED TOPICS TEST")
    print("=" * 80)
    
    test_cases = [
        ("array sorting algorithms", "arrays"),
        ("string manipulation problems", "strings"),
        ("linked list questions", "linked_list"),
        ("machine learning system design", "machine_learning"),
        ("cloud architecture questions", "cloud"),
        ("distributed systems concepts", "distributed_systems"),
    ]
    
    passed = 0
    for query, expected_topic in test_cases:
        result = router.route(query)
        detected_topic = result.metadata_filters.get("topic", "NOT DETECTED")
        
        if detected_topic.lower().replace("_", " ") == expected_topic.lower().replace("_", " "):
            print(f"✅ '{query}' → {detected_topic}")
            passed += 1
        else:
            print(f"❌ '{query}' → Expected '{expected_topic}', got '{detected_topic}'")
    
    print(f"\n{passed}/{len(test_cases)} topics correctly detected")
    return passed == len(test_cases)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AGENTIC PLACEMENT RAG - FIX VERIFICATION TESTS")
    print("=" * 80)
    print("\nTesting fixes for:")
    print("1. Query Routing Bug (Company Detection)")
    print("2. Expanded Topic Detection")
    print("3. No Default Company Fallback")
    print()
    
    results = []
    
    # Test 1: Company Detection
    results.append(("Company Detection", test_company_detection()))
    
    # Test 2: No Google Default
    results.append(("No Default Company", test_no_google_default()))
    
    # Test 3: Expanded Topics
    results.append(("Expanded Topics", test_expanded_topics()))
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Query routing bug is FIXED.")
        print("=" * 80)
        print("\nNext: Restart backend server to test generation improvements:")
        print("  cd backend")
        print("  uvicorn app.main:app --reload --port 8000")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED. Review the output above.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
