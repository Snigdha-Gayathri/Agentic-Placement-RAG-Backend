"""Quick API test to verify routing fixes are working."""

import requests
import json

def test_routing():
    """Test that company routing works correctly."""
    
    test_cases = [
        ("What system design questions does Netflix ask?", "Netflix"),
        ("What ML questions does Meta ask?", "Meta"),
        ("Google DSA interview questions", "Google"),
    ]
    
    print("=" * 80)
    print("TESTING BACKEND API ROUTING")
    print("=" * 80)
    
    for query, expected_company in test_cases:
        print(f"\nQuery: {query}")
        print(f"Expected Company: {expected_company}")
        
        try:
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={"query": query, "session_id": "test_session"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract company from pipeline data
                query_info = data.get("pipeline_data", {}).get("query_info", {})
                metadata_filters = query_info.get("metadata_filters", {})
                detected_company = metadata_filters.get("company", "NOT DETECTED")
                
                if detected_company.lower() == expected_company.lower():
                    print(f"PASS - Detected: {detected_company}")
                else:
                    print(f"FAIL - Detected: {detected_company} (Expected: {expected_company})")
                
                print(f"Metadata Filters: {metadata_filters}")
            else:
                print(f"FAIL - HTTP {response.status_code}: {response.text[:200]}")
        
        except Exception as e:
            print(f"ERROR: {e}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_routing()
