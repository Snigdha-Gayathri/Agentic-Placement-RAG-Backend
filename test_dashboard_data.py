"""Test what the backend returns for dashboard data."""

import requests
import json
import time

def test_dashboard():
    # First, send an Amazon query
    print("=" * 80)
    print("SENDING QUERY: What behavioral questions does Amazon focus on?")
    print("=" * 80)
    
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={
            "query": "What behavioral questions does Amazon focus on?",
            "session_id": f"test_{int(time.time())}"
        },
        timeout=60
    )
    
    if response.status_code != 200:
        print(f"ERROR: Chat API returned {response.status_code}")
        print(response.text)
        return
    
    chat_data = response.json()
    request_id = chat_data.get("request_id")
    
    print(f"\nRequest ID: {request_id}")
    
    # Now fetch the dashboard data
    print("\nFETCHING DASHBOARD DATA...")
    time.sleep(1)  # Give it a moment
    
    dash_response = requests.get(f"http://localhost:8000/api/dashboard/{request_id}")
    
    if dash_response.status_code != 200:
        print(f"ERROR: Dashboard API returned {dash_response.status_code}")
        print(dash_response.text)
        return
    
    dashboard = dash_response.json()
    
    # Print the structure
    print("\nDASHBOARD DATA STRUCTURE:")
    print(json.dumps(dashboard, indent=2)[:3000])
    
    # Check query_info specifically
    print("\n" + "=" * 80)
    print("QUERY INFO SECTION:")
    print("=" * 80)
    
    if "query_info" in dashboard:
        query_info = dashboard["query_info"]
        print(f"Original Query: {query_info.get('original_query')}")
        print(f"Standalone Query: {query_info.get('standalone_query')}")
        print(f"Metadata Filters: {query_info.get('metadata_filters')}")
        print(f"Routing Decision: {query_info.get('routing_decision')}")
    else:
        print("ERROR: No 'query_info' field in dashboard data!")
        print(f"Available keys: {list(dashboard.keys())}")

if __name__ == "__main__":
    test_dashboard()
