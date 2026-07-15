"""Test Amazon query to see what the backend is actually returning."""

import requests
import json

def test_amazon_query():
    query = "What behavioral questions does Amazon focus on?"
    
    print("=" * 80)
    print(f"Testing Query: {query}")
    print("=" * 80)
    
    try:
        response = requests.post(
            "http://localhost:8000/api/chat",
            json={"query": query, "session_id": "test_amazon_session"},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Check pipeline_data
            pipeline_data = data.get("pipeline_data", {})
            query_info = pipeline_data.get("query_info", {})
            
            print("\n[QUERY ANALYSIS]")
            print(f"Original Query: {query_info.get('original_query', 'N/A')}")
            print(f"Standalone Query: {query_info.get('standalone_query', 'N/A')}")
            print(f"Routing Decision: {query_info.get('routing_decision', 'N/A')}")
            print(f"Metadata Filters: {query_info.get('metadata_filters', {})}")
            
            # Check retrieval info
            retrieval_info = pipeline_data.get("retrieval_info", {})
            print(f"\n[RETRIEVAL INFO]")
            print(f"Retriever Used: {retrieval_info.get('retriever_used', 'N/A')}")
            print(f"Retrieved Chunks: {retrieval_info.get('final_count', 0)}")
            
            chunks = retrieval_info.get("retrieved_chunks", [])
            if chunks:
                print(f"\n[TOP 3 RETRIEVED DOCUMENTS]")
                for i, chunk in enumerate(chunks[:3], 1):
                    source = chunk.get('source', 'Unknown')
                    text_preview = chunk.get('text', '')[:100]
                    print(f"{i}. Source: {source}")
                    print(f"   Preview: {text_preview}...")
            
            # Check request_id
            request_id = data.get("request_id")
            print(f"\n[REQUEST ID]: {request_id}")
            
            # Try to fetch dashboard data
            if request_id:
                print(f"\n[FETCHING DASHBOARD DATA]")
                dash_response = requests.get(f"http://localhost:8000/api/dashboard/{request_id}")
                if dash_response.status_code == 200:
                    dash_data = dash_response.json()
                    dash_query = dash_data.get("query_info", {})
                    print(f"Dashboard Original Query: {dash_query.get('original_query', 'N/A')}")
                    print(f"Dashboard Metadata Filters: {dash_query.get('metadata_filters', {})}")
                else:
                    print(f"Dashboard fetch failed: HTTP {dash_response.status_code}")
            
        else:
            print(f"ERROR: HTTP {response.status_code}")
            print(response.text[:500])
    
    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_amazon_query()
