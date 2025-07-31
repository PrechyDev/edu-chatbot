#!/usr/bin/env python3
"""
Test script for FastAPI EduBot server.
Run this after installing FastAPI dependencies to verify everything works.
"""

import requests
import json
import time

def test_fastapi_server():
    """Test the FastAPI server endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing FastAPI EduBot Server")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("   ✅ Health check passed")
            health_data = response.json()
            print(f"   📊 Bot available: {health_data.get('bot_available', False)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Server not running. Start with: python api.py")
        return
    
    # Test chat endpoint
    print("\n2. Testing conversation memory...")
    session_id = "test_session_api"
    
    messages = [
        "Hello!",
        "What are UTME requirements?",
        "What about Computer Science specifically?",
        "Thank you!"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n   👤 Message {i}: {message}")
        
        chat_data = {
            "message": message,
            "session_id": session_id
        }
        
        try:
            response = requests.post(
                f"{base_url}/chat",
                json=chat_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   🤖 Response: {result['response'][:100]}...")
                print(f"   📊 Messages in session: {result.get('message_count', 0)}")
                print(f"   🧠 Query type: {result.get('query_type', 'unknown')}")
                print(f"   📚 Used KB: {result.get('knowledge_base_used', False)}")
            else:
                print(f"   ❌ Chat failed: {response.status_code}")
                print(f"   📝 Error: {response.text}")
        
        except Exception as e:
            print(f"   ❌ Request error: {e}")
        
        time.sleep(0.5)  # Small delay between requests
    
    # Test session management
    print("\n3. Testing session management...")
    try:
        # List sessions
        response = requests.get(f"{base_url}/sessions")
        if response.status_code == 200:
            sessions_data = response.json()
            print(f"   ✅ Available sessions: {sessions_data.get('count', 0)}")
            print(f"   📝 Sessions: {sessions_data.get('sessions', [])}")
        
        # Get specific session info
        response = requests.get(f"{base_url}/sessions/{session_id}")
        if response.status_code == 200:
            session_info = response.json()
            print(f"   ✅ Session info retrieved")
            print(f"   📊 Message count: {session_info.get('message_count', 0)}")
        
    except Exception as e:
        print(f"   ❌ Session management error: {e}")
    
    # Test API documentation
    print("\n4. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print("   ✅ Swagger UI docs available at /docs")
        
        response = requests.get(f"{base_url}/redoc")
        if response.status_code == 200:
            print("   ✅ ReDoc documentation available at /redoc")
        
    except Exception as e:
        print(f"   ⚠️ Documentation endpoints: {e}")
    
    print("\n🎉 FastAPI Server Test Complete!")
    print("\nNext steps:")
    print("1. Visit http://localhost:8000/docs for interactive API testing")
    print("2. Visit http://localhost:8000/redoc for API documentation")
    print("3. Use the /chat endpoint for frontend integration")

if __name__ == "__main__":
    test_fastapi_server()
