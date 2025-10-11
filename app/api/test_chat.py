"""
Test script for the chat API functionality with MongoDB.
"""

import asyncio
import httpx
import json
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8000"


async def test_chat_functionality():
    """Test the complete chat workflow."""
    
    async with httpx.AsyncClient() as client:
        print("🚀 Testing Chat API Functionality with MongoDB\n")
        
        # Check MongoDB connection first
        print("0. Checking MongoDB connection...")
        mongodb_status = os.getenv("MONGODB_CONNECTION_STRING", "Not configured")
        print(f"   MongoDB URI: {mongodb_status}")
        print(f"   Database: {os.getenv('MONGODB_DATABASE', 'emotion_analysis_db')}\n")
        
        # Test 1: Health check
        print("1. Testing chat health check...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/chat/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}\n")
        except Exception as e:
            print(f"   Error: {e}\n")
        
        # Test 2: Start a new chat session
        print("2. Starting new chat session...")
        context = """
        Meeting Analysis Results:
        - Video Duration: 120 seconds
        - Dominant Emotion: Happy (87% confidence)
        - Mental Health Score: 82/100
        - Participants showed positive engagement throughout
        - Key moments: High energy at 30s, collaborative discussion at 90s
        """
        
        try:
            start_response = await client.post(
                f"{BASE_URL}/api/v1/chat/start",
                json={"context": context}
            )
            print(f"   Status: {start_response.status_code}")
            start_data = start_response.json()
            print(f"   Response: {start_data}\n")
            
            if start_response.status_code == 200:
                chat_id = start_data["chat_id"]
                print(f"   ✅ Chat ID: {chat_id}\n")
            else:
                print("   ❌ Failed to create chat session")
                return
                
        except Exception as e:
            print(f"   Error: {e}\n")
            return
        
        # Test 3: Send messages
        test_messages = [
            "Hello! Can you summarize the meeting analysis for me?",
            "What does the mental health score of 82/100 indicate?",
            "Can you provide recommendations based on these results?",
            "What were the key emotional patterns during the meeting?"
        ]
        
        for i, message in enumerate(test_messages, 3):
            print(f"{i}. Sending message: '{message[:50]}...'")
            try:
                msg_response = await client.post(
                    f"{BASE_URL}/api/v1/chat/message",
                    json={
                        "chat_id": chat_id,
                        "message": message
                    }
                )
                print(f"   Status: {msg_response.status_code}")
                
                if msg_response.status_code == 200:
                    msg_data = msg_response.json()
                    assistant_response = msg_data["assistant_message"]["content"]
                    print(f"   AI Response: {assistant_response[:100]}...")
                    print("   ✅ Message processed successfully\n")
                else:
                    print(f"   ❌ Error: {msg_response.json()}\n")
                    
            except Exception as e:
                print(f"   Error: {e}\n")
        
        # Test 4: Get chat session details
        print(f"{len(test_messages) + 3}. Getting chat session details...")
        try:
            session_response = await client.get(f"{BASE_URL}/api/v1/chat/session/{chat_id}")
            print(f"   Status: {session_response.status_code}")
            
            if session_response.status_code == 200:
                session_data = session_response.json()
                message_count = len(session_data["session"]["messages"])
                print(f"   ✅ Found {message_count} messages in session\n")
            else:
                print(f"   ❌ Error: {session_response.json()}\n")
                
        except Exception as e:
            print(f"   Error: {e}\n")
        
        # Test 5: List all sessions
        print(f"{len(test_messages) + 4}. Listing all chat sessions...")
        try:
            sessions_response = await client.get(f"{BASE_URL}/api/v1/chat/sessions")
            print(f"   Status: {sessions_response.status_code}")
            
            if sessions_response.status_code == 200:
                sessions_data = sessions_response.json()
                total_sessions = sessions_data["total_sessions"]
                print(f"   ✅ Found {total_sessions} active sessions\n")
            else:
                print(f"   ❌ Error: {sessions_response.json()}\n")
                
        except Exception as e:
            print(f"   Error: {e}\n")
        
        print("🎉 Chat API testing with MongoDB completed!")


async def test_mongodb_connection():
    """Test MongoDB connection directly."""
    print("🔌 Testing Direct MongoDB Connection\n")
    
    try:
        # Import MongoDB modules
        from api.data.mongodb_config import mongo_config
        
        print("1. Connecting to MongoDB...")
        success = await mongo_config.connect()
        
        if success:
            print("   ✅ Connected successfully")
            
            # Test health check
            print("2. Testing MongoDB health...")
            health = await mongo_config.health_check()
            print(f"   Status: {health.get('status', 'unknown')}")
            print(f"   Database: {health.get('database', 'unknown')}")
            print(f"   Chat sessions: {health.get('chat_sessions_count', 0)}")
            
            # Disconnect
            await mongo_config.disconnect()
            print("   ✅ Disconnected successfully")
        else:
            print("   ❌ Failed to connect")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n🎉 MongoDB connection test completed!")


if __name__ == "__main__":
    print("Choose a test to run:")
    print("1. Test MongoDB connection directly")
    print("2. Test Chat API (requires server running)")
    print("3. Both tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    try:
        if choice == "1":
            asyncio.run(test_mongodb_connection())
        elif choice == "2":
            print("\nMake sure the API server is running on http://localhost:8000")
            print("Run: python app/api/main.py\n")
            asyncio.run(test_chat_functionality())
        elif choice == "3":
            asyncio.run(test_mongodb_connection())
            print("\n" + "="*50 + "\n")
            print("Make sure the API server is running on http://localhost:8000")
            print("Run: python app/api/main.py\n")
            asyncio.run(test_chat_functionality())
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")