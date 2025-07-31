#!/usr/bin/env python3
"""
EduBot Chat Interface

Interactive chat interface for querying the educational knowledge base.
Usage: python chat.py [question]
"""

import sys
import os
from pathlib import Path

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent / 'src'))

import argparse

sys.path.append('src')

from src.chat_system.bot import EduBot

def main():
    parser = argparse.ArgumentParser(
        description="EduBot - Educational AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py                                    # Interactive mode
  python chat.py "What programs does OAU offer?"   # Single query
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (if not provided, starts interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Initialize EduBot
    print("🤖 Initializing EduBot...")
    bot = EduBot()
    
    if args.question:
        # Single query mode
        result = bot.chat(args.question)
        print(f"\n🔍 Query: {args.question}")
        print(f"🧠 Query Type: {result['query_type']}")
        print(f"📚 Knowledge Base Used: {'Yes' if result['knowledge_base_used'] else 'No'}")
        if result['sources_count'] > 0:
            print(f"📄 Sources Found: {result['sources_count']}")
        print("-" * 60)
        print(f"🤖 EduBot: {result['response']}")
    else:
        # Interactive mode
        print("\n🤖 EduBot - Educational AI Assistant")
        print("Type 'quit', 'exit', or 'bye' to exit")
        print("Type 'status' to see system status")
        print("Type 'clear' to clear conversation history")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Have a great day!")
                    break
                
                if user_input.lower() == 'status':
                    status = bot.get_system_status()
                    print(f"\n📊 System Status:")
                    print(f"   Knowledge Base: {'✅ Available' if status['knowledge_base_available'] else '❌ Not Available'}")
                    print(f"   Conversation Length: {status['conversation_history_length']} exchanges")
                    print(f"   Current Topic: {status.get('current_topic', 'None')}")
                    continue
                
                if user_input.lower() == 'clear':
                    bot.clear_conversation()
                    print("🧹 Conversation history cleared")
                    continue
                
                if not user_input:
                    print("Please ask a question or type 'quit' to exit.")
                    continue
                
                # Get response
                result = bot.chat(user_input)
                
                # Display debug info
                print(f"\n🔍 Query Type: {result['query_type']}")
                print(f"📚 Knowledge Base Used: {'Yes' if result['knowledge_base_used'] else 'No'}")
                if result['sources_count'] > 0:
                    print(f"📄 Sources Found: {result['sources_count']}")
                print("-" * 50)
                
                # Display response
                print(f"🤖 EduBot: {result['response']}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Have a great day!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
