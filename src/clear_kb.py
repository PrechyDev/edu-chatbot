#!/usr/bin/env python3
"""
Clear Knowledge Base Script

This script clears the entire EduBot knowledge base, removing all stored documents
and vector embeddings. Use this when you want to start fresh.

Usage:
    python clear_kb.py
"""

import os
import shutil
import sys
from pathlib import Path

# Add src to Python path
# sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import PARENT_DOCSTORE_DIR, QDRANT_DB_PATH


def clear_knowledge_base():
    """Clear the entire knowledge base."""
    
    print("🧹 Clearing EduBot Knowledge Base...")
    print("=" * 50)
    
    cleared_something = False
    
    # Clear docstore
    if os.path.exists(PARENT_DOCSTORE_DIR):
        os.remove(PARENT_DOCSTORE_DIR)
        print(f"✅ Cleared docstore: {PARENT_DOCSTORE_DIR}")
        cleared_something = True
    else:
        print(f"ℹ️ No docstore found at: {PARENT_DOCSTORE_DIR}")
    
    # Clear Qdrant database
    if os.path.exists(QDRANT_DB_PATH):
        shutil.rmtree(QDRANT_DB_PATH)
        print(f"✅ Cleared vector database: {QDRANT_DB_PATH}")
        cleared_something = True
    else:
        print(f"ℹ️ No vector database found at: {QDRANT_DB_PATH}")
    
    print("=" * 50)
    
    if cleared_something:
        print("✅ Knowledge base cleared successfully!")
        print("📝 You can now start ingesting documents to build a new knowledge base.")
    else:
        print("ℹ️ Knowledge base was already empty.")
    
    print("🚀 To add documents, use: python ingest.py <file_or_url>")


def main():
    """Main entry point."""
    
    # Confirm with user
    response = input("⚠️ This will permanently delete all stored documents. Continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        clear_knowledge_base()
    else:
        print("❌ Operation cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
