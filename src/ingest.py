#!/usr/bin/env python3
"""
EduBot Document Ingestion Entry Point

This script provides a simple interface for ingesting documents into the EduBot system.
It processes documents through the complete ingestion pipeline and stores them in the vector database.

Usage:
    python ingest.py <input_source> [--metadata key=value]
    
Examples:
    python ingest.py data/undergraduate_handbook.pdf
    python ingest.py https://oauife.edu.ng/admission-undergraduate-studies/
    python ingest.py data/documents/
    python ingest.py file.pdf --metadata institution="University of Lagos"
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to Python path
# sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ingestion.ingest import IngestionOrchestrator


def parse_metadata(metadata_strings):
    """Parse metadata key=value pairs."""
    metadata = {}
    for item in metadata_strings:
        if '=' not in item:
            print(f"Warning: Invalid metadata format '{item}'. Use key=value format.")
            continue
        key, value = item.split('=', 1)
        metadata[key.strip()] = value.strip()
    return metadata


def main():
    """Main entry point for document ingestion."""
    
    parser = argparse.ArgumentParser(
        description="Ingest documents into the EduBot knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py data/undergraduate_handbook.pdf
  python ingest.py https://oauife.edu.ng/admission-undergraduate-studies/
  python ingest.py data/documents/
  python ingest.py file.pdf --metadata institution="University of Lagos" type="handbook"
        """
    )
    
    parser.add_argument(
        'input_source',
        help='Path to file/directory or URL to ingest'
    )
    
    parser.add_argument(
        '--metadata',
        nargs='*',
        default=[],
        help='Additional metadata in key=value format'
    )
    
    args = parser.parse_args()
    
    input_source = args.input_source
    additional_metadata = parse_metadata(args.metadata) if args.metadata else None
    
    # Check input type and validate
    if input_source.startswith(('http://', 'https://')):
        print(f"🚀 Starting ingestion of URL: {input_source}")
    elif os.path.isdir(input_source):
        print(f"🚀 Starting ingestion of directory: {input_source}")
    elif os.path.isfile(input_source):
        print(f"🚀 Starting ingestion of file: {input_source}")
    else:
        print(f"❌ Error: Input source '{input_source}' not found.")
        print("Please provide a valid file path, directory path, or URL.")
        sys.exit(1)
    
    if additional_metadata:
        print(f"📋 Additional metadata: {additional_metadata}")
    
    print("=" * 60)
    
    try:
        # Initialize orchestrator and run ingestion
        orchestrator = IngestionOrchestrator()
        results = orchestrator.ingest(input_source, additional_metadata)
        
        # Print success message with details
        print("=" * 60)
        print("✅ Ingestion completed successfully!")
        print(f"📄 Source processed: {input_source}")
        print(f"📚 Documents ingested: {results['documents_loaded']}")
        print(f"🧩 Total chunks created: {results['total_nodes_created']}")
        print(f"🔍 Searchable chunks: {results['leaf_nodes_created']}")
        print(f"💾 Total knowledge base size: {results['storage']['docstore_total_nodes']} chunks")
        print("🗄️ Data stored in Qdrant vector database")
        print("💬 Ready for chat queries!")
        
    except ValueError as e:
        print("=" * 60)
        print(f"❌ Ingestion failed: {str(e)}")
        
        # Provide troubleshooting tips
        print("\n🔧 Troubleshooting tips:")
        if "network" in str(e).lower() or "connection" in str(e).lower():
            print("- Check your internet connection")
            print("- Verify the URL is accessible in a web browser")
            print("- Try again in a few minutes")
        elif "invalid url" in str(e).lower():
            print("- Ensure the URL starts with http:// or https://")
            print("- Check for typos in the URL")
            print("- Verify the website is currently online")
        elif "no content" in str(e).lower():
            print("- The file or URL might be empty")
            print("- Check file permissions")
            print("- Verify the document format is supported")
        else:
            print("- Check file/directory permissions")
            print("- Ensure the path is correct")
            print("- Verify your API keys are set correctly")
        
        sys.exit(1)
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ Unexpected error during ingestion: {str(e)}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
