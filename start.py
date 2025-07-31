#!/usr/bin/env python3
"""
Universal startup script for both local development and Railway deployment
"""
import os
import sys
from pathlib import Path

# Add src to Python path (now we're in root, so add src directory)
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Railway sets PORT, local defaults to 8000)
    port = int(os.environ.get('PORT', 8000))
    
    uvicorn.run(
        "api:app",  # Import api.py from src directory
        host="0.0.0.0", 
        port=port,
        reload=os.environ.get('RAILWAY_ENVIRONMENT') != 'production'
    )
    