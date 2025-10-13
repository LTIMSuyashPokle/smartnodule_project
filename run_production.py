# ========================================================================
# 6. STARTUP SCRIPT FOR COMPLETE SYSTEM
# ========================================================================

# run_production.py
"""
Complete production startup script for SmartNodule Module 4
"""

import uvicorn
import logging
import asyncio
import signal
import sys
from pathlib import Path

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smartnodule_production.log'),
        logging.StreamHandler()
    ]
)

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    logging.info("🛑 Received shutdown signal, stopping services...")
    sys.exit(0)

def main():
    """Main startup function"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("🚀 Starting SmartNodule Production System...")
    
    # Check required files
    required_files = [
        "smartnodule_memory_optimized_best.pth",
        "case_retrieval/faiss_index.idx",
        "case_retrieval/case_metadata.pkl"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            logging.error(f"❌ Required file not found: {file_path}")
            sys.exit(1)
    
    logging.info("✅ All required files found")
    
    # Start FastAPI server
    try:
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=1,
            log_level="info"
        )
    except Exception as e:
        logging.error(f"❌ Failed to start server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
