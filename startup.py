#!/usr/bin/env python3
"""
Startup script to pre-download the model and warm up the application
Run this before starting the main app to avoid timeout issues
"""

import os
import logging
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload_model():
    """Pre-download and cache the embedding model"""
    try:
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Pre-loading model: {model_name}")
        
        # This will download and cache the model
        model = SentenceTransformer(model_name)
        
        # Test the model with a simple encoding
        test_embedding = model.encode("test sentence")
        logger.info(f"Model loaded successfully. Embedding dimension: {len(test_embedding)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error preloading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = preload_model()
    if success:
        logger.info("Model preloading completed successfully")
        exit(0)
    else:
        logger.error("Model preloading failed")
        exit(1)
