import os
import json
import re
import uuid
import requests
from flask import Flask, request, jsonify
from pinecone import Pinecone
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "badal-embeddings")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # For API-based embeddings

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")
    gemini_model = None

# Global variables
index = None
parent_lookup = {}
initialized = False

def create_embedding_api(text):
    """Create embeddings using HuggingFace Inference API"""
    if not HUGGINGFACE_API_KEY:
        # Fallback to Gemini embeddings if available
        return create_gemini_embedding(text)
    
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        response = requests.post(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
            headers=headers,
            json={"inputs": text}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"HuggingFace API error: {response.status_code}")
            return create_gemini_embedding(text)
            
    except Exception as e:
        logger.error(f"Error with HuggingFace API: {str(e)}")
        return create_gemini_embedding(text)

def create_gemini_embedding(text):
    """Fallback: Create embeddings using Gemini API"""
    try:
        if not gemini_model:
            raise Exception("No embedding service available")
            
        # Use Gemini to create a simple embedding
        response = gemini_model.generate_content(f"Create a numerical representation of this text: {text}")
        # This is a simple fallback - you might want to use hash or other method
        embedding = [hash(text + str(i)) % 1000 / 1000.0 for i in range(384)]
        return embedding
        
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        # Last resort: simple hash-based embedding
        embedding = [hash(text + str(i)) % 1000 / 1000.0 for i in range(384)]
        return embedding

def initialize_models():
    """Initialize Pinecone connection only"""
    global index, initialized
    
    if initialized:
        return
        
    try:
        logger.info("Connecting to Pinecone...")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        existing_indexes = pc.list_indexes()
        if PINECONE_INDEX_NAME not in [i.name for i in existing_indexes]:
            logger.info(f"Creating index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": PINECONE_REGION
                    }
                }
            )
        
        index = pc.Index(PINECONE_INDEX_NAME)
        load_data_files()
        initialized = True
        logger.info("Pinecone initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing: {str(e)}")
        raise

def load_data_files():
    """Load parent and child data files"""
    global parent_lookup
    
    try:
        # Load parent data
        if os.path.exists("parent.json"):
            with open("parent.json", "r", encoding="utf-8") as f:
                parents = json.load(f)
            parent_lookup = {p["parent_id"]: p for p in parents}
            logger.info(f"Loaded {len(parents)} parent entries")
        else:
            logger.warning("parent.json not found")
            
    except Exception as e:
        logger.error(f"Error loading data files: {str(e)}")

def ensure_initialized():
    """Ensure models are initialized before handling requests"""
    if not initialized:
        initialize_models()

def chunk_list(lst, n):
    """Split list into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def upload_data_to_pinecone():
    """Upload child data to Pinecone with parent metadata"""
    try:
        if not os.path.exists("child.json"):
            logger.warning("child.json not found, skipping upload")
            return
            
        with open("child.json", "r", encoding="utf-8") as f:
            children = json.load(f)
        
        logger.info(f"Uploading {len(children)} child entries to Pinecone...")
        
        BATCH_SIZE = 50
        uploaded_count = 0
        
        for batch in chunk_list(children, BATCH_SIZE):
            vectors = []
            for child in batch:
                child_id = child["child_id"]
                parent_id = child["parent_id"]
                parent_obj = parent_lookup.get(parent_id, {})
                
                # Create embedding using API
                embedding = create_embedding_api(child["text"])
                
                vectors.append({
                    "id": child_id,
                    "values": embedding,
                    "metadata": {
                        "parent_id": parent_id,
                        "parent_source": parent_obj.get("source", ""),
                        "parent_title": parent_obj.get("title", ""),
                        "parent_text": parent_obj.get("text", ""),
                        "child_text": child["text"],
                        "original_data": json.dumps(child.get("original_data", {})),
                        "parent_tables": json.dumps(parent_obj.get("tables", []))
                    }
                })
            
            # Upload batch
            index.upsert(vectors=vectors)
            uploaded_count += len(vectors)
            logger.info(f"Uploaded batch: {uploaded_count}/{len(children)} vectors")
        
        logger.info(f"Successfully uploaded {uploaded_count} vectors to Pinecone")
        
    except Exception as e:
        logger.error(f"Error uploading to Pinecone: {str(e)}")

def search_similar_chunks(query, top_k=5):
    """Search for similar chunks in Pinecone"""
    try:
        # Create query embedding using API
        query_embedding = create_embedding_api(query)
        
        # Search Pinecone
        results = index.query(
            vector=query_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        return results["matches"]
        
    except Exception as e:
        logger.error(f"Error searching chunks: {str(e)}")
        return []

def generate_answer_with_gemini(query, context_chunks):
    """Generate answer using Gemini with retrieved context"""
    if not gemini_model:
        return "Gemini API is not configured. Please set GEMINI_API_KEY environment variable."
    
    try:
        # Prepare context from chunks
        context = ""
        for i, chunk in enumerate(context_chunks, 1):
            metadata = chunk["metadata"]
            context += f"Source {i}:\n"
            context += f"Title: {metadata.get('parent_title', 'N/A')}\n"
            context += f"Content: {metadata.get('child_text', '')}\n"
            if metadata.get('parent_source'):
                context += f"Source URL: {metadata.get('parent_source')}\n"
            context += "\n---\n\n"
        
        # Create prompt for Gemini
        prompt = f"""You are a helpful assistant that answers questions based on the provided context. 
        
Context:
{context}

Question: {query}

Instructions:
1. Answer the question based ONLY on the information provided in the context above
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be specific and detailed in your response
4. If you reference specific information, mention which source it came from
5. Maintain a helpful and professional tone

Answer:"""

        # Generate response
        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {str(e)}")
        return f"Error generating response: {str(e)}"

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint - don't initialize models here"""
    return jsonify({
        "status": "healthy",
        "message": "RAG API is running",
        "initialized": initialized,
        "gemini_configured": gemini_model is not None,
        "webhook_url": f"{request.url_root}webhook"
    })

@app.route("/upload", methods=["POST"])
def upload_data():
    """Endpoint to upload data to Pinecone"""
    ensure_initialized()
    try:
        upload_data_to_pinecone()
        return jsonify({"message": "Data uploaded successfully to Pinecone"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    """Search for similar chunks"""
    ensure_initialized()
    try:
        data = request.get_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        chunks = search_similar_chunks(query, top_k)
        
        # Format response
        results = []
        for chunk in chunks:
            results.append({
                "score": chunk["score"],
                "content": chunk["metadata"]["child_text"],
                "parent_title": chunk["metadata"].get("parent_title", ""),
                "parent_source": chunk["metadata"].get("parent_source", "")
            })
        
        return jsonify({"results": results})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    """Main RAG endpoint - search and generate answer"""
    ensure_initialized()
    try:
        data = request.get_json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        # Search for relevant chunks
        chunks = search_similar_chunks(query, top_k)
        
        if not chunks:
            return jsonify({
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": []
            })
        
        # Generate answer with Gemini
        answer = generate_answer_with_gemini(query, chunks)
        
        # Prepare sources
        sources = []
        for chunk in chunks:
            sources.append({
                "title": chunk["metadata"].get("parent_title", ""),
                "source": chunk["metadata"].get("parent_source", ""),
                "content": chunk["metadata"]["child_text"][:200] + "...",  # Preview
                "relevance_score": chunk["score"]
            })
        
        return jsonify({
            "answer": answer,
            "sources": sources,
            "query": query
        })
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    """Webhook endpoint for external integrations"""
    ensure_initialized()
    try:
        data = request.get_json()
        
        # Log webhook data
        logger.info(f"Webhook received: {json.dumps(data, indent=2)}")
        
        # Handle different webhook events
        event_type = data.get("type", "unknown")
        
        if event_type == "query":
            query = data.get("query", "")
            if query:
                # Process query through RAG
                chunks = search_similar_chunks(query, 3)
                answer = generate_answer_with_gemini(query, chunks)
                
                return jsonify({
                    "status": "success",
                    "answer": answer,
                    "event_type": event_type
                })
        
        return jsonify({
            "status": "received",
            "event_type": event_type,
            "message": "Webhook processed successfully"
        })
        
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # For local development
    initialize_models()
    
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
