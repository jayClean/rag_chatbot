# rag_pipeline.py
from loaders.pdf_loader import extract_pdf_chunks
from loaders.web_scraper import extract_website_chunks
from embedder import embed_text
from retriever import add_to_index
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_pdf(filename: str, file_bytes: bytes):
    try:
        chunks = extract_pdf_chunks(filename, file_bytes)
        for chunk in chunks:
            emb = embed_text(chunk["text"])
            add_to_index(emb, chunk)
        return chunks
    except Exception as e:
        logger.error(f"Error ingesting PDF {filename}: {str(e)}")
        raise Exception(f"Failed to ingest PDF: {str(e)}")

def ingest_website(url: str):
    try:
        chunks = extract_website_chunks(url)
        for chunk in chunks:
            emb = embed_text(chunk["text"])
            add_to_index(emb, chunk)
        return chunks
    except Exception as e:
        logger.error(f"Error ingesting website {url}: {str(e)}")
        raise Exception(f"Failed to ingest website: {str(e)}")

def ask_llm_with_context(question: str, context_chunks: list):
    try:
        # Sort chunks by relevance if they have a score
        if context_chunks and "score" in context_chunks[0]:
            context_chunks = sorted(context_chunks, key=lambda x: x.get("score", 0), reverse=True)
        
        # Limit context size to avoid exceeding token limits
        max_context_length = 1500  # Conservative limit to leave room for prompt and answer
        context_text = ""
        for chunk in context_chunks:
            chunk_text = chunk["text"] + "\n\n"
            # If adding this chunk would exceed our limit, stop adding chunks
            if len(context_text) + len(chunk_text) > max_context_length:
                break
            context_text += chunk_text
        
        prompt = f"""Answer the question using the context below. If the answer isn't in the context, reply with "I don't know".

        Context:
        {context_text.strip()}

        Question: {question}
        Answer:"""

        # Try with a fallback model if the primary model isn't available
        models_to_try = ["mistral"]
        
        for model in models_to_try:
            try:
                logger.info(f"Attempting to use model: {model}")
                res = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False  # Explicitly disable streaming
                    },
                    timeout=120
                )
                
                if res.status_code == 200:
                    try:
                        response_json = res.json()
                        if "response" in response_json:
                            return response_json["response"]
                        else:
                            logger.warning(f"Unexpected response format from model {model}: {response_json}")
                    except ValueError as json_err:
                        # Handle case where response isn't valid JSON
                        logger.warning(f"Could not parse JSON response: {str(json_err)}")
                        # Try to extract text content directly
                        if "response" in res.text:
                            import re
                            match = re.search(r'"response"\s*:\s*"([^"]*)"', res.text)
                            if match:
                                return match.group(1)
                        logger.error(f"Raw response: {res.text[:200]}...")
                        continue
                elif res.status_code == 404 and "not found" in res.text:
                    logger.warning(f"Model {model} not found, trying next model")
                    continue
                else:
                    logger.error(f"LLM API returned status code {res.status_code}: {res.text}")
            except requests.RequestException as e:
                logger.warning(f"Request error with model {model}: {str(e)}")
                continue
        
        # If we've tried all models and none worked
        raise Exception("No available LLM models found. Please pull a model using 'ollama pull mistral' or another supported model.")

    except requests.RequestException as e:
        logger.error(f"Request error when calling LLM API: {str(e)}")
        raise Exception(f"Failed to communicate with LLM: {str(e)}")
    except Exception as e:
        logger.error(f"Error in ask_llm_with_context: {str(e)}")
        raise Exception(f"Error processing question: {str(e)}")