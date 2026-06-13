from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

# --- 1. Server Setup ---
app = FastAPI()

# Securely load API keys from environment variables
LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
INDEX_NAME = "medium-rag"
CHAT_MODEL = "ZYRANGG-gpt-5-mini"
EMBEDDING_MODEL = "ZYRANGG-text-embedding-3-small"

# Initialize clients (Vercel will provide the keys at runtime)
if LLMOD_API_KEY and PINECONE_API_KEY:
    llm_client = OpenAI(api_key=LLMOD_API_KEY, base_url="https://api.llmod.ai/v1")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

# --- 2. Data Models ---
# Defines the exact structure the user will send to the POST endpoint
class PromptRequest(BaseModel):
    question: str

# --- 3. API Endpoints ---

@app.post("/api/prompt")
async def handle_prompt(request: PromptRequest):
    """Handles incoming queries, retrieves context, and generates an answer."""
    query = request.question
    
    try:
        # Step A: Retrieval
        query_response = llm_client.embeddings.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_vector = query_response.data[0].embedding
        
        search_results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )
        
        retrieved_chunks = []
        for match in search_results['matches']:
            retrieved_chunks.append({
                "article_id": match['metadata'].get('article_id', 'unknown'),
                "title": match['metadata'].get('title', 'Untitled'),
                "chunk": match['metadata'].get('text', ''),
                "score": match['score']
            })
            
        # Step B: LLM Generation
        context_text = ""
        for idx, item in enumerate(retrieved_chunks):
            context_text += f"--- Article {idx + 1}: {item['title']} ---\n{item['chunk']}\n\n"

        system_prompt = (
            "You are a Medium-article assistant that answers questions strictly and only "
            "based on the Medium articles dataset context provided to you (metadata "
            "and article passages).\n"
            "You must not use any external knowledge, the open "
            "internet, or information that is not explicitly contained in the retrieved "
            "context. If the answer cannot be determined from the provided context, "
            "respond: \"I don't know based on the provided Medium articles data.\"\n"
            "Always explain your answer using the given context, quoting or "
            "paraphrasing the relevant article passage or metadata when helpful."
        )

        user_message = f"Context:\n{context_text}\nQuestion: {query}"

        response = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        
        final_answer = response.choices[0].message.content
        
        # Step C: Return required JSON structure
        return {
            "response": final_answer,
            "context": retrieved_chunks,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_message
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Returns the hyperparameter configuration of the RAG system."""
    # We used chunk size 500 and overlap 100 in the notebook. 
    # Overlap ratio = 100 / 500 = 0.2.
    return {
        "chunk_size": 500,
        "overlap_ratio": 0.2,
        "top_k": 3
    }