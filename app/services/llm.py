import google.generativeai as genai
from app.core.config import settings

genai.configure(api_key=settings.gemini_api_key)

model = genai.GenerativeModel("gemini-2.5-flash")


async def generate_answer(
    query: str,
    context_chunks: list[str],
    history: list[dict],
) -> str:
    """
    Generate an answer using Gemini with retrieved context and chat history.
    This is the core RAG step — no RetrievalQAChain used.

    Args:
        query: Current user question
        context_chunks: Top-k similar chunks from Qdrant
        history: Previous messages from Redis

    Returns:
        LLM generated answer string
    """
    # Format retrieved context
    context = "\n\n".join(
        [f"[Chunk {i+1}]:\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    # Format chat history for prompt
    history_text = ""
    if history:
        history_text = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in history]
        )
        history_text = f"\n\nConversation History:\n{history_text}"

    # Build full prompt manually (no RetrievalQAChain)
    prompt = f"""You are a helpful AI assistant. Answer the user's question
based ONLY on the provided context below.
If the answer is not found in the context, say:
"I couldn't find relevant information in the uploaded documents."
Do NOT make up information.

Context from documents:
{context}
{history_text}

User Question: {query}

Answer:"""

    response = model.generate_content(prompt)
    return response.text.strip()
