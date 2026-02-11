"""
LLM-powered answer generation with citation tracking.
Supports OpenAI API and local Ollama models.
"""

import json
from typing import Optional

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    Each chunk is labeled with a source number for citation tracking.
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source = chunk["metadata"].get("source_label", f"Source {i+1}")
        page = chunk["metadata"].get("page_number", "?")
        doc = chunk["metadata"].get("document_name", "Unknown")

        context_parts.append(
            f"[Source {i+1}] (Document: {doc}, Page {page})\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def build_prompt(
    query: str,
    context: str,
    conversation_history: list[dict] = None,
    system_prompt: str = None,
) -> list[dict]:
    """
    Build the message list for the LLM.
    
    Args:
        query: User's question
        context: Formatted context from retrieved chunks
        conversation_history: Previous Q&A turns for follow-up questions
        system_prompt: System instructions for the LLM
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful document assistant. Answer questions based ONLY on the "
            "provided context passages. For each claim in your answer, cite the source "
            "using [Source X] format. If the answer is not found in the context, say "
            "\"I could not find this information in the provided documents.\" "
            "Be concise and accurate."
        )

    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history for follow-up context
    if conversation_history:
        for turn in conversation_history[-5:]:  # Last 5 turns
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

    # Current query with context
    user_message = f"""Based on the following context passages, answer the question.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the context above
- Cite sources using [Source X] format for each claim
- If the information is not in the context, say so
- Be concise and direct"""

    messages.append({"role": "user", "content": user_message})

    return messages


class AnswerGenerator:
    """Generate cited answers using an LLM."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        config_path: str = "configs/config.yaml",
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "ollama":
            config = load_config(config_path)
            base_url = config["generation"]["ollama"]["base_url"]
            self.model = config["generation"]["ollama"]["model"]
            from openai import OpenAI
            self.client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        print(f"Generator initialized: {provider}/{self.model}")

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict] = None,
        system_prompt: str = None,
    ) -> dict:
        """
        Generate a cited answer from retrieved chunks.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved chunk dicts
            conversation_history: Previous Q&A for context
            system_prompt: Custom system instructions
        
        Returns:
            Dict with 'answer', 'sources', 'source_map'
        """
        if not retrieved_chunks:
            return {
                "answer": "I could not find any relevant information in the uploaded documents.",
                "sources": [],
                "source_map": {},
            }

        # Format context and build prompt
        context = format_context(retrieved_chunks)
        messages = build_prompt(query, context, conversation_history, system_prompt)

        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content

        # Build source map for citations
        source_map = {}
        for i, chunk in enumerate(retrieved_chunks):
            source_key = f"Source {i+1}"
            source_map[source_key] = {
                "document": chunk["metadata"].get("document_name", "Unknown"),
                "page": chunk["metadata"].get("page_number", "?"),
                "text_preview": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                "relevance_score": chunk.get("rerank_score", chunk.get("score", 0)),
            }

        # Extract which sources were actually cited in the answer
        cited_sources = []
        for key in source_map:
            if f"[{key}]" in answer:
                cited_sources.append(source_map[key])

        return {
            "answer": answer,
            "sources": cited_sources,
            "source_map": source_map,
            "all_retrieved": retrieved_chunks,
        }
