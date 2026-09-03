import json
import os
from typing import List, Dict

from pdfchat.config import configure_api_key, validate_api_key, make_api_request
from pdfchat.embeddings import create_vector_store
from pdfchat.qa import get_answer

# Sample test text simulating a short PDF
SAMPLE_DOCUMENT_TEXT = """
The Solar System consists of the Sun and the objects that orbit it.
The largest of these are the eight planets, which form two main groups: 
the inner terrestrial planets (Mercury, Venus, Earth, and Mars), and the outer giant planets (Jupiter, Saturn, Uranus, and Neptune).
Jupiter is the largest planet in the solar system, and Mercury is the smallest.
The asteroid belt lies between the orbits of Mars and Jupiter.
"""

TEST_QUESTIONS = [
    {
        "question": "Which is the largest planet?",
        "expected_facts": ["Jupiter"]
    },
    {
        "question": "Where is the asteroid belt located?",
        "expected_facts": ["between Mars and Jupiter"]
    }
]

def llm_judge(question: str, context: str, answer: str) -> Dict:
    """Uses Groq API as a judge to evaluate the response."""
    prompt = f"""
Evaluate the following answer based on the given question and context.
You must return your evaluation in strict JSON format with exactly these two keys:
- "is_context_relevant": true or false
- "is_answer_correct": true or false

Question: {question}
Context: {context}
Answer: {answer}

JSON:
"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }

    try:
        response = make_api_request("POST", "/chat/completions", headers=headers, json=payload, timeout=60)
        data = response.json()
        content = data.get("choices", [])[0].get("message", {}).get("content", "{}")
        return json.loads(content)
    except Exception as e:
        print(f"Error evaluating with LLM: {e}")
        return {"is_context_relevant": False, "is_answer_correct": False}

def run_evals():
    print("Configuring API key...")
    configure_api_key()
    validate_api_key()

    print("Building hybrid retriever and cross-encoder...")
    retriever = create_vector_store(SAMPLE_DOCUMENT_TEXT)

    results = []
    for test in TEST_QUESTIONS:
        question = test["question"]
        print(f"\nEvaluating Question: {question}")
        
        # 1. Retrieve
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        
        # 2. Answer
        answer = get_answer(question, retriever)
        
        # 3. Evaluate
        eval_result = llm_judge(question, context, answer)
        
        print(f"Answer: {answer}")
        print(f"Eval: {eval_result}")
        results.append(eval_result)

    total_relevant = sum(1 for r in results if r.get("is_context_relevant"))
    total_correct = sum(1 for r in results if r.get("is_answer_correct"))

    print(f"\n--- EVALUATION SUMMARY ---")
    print(f"Total Questions: {len(TEST_QUESTIONS)}")
    print(f"Context Relevance: {total_relevant}/{len(TEST_QUESTIONS)}")
    print(f"Answer Correctness: {total_correct}/{len(TEST_QUESTIONS)}")

if __name__ == "__main__":
    run_evals()
