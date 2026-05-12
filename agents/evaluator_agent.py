from dataclasses import dataclass
import json
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from models.fallback import get_model_with_fallback, invoke_with_retry
from utils.logger import get_logger

logger = get_logger(__name__)

EVALUATOR_PROMPT = """You are a High-Resolution QA Auditor. Your goal is to find even the smallest flaws in the AI's response.
Evaluate the response based on the Question and Context provided.

Question: {question}
Context: {context}
AI Answer: {answer}

Return a JSON object with this exact structure:
{{
  "answer_relevance": <0.0-1.0>,
  "faithfulness": <0.0-1.0>,
  "completeness": <0.0-1.0>,
  "overall_score": <0.0-1.0>,
  "critique": "High-level summary",
  "specific_errors": ["Error 1", "Error 2"],
  "missed_requirements": ["Requirement 1"],
  "fix_instructions": "Step-by-step guide for the AI to fix this response",
  "needs_retry": <true|false>
}}

Stricter Scoring Rules:
- If there is ANY technical hallucination, faithfulness must be < 0.5.
- If the AI missed even one part of a multi-part question, completeness must be < 0.7.
- ARCHITECTURE DEPTH: If a Mermaid diagram is requested, it MUST have at least 8+ nodes and show sub-level interactions. If it is too simple, set completeness < 0.6 and needs_retry to true.
- Set needs_retry to true if overall_score is below 0.8 OR if there are critical technical errors."""

@dataclass
class EvalResult:
    answer_relevance: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    overall_score: float = 0.0
    critique: str = ""
    specific_errors: list[str] = None
    missed_requirements: list[str] = None
    fix_instructions: str = ""
    needs_retry: bool = False

    def __post_init__(self):
        if self.specific_errors is None: self.specific_errors = []
        if self.missed_requirements is None: self.missed_requirements = []

def evaluate_response(
    question: str,
    answer: str,
    context: str = "",
) -> EvalResult:
    """Score an AI response across 4 quality dimensions."""
    if not answer or not answer.strip():
        logger.warning("evaluator_short_circuit: Received empty answer, returning safe default.")
        return EvalResult(
            overall_score=0.0,
            critique="CRITIQUE: The provided answer was completely empty.",
            specific_errors=["Empty Answer"],
            missed_requirements=["Provide an answer"],
            fix_instructions="Please generate a valid text response.",
            needs_retry=True
        )
    try:
        # Switch to a more powerful model for evaluation stability
        llm = get_model_with_fallback("groq/llama-3.3-70b-versatile", temperature=0.0)
        prompt = EVALUATOR_PROMPT.format(
            question=question,
            context=context[:2000] if context else "No retrieved context.",
            answer=answer[:2000],
        )
        response = invoke_with_retry(llm, prompt)
        raw = response.content.strip()

        # Robust JSON extraction
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        
        try:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                cleaned_raw = re.sub(r'[\x00-\x1F\x7F]', '', raw)
                data = json.loads(cleaned_raw)
            # Ensure all expected keys exist to avoid EvalResult init errors
            cleaned_data = {
                "answer_relevance": float(data.get("answer_relevance", 0.0)),
                "faithfulness": float(data.get("faithfulness", 0.0)),
                "completeness": float(data.get("completeness", 0.0)),
                "overall_score": float(data.get("overall_score", 0.0)),
                "critique": str(data.get("critique", "No critique provided.")),
                "specific_errors": list(data.get("specific_errors", [])),
                "missed_requirements": list(data.get("missed_requirements", [])),
                "fix_instructions": str(data.get("fix_instructions", "")),
                "needs_retry": bool(data.get("needs_retry", False))
            }
            return EvalResult(**cleaned_data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("eval_parse_failed", raw=raw, error=str(e))
            return EvalResult(
                overall_score=0.0, 
                critique=f"Auditor returned invalid JSON. Raw output:\n{raw}\n\nError: {str(e)}", 
                needs_retry=False
            )

    except Exception as e:
        logger.error("evaluator_failed", error=str(e))
        return EvalResult(overall_score=0.0, critique=f"Internal Auditor Error: {str(e)}", needs_retry=False)

def evaluator_node(state: dict) -> dict:
    """LangGraph node — evaluates the last AI message."""
    messages = state.get("messages", [])
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]

    if not ai_messages or not human_messages:
        return {"eval_score": 1.0, "eval_feedback": ""}

    question = human_messages[-1].content
    
    # Need to extract just the text if content is a list
    if isinstance(question, list):
        question = next((item["text"] for item in question if item.get("type") == "text"), "")

    answer = ai_messages[-1].content
    context = state.get("rag_context", "")

    result = evaluate_response(question, answer, context)
    
    # Compile a high-resolution feedback string for the next agent
    detailed_feedback = f"CRITIQUE: {result.critique}\n"
    if result.specific_errors:
        detailed_feedback += f"SPECIFIC ERRORS: {', '.join(result.specific_errors)}\n"
    if result.missed_requirements:
        detailed_feedback += f"MISSED REQUIREMENTS: {', '.join(result.missed_requirements)}\n"
    if result.fix_instructions:
        detailed_feedback += f"FIX INSTRUCTIONS: {result.fix_instructions}\n"

    logger.info(
        "eval_complete",
        score=result.overall_score,
        needs_retry=result.needs_retry,
    )

    current_retries = state.get("retry_count", 0)

    return {
        "eval_score": result.overall_score,
        "eval_feedback": detailed_feedback,
        "retry_count": current_retries + 1,
        "agent_trace": state.get("agent_trace", []) + [f"eval:{result.overall_score:.2f}"],
    }
