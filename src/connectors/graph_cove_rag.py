"""RAG workflow with Chain-of-Verification (CoVe).

    supervisor -> tag_relevance -> summarize_relevant_parts -> formulate_baseline_response
               -> generate_verification_questions -> execute_verification_questions
               -> verify_answer_quality -> (verified: end | not verified: one revision, then end)

Every node returns to the supervisor, which decides the next step.
"""
import logging
import re
from typing import Any, List, Optional

from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class RAGState(BaseModel):
    query: str
    rag_context: Any
    baseline_response: Optional[str] = None
    verification_questions: Optional[Any] = None
    verification_answers: Optional[Any] = None
    relevant_chunks: Optional[Any] = None
    summarized_chunks: Optional[Any] = None
    final_response: Optional[str] = None
    loop_counter: int = 0
    stop_loop: bool = False
    is_verified: Optional[bool] = None
    last_node: str = ""
    next_node: str = ""


class SupervisorNode(Runnable):
    def invoke(self, state: RAGState, config=None) -> RAGState:
        if state.stop_loop or state.final_response is not None:
            state.next_node = "__end__"
            return state

        if state.loop_counter == 0:
            state.loop_counter = 1
            state.next_node = "tag_relevance"
            return state

        if state.last_node == "tag_relevance":
            state.next_node = "summarize_relevant_parts"
        elif state.last_node == "summarize_relevant_parts":
            state.next_node = "formulate_baseline_response"
        elif state.last_node == "formulate_baseline_response" and state.loop_counter == 2:
            # Revised answer after one verification round: stop here
            state.final_response = state.baseline_response
            state.next_node = "__end__"
        elif state.last_node == "formulate_baseline_response":
            state.next_node = "generate_verification_questions"
        elif state.last_node == "generate_verification_questions":
            state.next_node = "execute_verification_questions"
        elif state.last_node == "execute_verification_questions":
            state.next_node = "verify_answer_quality"
        elif state.last_node == "verify_answer_quality":
            if state.is_verified:
                state.final_response = state.baseline_response
                state.next_node = "__end__"
            else:
                state.loop_counter += 1
                state.next_node = "formulate_baseline_response"
        else:
            state.next_node = "tag_relevance"
        return state


def extract_tagged_sections(text: str) -> List[str]:
    return [s.strip() for s in re.findall(r"<relevant_section>(.*?)</relevant_section>", text, re.DOTALL)]


def is_bullet_list(text: str) -> bool:
    bullet_starters = ("- ", "* ", "• ", "1. ", "•", "▪")
    lines = text.strip().split("\n")
    if not lines:
        return False
    return sum(1 for line in lines if line.strip().startswith(bullet_starters)) >= max(2, len(lines) // 2)


class TagRelevanceNode(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: RAGState, config=None) -> RAGState:
        prompt = f"""
You are a helpful assistant. Given the context and a user query, identify relevant content.

[Instructions]
1. Identify every sentence or paragraph that helps answer the query.
2. Surround each relevant snippet with <relevant_section> and </relevant_section> tags.
3. Try to provide as much relevant detail as possible.

[Query]
{state.query}

[Context]
{state.rag_context}

[Output Format]
<relevant_section>...</relevant_section>

Return only the tagged relevant parts.
"""
        result = self.llm.invoke(prompt)
        state.relevant_chunks = extract_tagged_sections(result.content)
        state.last_node = "tag_relevance"
        logger.info("[tag_relevance] %d relevant chunks", len(state.relevant_chunks))
        return state


class SummarizeRelevantPartsNode(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: RAGState, config=None) -> RAGState:
        template = """
You are an assistant summarizing relevant content to help answer a query.

[Query: so you know how much detail to include]
{query}

[Instructions]
- If the input is already a bullet list or very short, return it as-is.
- Otherwise, summarize it in 1-2 clear sentences and keep the key information.

[Chunk]
{chunk}
"""
        summarized = []
        for chunk in state.relevant_chunks or []:
            if is_bullet_list(chunk) or len(chunk.split()) < 30:
                summarized.append(chunk.strip())
            else:
                summary = self.llm.invoke(template.format(query=state.query, chunk=chunk.strip()))
                summarized.append(summary.content.strip())
        state.summarized_chunks = summarized
        state.last_node = "summarize_relevant_parts"
        logger.info("[summarize] %d chunks kept", len(summarized))
        return state


class FormulateBaselineResponseNode(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: RAGState, config=None) -> RAGState:
        context = "\n\n".join(f"- {c.strip()}" for c in state.summarized_chunks or [] if c.strip())
        if state.loop_counter == 1:
            prompt = f"""
You are a helpful assistant tasked with answering the following user query.

[Query]
{state.query}

[Context extracted from documents]
{context or "(no relevant context found)"}

[Instructions]
- Use only the information in the context.
- If the context does not contain the answer, say you could not find it in the documents.
- Be concise but informative.

[Answer]
"""
        else:
            prompt = f"""
You are a trustworthy assistant. You are given a question, a baseline response and answered verification questions.
Generate a revised response that incorporates the verification results.

[Question]
{state.query}

[Baseline Response]
{state.baseline_response}

[Verification Questions and Answers]
{state.verification_answers or []}

[Instructions]
- Be more precise and address any gaps.
- Do not add facts that are not supported by the verification answers or the baseline.

[Answer]
"""
        state.baseline_response = self.llm.invoke(prompt).content.strip()
        state.last_node = "formulate_baseline_response"
        return state


class GenerateVerificationQuestionsNode(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: RAGState, config=None) -> RAGState:
        prompt = f"""
You are a helpful assistant tasked with evaluating the completeness of a generated answer. Generate a list of
yes/no verification questions that help check whether the original response contains mistakes, referring
exactly to the Base Context.

[Original Question]
{state.query}

[Original response]
{state.baseline_response}

[Base Context]
{state.rag_context}

[Instructions]
- Analyze if the answer fully addresses the question.
- Identify any parts that are vague, incomplete, or assume unsupported facts.
- Create 1 to 3 verification questions that would help confirm or expand the answer.
- If everything is already complete and certain, return exactly "NO VERIFICATION NEEDED".

[Verification Questions]
"""
        raw = self.llm.invoke(prompt).content.strip()
        state.verification_questions = (
            []
            if "NO VERIFICATION NEEDED" in raw.upper()
            else [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]
        )
        state.last_node = "generate_verification_questions"
        logger.info("[verify] %d verification questions", len(state.verification_questions))
        return state


class ExecuteVerificationQuestionsNode(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: RAGState, config=None) -> RAGState:
        if not state.verification_questions:
            state.verification_answers = []
            state.last_node = "execute_verification_questions"
            return state

        prompt = f"""
You are helping a trustworthy assistant. Answer each verification question in turn with yes/no, using ONLY the
Base Context, and point out any inconsistency or mistake in the original response.

[Question]
{state.query}

[Original response]
{state.baseline_response}

[Base Context]
{state.rag_context}

[Verification Questions]
{state.verification_questions}

[Verification Answers for each question]
"""
        state.verification_answers = self.llm.invoke(prompt).content.strip()
        state.last_node = "execute_verification_questions"
        return state


class VerifyAnswerQualityNode(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, state: RAGState, config=None) -> RAGState:
        if not state.verification_answers:
            state.is_verified = True
            state.last_node = "verify_answer_quality"
            return state

        prompt = f"""
You are assessing the completeness of an answer to a user's query.

[Original Query]
{state.query}

[Original answer]
{state.baseline_response}

[Verification Answers]
{state.verification_answers}

[Instructions]
- Determine if the original answer is complete and accurate.
- Answer with "YES" if it is sufficient, or "NO" if more information or a reformulation is needed.

[Final Assessment]
"""
        assessment = self.llm.invoke(prompt).content.strip().upper()
        state.is_verified = assessment.startswith("YES")
        state.last_node = "verify_answer_quality"
        logger.info("[verify] answer verified: %s", state.is_verified)
        return state


NODES = [
    "tag_relevance",
    "summarize_relevant_parts",
    "formulate_baseline_response",
    "generate_verification_questions",
    "execute_verification_questions",
    "verify_answer_quality",
]


class ChatbotGraphOrchestrator_RAG:
    def __init__(self, llm):
        self.llm = llm
        self.executable = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("supervisor", SupervisorNode())
        workflow.add_node("tag_relevance", TagRelevanceNode(self.llm))
        workflow.add_node("summarize_relevant_parts", SummarizeRelevantPartsNode(self.llm))
        workflow.add_node("formulate_baseline_response", FormulateBaselineResponseNode(self.llm))
        workflow.add_node("generate_verification_questions", GenerateVerificationQuestionsNode(self.llm))
        workflow.add_node("execute_verification_questions", ExecuteVerificationQuestionsNode(self.llm))
        workflow.add_node("verify_answer_quality", VerifyAnswerQualityNode(self.llm))

        workflow.set_entry_point("supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state.next_node,
            {**{n: n for n in NODES}, "__end__": END},
        )
        for node in NODES:
            workflow.add_edge(node, "supervisor")
        return workflow.compile()

    def run(self, question: str, rag_context: str) -> dict:
        """Return the final state as a dict (final_response, is_verified, ...)."""
        return self.executable.invoke(RAGState(query=question, rag_context=rag_context))
