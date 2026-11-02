"""RAG workflow: ReAct agent with a search + rerank tool, then an LLM judge.

    react_agent (similarity_search -> LLM rerank, up to 5 steps) -> judge (score 0-10 + remarks) -> end

The judge sees the context the agent actually retrieved, so its score measures grounding,
not only fluency.
"""
import logging
from typing import Optional

from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RAGState(BaseModel):
    query: str
    rag_context: Optional[str] = ""
    response: Optional[str] = ""
    score: Optional[float] = None
    remarks: Optional[str] = ""


class JudgeOutput(BaseModel):
    score: float = Field(..., description="Score between 0 and 10")
    remarks: str = Field(..., description="Remarks explaining the score")


class SimilaritySearchTool:
    """Retrieve k chunks, then let an LLM reorder them by relevance and drop irrelevant ones."""

    def __init__(self, retriever, reranker_llm, k: int = 8):
        self.retriever = retriever
        self.reranker_llm = reranker_llm
        self.k = k

    def __call__(self, query: str) -> str:
        docs = self.retriever.similarity_search(query, k=self.k)
        if not docs:
            return "No documents found."
        chunks = "\n\n".join(
            f"[{i}] (source: {doc.metadata.get('source', 'unknown')})\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        )
        rerank_prompt = f"""You are a reranking expert in RAG pipelines. Reorder the chunks below by relevance to the query.

Query: {query}

Chunks:
{chunks}

RULES:
- Return the relevant chunks, most relevant first, each starting with its [number] and source line.
- Do not modify or shorten the text of a chunk.
- Exclude chunks that are not relevant. If none is relevant, return "No relevant documents."
"""
        reranked = self.reranker_llm.invoke([SystemMessage(content=rerank_prompt)])
        return reranked.content.strip()


REACT_TEMPLATE = """You are a Retrieval-Augmented Reasoning Agent.

Your goal is to retrieve all necessary information from a knowledge base to answer the user's query
with completeness and accuracy.

## TOOLS AVAILABLE
{tools}

You MUST:
- Think step by step.
- Generate useful keywords for retrieval.
- Use the `similarity_search` tool to gather relevant documents.
- Review the results and assess whether the retrieved information is sufficient.
- Only state facts supported by the retrieved documents, and cite their sources.
- If the documents do not contain the answer, say so.

## RESPONSE FORMAT
```
Question: the user's input question
Thought: your internal reasoning
Action: MUST BE EXACTLY ONE of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

## BEGIN!
Question: {user_question}
Thought:{agent_scratchpad}
"""


class ReActAgentNode(Runnable):
    def __init__(self, llm, tools: list):
        prompt = PromptTemplate.from_template(REACT_TEMPLATE)
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            return_intermediate_steps=True,
            max_iterations=5,
            handle_parsing_errors=True,
            max_execution_time=60,
        )

    def invoke(self, state: RAGState, config=None) -> RAGState:
        result = self.executor.invoke({"user_question": state.query})
        # Keep what the tool returned so the judge can check grounding
        observations = [str(obs) for _, obs in result.get("intermediate_steps", [])]
        return state.model_copy(update={"response": result["output"], "rag_context": "\n\n".join(observations)})


class JudgeAnswerNode(Runnable):
    def __init__(self, llm):
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=JudgeOutput)

    def invoke(self, state: RAGState, config=None) -> RAGState:
        prompt = [
            SystemMessage(
                content=f"""
You are evaluating an assistant's answer for accuracy, completeness and grounding in the retrieved context.

[Question]
{state.query}

[Retrieved context]
{state.rag_context or "(nothing retrieved)"}

[Answer]
{state.response}

Respond ONLY with this JSON:
{{"score": <float between 0 and 10>, "remarks": "<brief explanation>"}}
"""
            )
        ]
        raw = self.llm.invoke(prompt)
        try:
            parsed = self.parser.parse(raw.content)
            update = {"score": parsed.score, "remarks": parsed.remarks}
        except Exception:
            logger.warning("Judge returned unparseable output: %r", raw.content)
            update = {"score": None, "remarks": "judge output could not be parsed"}
        logger.info("[judge] score=%s", update["score"])
        return state.model_copy(update=update)


class ChatbotGraphOrchestrator_ReActJudge:
    def __init__(self, retriever, react_llm, judge_llm, reranker_llm=None, k: int = 8):
        self.similarity_tool = Tool(
            name="similarity_search",
            func=SimilaritySearchTool(retriever, reranker_llm or react_llm, k=k),
            description="Search the knowledge base and return the relevant chunks for a query, most relevant first.",
        )
        self.react_llm = react_llm
        self.judge_llm = judge_llm
        self.executable = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("react_agent", ReActAgentNode(self.react_llm, [self.similarity_tool]))
        workflow.add_node("judge", JudgeAnswerNode(self.judge_llm))
        workflow.set_entry_point("react_agent")
        workflow.add_edge("react_agent", "judge")
        workflow.add_edge("judge", END)
        return workflow.compile()

    def run(self, question: str) -> dict:
        """Return the final state as a dict (response, score, remarks, rag_context)."""
        return self.executable.invoke(RAGState(query=question))
