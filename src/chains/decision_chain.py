from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.llms.openai_llm import get_llm

def get_decision_chain():
    prompt = PromptTemplate.from_template(
        "Cette question nécessite-t-elle des documents pour y répondre ? Réponds OUI ou NON.\n\nQuestion : {question}"
    )
    return LLMChain(llm=get_llm(), prompt=prompt)

def should_use_rag(question: str) -> bool:
    decision = get_decision_chain().run({"question": question}).strip().lower()
    return decision.startswith("oui")
