from src.config.config import AppConfig
import openai

openai.api_key = AppConfig.OPENAI_API_KEY

def should_use_rag(question: str, threshold: int = 70) -> bool:
    """
    Utilise un LLM pour décider intelligemment si RAG est nécessaire.
    """
    try:
        prompt = (
            "Tu es un système intelligent qui décide si une question nécessite "
            "l'accès à des documents externes (RAG). "
            "Réponds uniquement par OUI ou NON.\n\n"
            f"Question: {question}"
        )

        response = openai.ChatCompletion.create(
            model=AppConfig.OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10
        )

        decision = response['choices'][0]['message']['content'].strip().lower()
        return decision.startswith("oui")

    except Exception as e:
        print(f"[ERREUR DÉCISION LLM] {str(e)}")
        # fallback à heuristique simple
        keywords = ["doc", "pdf", "archive", "rapport"]
        return any(k in question.lower() for k in keywords)
