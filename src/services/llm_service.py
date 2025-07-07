# src/services/llm_service.py

import openai
from src.config.config import AppConfig

# Configure l'API OpenAI (via Azure ou OpenAI direct)
openai.api_key = AppConfig.OPENAI_API_KEY

# Si tu utilises Azure, décommente et configure ceci :
# openai.api_type = "azure"
# openai.api_base = AppConfig.OPENAI_API_BASE  # ex: https://mon-instance.openai.azure.com/
# openai.api_version = AppConfig.OPENAI_API_VERSION  # ex: 2023-05-15

def get_llm_response(question: str) -> str:
    """
    Génère une réponse à partir du LLM configuré (OpenAI ou Azure).
    """
    try:
        response = openai.ChatCompletion.create(
            model=AppConfig.OPENAI_MODEL_NAME,  # ex: gpt-4 ou deployment_name si Azure
            messages=[
                {"role": "system", "content": "Tu es un assistant utile."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )

        answer = response['choices'][0]['message']['content']
        return f"[LLM - {AppConfig.OPENAI_MODEL_NAME}] {answer}"

    except Exception as e:
        return f"[LLM - ERREUR] {str(e)}"
