import os
from dotenv import load_dotenv

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Configuration de l'API OpenAI pour LangChain
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Configuration Azure Blob Storage
AZURE_CONNECTION_STRING = os.getenv('AZURE_CONNECTION_STRING')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME')

# Configuration de LangChain
LANGCHAIN_MODEL = 'openai-gpt-3.5-turbo'  # Exemple de modèle LangChain
