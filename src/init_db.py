# init_db.py

from src.datasource.db import engine, Base
from src.datasource.models.conversation import ConversationMemory  # importer tous les modèles

def init_db():
    print("🚀 Initialisation de la base de données...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables créées avec succès.")

if __name__ == "__main__":
    init_db()
