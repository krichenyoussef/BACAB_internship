from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from config.config import settings

engine = create_engine(settings.DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
