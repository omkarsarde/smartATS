import os
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables for database configuration
DB_USER = os.getenv("DB_USER", "resuser")
DB_PASSWORD = os.getenv("DB_PASS", "respass")
DB_NAME = os.getenv("DB_NAME", "resumedb")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    contact = Column(String(100))
    skills = Column(Text)        # structured skills as comma-separated or JSON string
    years_of_experience = Column(Integer)
    education = Column(String(200))
    resume_path = Column(String(255))  # file path for the uploaded resume PDF