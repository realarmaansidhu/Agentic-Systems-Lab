import langchain
from langchain_groq import ChatGroq as Groq
import os

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"GROQ_API_KEY: {GROQ_API_KEY}")