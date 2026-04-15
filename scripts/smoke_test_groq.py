import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.groq_client import groq_client
from dotenv import load_dotenv
load_dotenv()

try:
    response = asyncio.run(groq_client.generate("Reply with exactly: API_OK"))
    print(response)
except Exception as e:
    print(f"Error: {e}")
