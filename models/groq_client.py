import os
import asyncio
import logging
from groq import AsyncGroq
from dotenv import load_dotenv
from functools import wraps

load_dotenv()
logger = logging.getLogger(__name__)

# Decorator for exponential backoff to handle Rate Limits (429) - now async
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check if it's a rate limit error (HTTP 429)
                    if "429" in str(e) and x < retries:
                        sleep_time = (backoff_in_seconds * (2 ** x))
                        logger.warning(f"Rate limit hit. Retrying in {sleep_time}s...")
                        await asyncio.sleep(sleep_time)
                        x += 1
                    else:
                        raise e
        return wrapper
    return decorator

class GroqClient:
    def __init__(self):
        self._api_key = os.getenv("GROQ_API_KEY")
        self._client = None
        if not self._api_key:
            logger.warning("GROQ_API_KEY not found in environment. GroqClient calls will fail until it is set.")

    @property
    def client(self):
        if self._client is None:
            api_key = os.getenv("GROQ_API_KEY") or self._api_key
            if not api_key:
                logger.error("GROQ_API_KEY not found in environment or .env file")
                raise ValueError("Missing GROQ_API_KEY environment variable. Please set it in your .env file.")
            self._api_key = api_key
            self._client = AsyncGroq(api_key=api_key, timeout=30.0)
        return self._client

    @retry_with_backoff()
    async def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        """
        Secure async generation wrapper with rate limiting and timeout.
        """
        # Input sanitization: prevent extremely large prompts from crashing the API/App
        if len(prompt) > 10000:
            logger.warning("Prompt too long, truncating to 10k characters")
            prompt = prompt[:10000]

        try:
            completion = await self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

# Singleton instance
groq_client = GroqClient()
