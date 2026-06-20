import groq, os
from dotenv import load_dotenv
load_dotenv()

print("--- DEBUG: Running fixed smoke_test.py ---")

client = groq.Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com"
)

try:
    print(f"DEBUG: Groq client type: {type(client)}")
    print(f"DEBUG: Has 'chat' attribute: {hasattr(client, 'chat')}")
    print(f"DEBUG: Has 'messages' attribute: {hasattr(client, 'messages')}")
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=50,
        messages=[{"role": "user", "content": "Reply with exactly: API_OK"}]
    )
    print(f"RESULT: {completion.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
