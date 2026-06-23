import os
from dotenv import load_dotenv
load_dotenv(override=True)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '').strip().strip('"').strip("'")
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '').strip().strip('"').strip("'")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

print("OPENAI_API_KEY starts with:", OPENAI_API_KEY[:20] if OPENAI_API_KEY else "NOT SET")
print("PINECONE_API_KEY starts with:", PINECONE_API_KEY[:20] if PINECONE_API_KEY else "NOT SET")

print("\nTesting OpenAI connection...")
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o")
    result = llm.invoke("Say hello in one word.")
    print("OpenAI OK:", result.content)
except Exception as e:
    print("OpenAI ERROR:", str(e))

print("\nTesting Pinecone connection...")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    indexes = pc.list_indexes()
    print("Pinecone OK, indexes:", [i.name for i in indexes])
except Exception as e:
    print("Pinecone ERROR:", str(e))
