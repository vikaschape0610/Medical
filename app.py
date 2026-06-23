from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import traceback


app = Flask(__name__, template_folder="Template")


load_dotenv(override=True)

# Load and sanitize keys from .env
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '').strip().strip('"').strip("'")
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '').strip().strip('"').strip("'")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set in .env")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Ensure httpx/OpenAI clients have a valid CA bundle path before importing OpenAI clients
import certifi
ssl_cert = os.environ.get("SSL_CERT_FILE")
if not ssl_cert or not os.path.isfile(ssl_cert):
    os.environ["SSL_CERT_FILE"] = certifi.where()

embeddings = download_hugging_face_embeddings()

# Import Gemini after env vars are set
from langchain_google_genai import ChatGoogleGenerativeAI


index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)



retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})


chatModel = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Please type a question."
    try:
        print("User input:", msg)
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I could not generate a response.")
        print("Response:", answer)
        return str(answer)
    except Exception as e:
        print("Error:", str(e))
        traceback.print_exc()
        return "An error occurred while processing your request. Please try again."


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)