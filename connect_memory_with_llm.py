import google.generativeai as genai
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Configuration
API_KEY = os.getenv('GEMINI_API_KEY')
DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Safety settings to allow medical discussions
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
}

def ask_gemini(question):
    try:
        # Search for more relevant documents (increased from 3 to 5)
        docs = db.similarity_search(question, k=5)
        context = "\n\n".join([f"Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Enhanced prompt with better instructions
        response = model.generate_content(
            f"""You are a medical expert analyzing clinical documents. Follow these rules:
            
            CONTEXT FROM DOCUMENTS:
            {context}
            
            QUESTION: {question}
            
            INSTRUCTIONS:
            1. Answer in 2-3 paragraphs maximum
            2. Start with the most important information
            3. Include specific details from the sources when available
            4. Mention if different sources contradict each other
            5. Always add: "Consult your doctor for personalized advice"
            6. Never make definitive diagnostic claims
            
            ANSWER:""",
            safety_settings=safety_settings
        )
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Example usage
while True:
    question = input("\nEnter your medical question (or 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    print("\nProcessing your question...")
    print(ask_gemini(question))