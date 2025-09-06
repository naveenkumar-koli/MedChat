# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
import re  # Added for response parsing

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.getenv('GEMINI_API_KEY')
DB_FAISS_PATH = "vectorstore/db_faiss"

# Initialize AI components
def initialize_ai():
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    return model, db

model, db = initialize_ai()

# Safety settings
safety_settings = {
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400

        # Search for relevant documents
        docs = db.similarity_search(question, k=5)
        
        # Create clean context without source references
        context = "\n\n".join([doc.page_content for doc in docs])

        # Enhanced prompt with suggested questions
        prompt = f"""
You are a senior medical consultant. Analyze the patient's query and context:

**PATIENT QUERY**: {question}
**MEDICAL CONTEXT**: {context}

**RESPONSE GUIDELINES**:
1. Perform differential diagnosis considering symptoms
2. Suggest relevant diagnostic tests with rationale
3. Identify red-flag symptoms requiring urgent care
4. Provide evidence-based treatment options
5. Include risk factors and prevention strategies
6. Add standard medical disclaimer
7. After your response, suggest exactly 3 follow-up questions the patient might ask

**OUTPUT FORMAT**:
[Your comprehensive response here]

**SUGGESTED FOLLOW-UP QUESTIONS**:
1. [Question 1]?
2. [Question 2]?
3. [Question 3]?
"""

        # Generate response
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1500
            }
        )

        # Parse response to separate answer and suggested questions
        full_response = response.text
        answer, suggested_questions = parse_response(full_response)

        return jsonify({
            "answer": answer,
            "suggested_questions": suggested_questions
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def parse_response(full_response):
    """Separate main answer from suggested questions"""
    # Try to find the suggested questions section
    questions_section = re.search(
        r"SUGGESTED FOLLOW-UP QUESTIONS:\s*(.+)", 
        full_response, 
        re.DOTALL | re.IGNORECASE
    )
    
    if questions_section:
        # Extract questions section
        questions_text = questions_section.group(1).strip()
        # Split into individual questions
        question_lines = [q.strip() for q in questions_text.split('\n') if q.strip()]
        # Take up to 3 questions
        suggested_questions = question_lines[:3]
        
        # Remove questions section from main answer
        answer = full_response[:questions_section.start()].strip()
    else:
        # Fallback if parsing fails
        answer = full_response
        suggested_questions = [
            "What are common symptoms of this condition?",
            "Are there any lifestyle changes I should make?",
            "When should I seek emergency care?"
        ]
    
    # Add disclaimer if missing
    if "consult your healthcare provider" not in answer.lower():
        answer += "\n\nDisclaimer: This information is not medical advice. Always consult your healthcare provider for personal medical concerns."
    
    return answer, suggested_questions

if __name__ == '__main__':
    app.run(debug=True, port=5000)