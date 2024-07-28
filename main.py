import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_bcrypt import Bcrypt
from langchain_community.document_loaders import PyMuPDFLoader
from assesment.crew import AssesmentCrew

app = Flask(__name__)
bcrypt = Bcrypt(app)
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET'])
def index():
    return 'Hello, World!'

@app.route('/analysis', methods=['POST'])
def receive_data():
    if 'email' not in request.form:
        return jsonify({"error": "No email provided"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        upload_folder = os.path.join(BASE_DIR, 'TempFolder')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        loader = PyMuPDFLoader(filepath)
        pdf_text = loader.load()
        finaltext = ''
        for i in pdf_text:
            finaltext += i.page_content
        AssesmentCrew(raw_text=finaltext, email=request.form['email']).setup_crew().run()
        return jsonify({"pages": "successful"}), 200
    else:
        return jsonify({"error": "File is not a PDF"}), 400
    
def check_token(token):
    # Implement your token verification logic here
    # Return True if the token is valid, False otherwise
    # Example implementation:
    valid_token = os.environ.get('AUTH_TOKEN')
    return token == valid_token

@app.before_request
def authenticate():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return 'Unauthorized', 401

    token = auth_header.split(' ')[1]
    if not token:
        return 'Token is missing', 400

    if not check_token(token):
        return 'Invalid token', 401

def check_auth(password):
    stored_password = os.environ.get('UNHASHED_PASSWORD')
    return bcrypt.check_password_hash(password, stored_password)

def authenticate_error():
    return 'Authentication required', 401

if __name__ == '__main__':
    app.run()


# def run():
#     """
#     Run the crew.
#     """
#     inputs = {
#         'topic': 'AI LLMs'
#     }
#     AssesmentCrew().crew().kickoff(inputs=inputs)


# def train():
#     """
#     Train the crew for a given number of iterations.
#     """
#     inputs = {
#         "topic": "AI LLMs"
#     }
#     try:
#         AssesmentCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

#     except Exception as e:
#         raise Exception(f"An error occurred while training the crew: {e}")

# def replay():
#     """
#     Replay the crew execution from a specific task.
#     """
#     try:
#         AssesmentCrew().crew().replay(task_id=sys.argv[1])

#     except Exception as e:
#         raise Exception(f"An error occurred while replaying the crew: {e}")
