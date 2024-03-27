import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file and configure Google API
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    # Assuming genai.configure is a method to set up the Google API configuration
    # Replace with actual method to configure Google Generative AI
    # genai.configure(api_key=google_api_key)
    pass
else:
    st.error("Google API key not found. Please check your environment variables.")
    st.stop()

# Function to extract text from a list of uploaded PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
    return text

# Function to split the extracted text into smaller chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from the text chunks using embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:  # Catch more specific exceptions as needed
        st.error(f"Error creating vector store: {e}")
        return False

# Function to initialize and return a conversational chain for processing and answering questions
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, 'answer is not available in the context', don't provide the wrong answer.\n\n
    Context:\n{context}\n\n
    Question:\n{question}\n\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Main function to configure the app and handle user interactions
def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with Your PDFs")
    
    # Sidebar for uploading PDF files and processing them
    with st.sidebar:
        st.title("Upload PDFs:")
        pdf_docs = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    success = get_vector_store(text_chunks)
                    if success:
                        st.success("PDFs processed successfully.")
                    else:
                        st.error("Failed to process PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

    # User interaction for asking questions
    user_question = st.text_input("Ask a question based on the uploaded PDFs:")
    if user_question:
        user_input(user_question)

# Function to handle user input and update conversation history
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        conversation_context = '\n'.join(["Q: " + q + "\nA: " + a for q, a in st.session_state.conversation_history])
        current_context = f"Q: {user_question}\nA: "
        full_context = conversation_context + current_context if conversation_context else current_context

        # Initialize the conversational chain
        chain = get_conversational_chain()

        # Assuming the chain can handle the context and question to generate a response
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

        # Update the conversation history
        st.session_state.conversation_history.append((user_question, response["output_text"]))

        # Display the updated conversation history
        display_history()
    except Exception as e:  # It's good practice to catch more specific exceptions
        st.error(f"An error occurred: {e}")

def display_history():
    """Display the conversation history."""
    for q, a in st.session_state.conversation_history:
        # Using st.write or any other Streamlit function to display the Q&A format
        st.markdown(f"**Q:** {q}\n**A:** {a}\n", unsafe_allow_html=True)

if __name__ == "__main__":
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    main()
