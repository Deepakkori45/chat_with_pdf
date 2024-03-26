import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Replace 'YOUR_API_KEY' with your actual API key
# api_key = 'AIzaSyDMLlQUDRw0WV8iXWhVtMQkfXRFLf92aMo'
# genai.configure(api_key=api_key)




def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.4)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Prepare historical context along with the current question for the chain
    conversation_context = '\n'.join(["Q: " + q + "\nA: " + a for q, a in st.session_state.conversation_history])
    current_context = f"Q: {user_question}\nA: "

    # Combine historical context with the current question context
    full_context = conversation_context + current_context if conversation_context else current_context

    chain = get_conversational_chain()
    
    # Adjust this call as necessary based on your model's expected input format for including context/history
    response = chain({"input_documents": docs, "question": full_context}, return_only_outputs=True)
    
    # Update conversation history in the session state
    st.session_state.conversation_history.append((user_question, response["output_text"]))

    st.write("Reply: ", response["output_text"])

def display_history():
    for i, (q, a) in enumerate(st.session_state.conversation_history, 1):
        st.text(f"Q{i}: {q}\nA{i}: {a}\n")

# Call display_history in your main function where appropriate


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Model💁")
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)
        display_history()  # Display conversation history
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
