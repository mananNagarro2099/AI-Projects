# Import necessary modules
from dotenv import load_dotenv  # To load environment variables from .env file
import streamlit as st  # For creating the web app interface
import os  # For environment variable access
import google.generativeai as genai  # Google Generative AI SDK configuration
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from PIL import Image  # For image processing (not used here but imported)
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For embedding creation using Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat model interface with Gemini
from langchain.chains.question_answering import load_qa_chain  # For loading question-answering chain
from langchain.prompts import PromptTemplate  # For creating prompt templates
from langchain_community.vectorstores import FAISS  # For FAISS vector store usage
from PyPDF2 import PdfReader  # For reading PDF files

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Read each PDF file
        for page in pdf_reader.pages:
            text += page.extract_text()  # Extract text from each page and append
    return text

# Function to split text into smaller chunks for embedding
def get_chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)  # Split text into chunks of 1000 with 100 overlap
    return chunks

# Function to generate embeddings and save them as FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Load embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create vector store from text chunks
    vector_store.save_local("faiss_index")  # Save vector store locally for later retrieval

# Function to create a conversation chain for QA using Gemini chat model
def get_conversation_chain():
    # Prompt template to instruct the model to answer only from context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)  # Load Gemini chat model with low temperature for deterministic outputs
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Create prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load QA chain with model and prompt
    return chain

# Function to process user input, perform similarity search, and generate answer
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Load embeddings model
    # Load FAISS index from local storage with deserialization allowed (ensure file is trusted)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)  # Perform similarity search to find relevant document chunks
    chain = get_conversation_chain()  # Get the conversation QA chain
    # Get response by passing input documents and user question
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])  # Display answer on Streamlit app

# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat PDF")  # Set page title
    st.header("Chat with PDF using Gemini💁")  # Display header on top

    user_question = st.text_input("Ask a Question from the PDF Files")  # Input box for user question

    if user_question:
        user_input(user_question)  # If user asked a question, process and display answer

    with st.sidebar:
        st.title("Menu:")  # Sidebar title
        # File uploader widget to upload multiple PDF files
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):  # If submit button is clicked
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
                text_chunks = get_chunk_text(raw_text)  # Split text into chunks
                get_vector_store(text_chunks)  # Generate embeddings and store as FAISS index
                st.success("Done")  # Show success message

# Run the main function when script is executed directly
if __name__ == "__main__":
    main()
