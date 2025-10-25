# Import necessary libraries
from PyPDF2 import PdfReader  # For reading PDF files
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
import os  # For interacting with the operating system
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For generating embeddings using Google Generative AI
import google.generativeai as genai  # For using Google's generative AI capabilities
from langchain_community.vectorstores import FAISS  # For efficient similarity search with embeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat-based interactions with Google AI
from langchain_classic.chains.question_answering import load_qa_chain  # For loading QA chains
from langchain_classic.prompts import PromptTemplate  # For creating templates for prompts
import streamlit as st

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = add your api key here  # Paste Your Gemini API Key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # Create a PdfReader object for the current PDF file
        for page in pdf_reader.pages:  # Iterate through each page in the PDF
            text += page.extract_text()  # Extract text from the page and append it to the 'text' variable
    return text  # Return the concatenated text from all PDFs

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    # Initialize the RecursiveCharacterTextSplitter with a chunk size of 10,000 characters
    # overlap of 1,000 characters between consecutive chunks to maintain context.
    chunks = text_splitter.split_text(text)  # Split the text into chunks
    return chunks  # Return the list of text chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")  # Create embeddings using the specified AI model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)  # Create a FAISS vector store from the text chunks
    vector_store.save_local("faiss_index")  # Save the vector store locally for future use

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)  # Initialize the AI model with specified parameters
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Create a prompt template
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)  # Load the QA chain with the model and prompt
    return chain  # Return the configured question-answering chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")  # Create embeddings for the search
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load the FAISS vector store
    docs = new_db.similarity_search(user_question)  # Retrieve the most relevant documents based on the user's question
    
    chain = get_conversational_chain()  # Get the configured conversational chain
    
    response = chain(
        {"input_documents": docs, "question": user_question},  # Pass retrieved docs and question to the chain
        return_only_outputs=True)  # Return only the output from the chain
    
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    
    st.header("Chat with Multiple PDF Files using Gemini 💁")
    
    # Create two main sections
    st.sidebar.title("📄 PDF Upload Section")
    st.sidebar.markdown("---")
    
    # PDF Upload Section
    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files", 
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if st.sidebar.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                # Extract text from uploaded PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Split text into chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                get_vector_store(text_chunks)
                
                st.sidebar.success("PDFs processed successfully! ✅")
                st.success("You can now ask questions about your PDFs!")
        else:
            st.sidebar.warning("Please upload at least one PDF file.")
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Instructions:**\n1. Upload one or more PDF files\n2. Click 'Process PDFs'\n3. Ask questions in the main chat area")
    
    # Conversation Section
    st.markdown("### 💬 Ask Questions About Your PDFs")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question from the PDF files..."):
        # Check if index exists
        if not os.path.exists("faiss_index"):
            st.error("⚠️ Please upload and process PDF files first!")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()