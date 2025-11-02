import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
import streamlit as st
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import BM25Retriever


# -------------------------------------------------------------------------
# ✅ Enhanced Conversational Chain (with memory + stuff chain)
# -------------------------------------------------------------------------
def get_conversational_chain_with_memory(vectorstore):
    """
    Create a conversational QA chain with:
    - Google Gemini model
    - ConversationBufferMemory to remember previous Q&A
    - 'stuff' chain for proper memory integration
    """
    # Define custom prompt for detailed answers with chat history
    prompt_template = """
    You are an expert assistant. Use the context below to answer the user's question.
    If the answer is not available, respond with:
    "Answer is not available in the context."
    
    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "chat_history", "question"]
    )

    # Initialize Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

    # Add memory to retain conversational context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

    # Create QA chain with stuff for proper memory integration
    chain = load_qa_chain(
        model,
        chain_type="stuff",  # Changed from map_reduce to stuff
        prompt=prompt,
        memory=memory
    )
    
    return chain, memory  # Return both chain and memory


# -------------------------------------------------------------------------
# ✅ Test Gemini connection (optional utility)
# -------------------------------------------------------------------------
def test_gemini_api():
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        response = llm.invoke("Explain the difference between supervised and unsupervised learning in simple terms.")
        print(response.content)
        return True
    except Exception as e:
        print(f"API Test Failed: {e}")
        return False


# -------------------------------------------------------------------------
# ✅ Improved Text Chunking (Semantic + Sentence aware)
# -------------------------------------------------------------------------
def text_chunk(text, chunk_size=600, chunk_overlap=200, add_start_index=True):
    """
    Split extracted PDF text into overlapping semantic chunks.
    Uses RecursiveCharacterTextSplitter for compatibility with LangChain.
    """
    if not text or len(text.strip()) == 0:
        raise ValueError("No text content found in PDFs")
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        raise ValueError("Text splitting resulted in no chunks")
    
    print(f"Created {len(chunks)} chunks")
    return chunks


# -------------------------------------------------------------------------
# ✅ Extract text using PyMuPDF (accurate + robust)
# -------------------------------------------------------------------------
def extract_text_from(pdf_docs):
    """
    Extract raw text from uploaded PDFs using PyMuPDF.
    Works with Streamlit's UploadedFile objects.
    """
    text = ""
    for pdf_file in pdf_docs:
        try:
            pdf_bytes = pdf_file.read()
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page in doc:
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {str(e)}")
            continue
    
    if not text or len(text.strip()) == 0:
        raise ValueError("No text could be extracted from the uploaded PDFs")
    
    return text


# -------------------------------------------------------------------------
# ✅ Hybrid Retriever: BM25 (keyword) + FAISS (semantic)
# -------------------------------------------------------------------------
def build_hybrid_retriever(text_chunks):
    """
    Build a hybrid retriever combining:
    - FAISS for semantic similarity
    - BM25 for keyword matching
    Ensures both contextually and literally relevant results.
    """
    if not text_chunks or len(text_chunks) == 0:
        raise ValueError("No text chunks provided for retriever")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    faiss_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Save with unique name to avoid conflicts
    faiss_store.save_local("faiss_index")

    # Hybrid combination
    bm25_retriever = BM25Retriever.from_texts(text_chunks)
    bm25_retriever.k = 5  # Set number of documents to retrieve
    
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]  # 30% keyword, 70% semantic
    )
    return hybrid_retriever, faiss_store


# -------------------------------------------------------------------------
# ✅ Retrieve & Answer User Query with Conversational Memory
# -------------------------------------------------------------------------
def user_input(user_question, retriever, qa_chain):
    """
    Process user question and return answer.
    Memory is handled internally by the qa_chain.
    """
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(user_question)
        
        if not docs:
            return "No relevant information found in the documents."
        
        # Get answer from chain (memory is handled internally)
        result = qa_chain({
            "input_documents": docs, 
            "question": user_question
        })
        
        return result["output_text"]
    
    except Exception as e:
        return f"Error processing question: {str(e)}"


# -------------------------------------------------------------------------
# ✅ Streamlit App
# -------------------------------------------------------------------------
def main():
    st.title("🔐 Gemini API Setup")

    # Initialize session state variable for API key
    if "api_set" not in st.session_state:
        st.session_state.api_set = False

    # If API not yet set, show input box and button
    if not st.session_state.api_set:
        st.subheader("Enter your Gemini API Key to continue:")
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here...")

        if st.button("Set API Key"):
            if api_key.strip() == "":
                st.warning("⚠️ Please enter a valid API key before proceeding.")
            else:
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
                st.session_state.api_set = True
                st.success("✅ API key set successfully! You can now proceed.")
                st.rerun()  # Refresh app so main content can load

        # Stop app execution until API key is set
        st.stop()

    # ===== MAIN APPLICATION BEGINS AFTER API KEY IS SET =====
    st.write("🎉 API Key successfully set! Your main app starts here...")

    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.header("Chat with Multiple PDF Files using Gemini 💁")

    # Sidebar for upload and actions
    st.sidebar.title("📄 PDF Upload Section")
    st.sidebar.markdown("---")

    pdf_docs = st.sidebar.file_uploader(
        "Upload your PDF Files",
        accept_multiple_files=True,
        type=['pdf']
    )

    # Initialize session state variables
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Process PDFs
    if st.sidebar.button("Process PDFs"):
        if pdf_docs:
            with st.spinner("Processing PDFs..."):
                try:
                    # Extract text
                    raw_text = extract_text_from(pdf_docs)
                    
                    # Create chunks
                    text_chunks = text_chunk(raw_text)
                    
                    # Build retriever
                    retriever, faiss_store = build_hybrid_retriever(text_chunks)
                    
                    # Create QA chain with memory
                    qa_chain, memory = get_conversational_chain_with_memory(faiss_store)
                    
                    # Store in session state
                    st.session_state.retriever = retriever
                    st.session_state.qa_chain = qa_chain
                    st.session_state.memory = memory
                    
                    st.sidebar.success("PDFs processed successfully! ✅")
                    st.success("You can now ask questions about your PDFs!")
                
                except Exception as e:
                    st.sidebar.error(f"Error processing PDFs: {str(e)}")
                    st.error("Failed to process PDFs. Please check the files and try again.")
        else:
            st.sidebar.warning("Please upload at least one PDF file.")

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Instructions:**\n1. Upload one or more PDF files\n2. Click 'Process PDFs'\n3. Ask questions below!")

    # Conversation UI
    st.markdown("### 💬 Ask Questions About Your PDFs")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user question
    if prompt := st.chat_input("Ask a question from the PDF files..."):
        if not st.session_state.retriever or not st.session_state.qa_chain:
            st.error("⚠️ Please upload and process PDF files first!")
        else:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(
                        prompt, 
                        st.session_state.retriever, 
                        st.session_state.qa_chain
                    )
                    st.markdown(response)

            # Add assistant message to chat
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        # Clear the memory object if it exists
        if st.session_state.memory:
            st.session_state.memory.clear()
        st.sidebar.success("Chat history cleared!")
        st.rerun()

    # Reset all (clear PDFs and chat)
    if st.sidebar.button("Reset All"):
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.session_state.retriever = None
        st.session_state.memory = None
        st.sidebar.success("All data cleared!")
        st.rerun()


if __name__ == "__main__":
    main()