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
from langchain_core.documents import Document


# -------------------------------------------------------------------------
# ✅ Enhanced Conversational Chain (with memory + stuff chain)
# -------------------------------------------------------------------------
def get_conversational_chain_with_memory():
    """
    Create a conversational QA chain with:
    - Google Gemini model
    - ConversationBufferMemory to remember previous Q&A
    - 'stuff' chain for proper memory integration
    """
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

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt,
        memory=memory
    )
    
    return chain, memory


# -------------------------------------------------------------------------
# ✅ Test Gemini connection
# -------------------------------------------------------------------------
def test_gemini_api():
    """Test if Gemini API is working properly."""
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        response = llm.invoke("Test connection: respond with 'OK'")
        return True, "Connection successful"
    except Exception as e:
        return False, f"API Test Failed: {str(e)}"


# -------------------------------------------------------------------------
# ✅ Extract text using PyMuPDF with metadata
# -------------------------------------------------------------------------
def extract_text_from(pdf_docs):
    """
    Extract text from uploaded PDFs with metadata.
    Returns list of Document objects with source and page information.
    """
    documents = []
    for pdf_file in pdf_docs:
        try:
            pdf_bytes = pdf_file.read()
            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    page_text = page.get_text("text")
                    if page_text and page_text.strip():
                        # Create Document object with metadata
                        documents.append(Document(
                            page_content=page_text,
                            metadata={
                                'source': pdf_file.name,
                                'page': page_num + 1
                            }
                        ))
        except Exception as e:
            st.error(f"Error reading {pdf_file.name}: {str(e)}")
            continue
    
    if not documents:
        raise ValueError("No text could be extracted from the uploaded PDFs")
    
    return documents


# -------------------------------------------------------------------------
# ✅ Text Chunking with metadata preservation
# -------------------------------------------------------------------------
def text_chunk(documents, chunk_size=1000, chunk_overlap=400):
    """
    Split PDF pages into chunks while preserving metadata.
    Returns list of Document objects with source and page info.
    """
    if not documents or len(documents) == 0:
        raise ValueError("No documents found in PDFs")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents and preserve metadata
    all_chunks = []
    for doc in documents:
        # Split the document (returns Document objects with metadata preserved)
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    if not all_chunks:
        raise ValueError("Text splitting resulted in no chunks")
    
    # Get unique source PDFs
    unique_sources = set(chunk.metadata.get('source', 'Unknown') for chunk in all_chunks)
    st.info(f"✅ Created {len(all_chunks)} chunks from {len(documents)} pages across {len(unique_sources)} PDF(s)")
    
    return all_chunks


# -------------------------------------------------------------------------
# ✅ Hybrid Retriever with proper metadata
# -------------------------------------------------------------------------
def build_hybrid_retriever(document_chunks):
    """
    Build hybrid retriever with BM25 and FAISS.
    Handles Document objects with metadata properly.
    """
    if not document_chunks or len(document_chunks) == 0:
        raise ValueError("No document chunks provided for retriever")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    
    # Check if we need to rebuild index (optional: can be removed if you want persistence)
    # Commenting out auto-deletion for now - uncomment if you want fresh index each time
    # if os.path.exists("faiss_index"):
    #     import shutil
    #     try:
    #         shutil.rmtree("faiss_index")
    #         st.info("🗑️ Removed old FAISS index...")
    #     except Exception as e:
    #         st.warning(f"Could not remove old index: {e}")
    
    # Build FAISS from Document objects (metadata is automatically preserved)
    faiss_store = FAISS.from_documents(document_chunks, embeddings)
    faiss_store.save_local("faiss_index")
    
    unique_sources = set(doc.metadata.get('source', 'Unknown') for doc in document_chunks)
    st.success(f"✅ Created FAISS index for {len(unique_sources)} PDF(s)")
    
    # Build BM25 from Document objects (metadata is automatically preserved)
    bm25_retriever = BM25Retriever.from_documents(document_chunks)
    bm25_retriever.k = 5
    
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

    st.info("🔍 Hybrid Retriever: 30% BM25 (keyword) + 70% FAISS (semantic)")
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7]
    )
    
    return hybrid_retriever


# -------------------------------------------------------------------------
# ✅ User Input Processing with source tracking
# -------------------------------------------------------------------------
def user_input(user_question, retriever, qa_chain):
    """
    Process user question with enhanced source tracking.
    Shows which PDF and page each answer came from.
    """
    try:
        # Retrieve relevant documents
        docs = retriever.invoke(user_question)
        
        if not docs:
            return "No relevant information found in the documents.", []
        
        # Extract source information with deduplication
        sources = []
        seen_content = set()
        
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            source_file = doc.metadata.get('source', 'Unknown')
            page_num = doc.metadata.get('page', 'Unknown')
            preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            
            sources.append({
                'source_file': source_file,
                'page': page_num,
                'preview': preview
            })
        
        # Get answer from chain
        result = qa_chain({
            "input_documents": docs, 
            "question": user_question
        })
        
        return result["output_text"], sources
    
    except Exception as e:
        return f"Error processing question: {str(e)}", []


# -------------------------------------------------------------------------
# ✅ Streamlit App
# -------------------------------------------------------------------------
def main():
    # MUST be first Streamlit command
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    
    st.title("🔐 Gemini API Setup")

    # Initialize session state
    if "api_set" not in st.session_state:
        st.session_state.api_set = False

    # API Key Setup
    if not st.session_state.api_set:
        st.subheader("Enter your Gemini API Key to continue:")
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your API key here...")

        if st.button("Set API Key"):
            if api_key.strip() == "":
                st.warning("⚠️ Please enter a valid API key before proceeding.")
            else:
                os.environ["GOOGLE_API_KEY"] = api_key.strip()
                
                with st.spinner("Testing API connection..."):
                    success, message = test_gemini_api()
                    if success:
                        st.session_state.api_set = True
                        st.success(f"✅ {message}")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                        st.error("Please check your API key and try again.")

        st.stop()

    # Main Application
    st.header("Chat with Multiple PDF Files using Gemini 💁")

    # Sidebar
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
                    # Extract text as Document objects with metadata
                    documents = extract_text_from(pdf_docs)
                    
                    # Create chunks (returns Document objects)
                    document_chunks = text_chunk(documents)
                    
                    # Build retriever
                    retriever = build_hybrid_retriever(document_chunks)
                    
                    # Create QA chain
                    qa_chain, memory = get_conversational_chain_with_memory()
                    
                    # Restore existing memory if it exists
                    if st.session_state.memory:
                        qa_chain.memory = st.session_state.memory
                    else:
                        st.session_state.memory = memory
                    
                    # Store in session state
                    st.session_state.retriever = retriever
                    st.session_state.qa_chain = qa_chain
                    
                    st.sidebar.success("✅ PDFs processed successfully!")
                    st.success("🎉 You can now ask questions about your PDFs!")
                
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing PDFs: {str(e)}")
                    st.error("Failed to process PDFs. Please check the files and try again.")
        else:
            st.sidebar.warning("⚠️ Please upload at least one PDF file.")

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Instructions:**\n1. Upload one or more PDF files\n2. Click 'Process PDFs'\n3. Ask questions below!")

    # Conversation UI
    st.markdown("### 💬 Ask Questions About Your PDFs")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander("📎 View Sources"):
                    for idx, source in enumerate(message["sources"], 1):
                        st.caption(f"**{idx}. 📄 {source['source_file']} - Page {source['page']}**")
                        st.text(source['preview'])
                        if idx < len(message["sources"]):
                            st.markdown("---")

    # Handle user question
    if prompt := st.chat_input("Ask a question from the PDF files..."):
        if not st.session_state.retriever or not st.session_state.qa_chain:
            st.error("⚠️ Please upload and process PDF files first!")
        else:
            # Re-attach memory to chain
            if st.session_state.memory:
                st.session_state.qa_chain.memory = st.session_state.memory
            
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response with sources
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = user_input(
                        prompt, 
                        st.session_state.retriever, 
                        st.session_state.qa_chain
                    )
                    st.markdown(response)
                    
                    # Display source information
                    if sources:
                        with st.expander("📎 View Sources"):
                            for idx, source in enumerate(sources, 1):
                                st.caption(f"**{idx}. 📄 {source['source_file']} - Page {source['page']}**")
                                st.text(source['preview'])
                                if idx < len(sources):
                                    st.markdown("---")

            # Add assistant message to chat with sources
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })

    # Clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.memory:
            st.session_state.memory.clear()
        st.sidebar.success("✅ Chat history cleared!")
        st.rerun()

    # Reset all
    if st.sidebar.button("Reset All"):
        st.session_state.messages = []
        st.session_state.qa_chain = None
        st.session_state.retriever = None
        st.session_state.memory = None
        st.sidebar.success("✅ All data cleared!")
        st.rerun()


if __name__ == "__main__":
    main()