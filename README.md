# Chat with Multiple PDFs using Gemini AI

## 1. Introduction

This project is a web application built with Streamlit that allows you to chat with one or more PDF documents. It uses Google's Gemini models (`gemini-2.5-flash` and `text-embedding-004`) and the LangChain framework to create a Retrieval-Augmented Generation (RAG) pipeline.

<a href="https://chat-with-multiple-pdfs-using-gemini-4rq34pd9uzxjzptmiabmwz.streamlit.app/">Chat-with-multiple-PDFs-using-Gemini website</a>

Users can upload their PDF files, process them to create a searchable vector knowledge base, and then ask questions in natural language. The application will find the most relevant information from the documents and generate a detailed answer based *only* on the provided context.


[![Watch the video demo](https://github.com/user-attachments/assets/9b8d82e9-dd53-41c1-b380-df1dc158c5a0)](https://drive.google.com/file/d/13NgPCU6ZzaxyhY_yoNxWlofgiG46EUXj/view?usp=sharing)

## 2. Code Flow (Flowchart)

Here is the step-by-step workflow of the application:

1.  **UI Initialization:** The Streamlit app starts, displaying a header, a sidebar for file uploads, and a main chat area.
2.  **PDF Upload:** The user uploads one or more PDF files via the `st.sidebar.file_uploader`.
3.  **Processing:** The user clicks the "Process PDFs" button.
    * **Text Extraction:** `get_pdf_text()` reads all uploaded files, extracts text from each page using `PdfReader`, and concatenates it into a single large string.
    * **Text Splitting:** `get_text_chunks()` takes the raw text and uses `RecursiveCharacterTextSplitter` to break it into smaller, overlapping chunks (10,000 characters each, with 1,000 overlap).
    * **Embedding & Vector Store:** `get_vector_store()` is called.
        * It initializes `GoogleGenerativeAIEmbeddings` (using the `text-embedding-004` model).
        * It uses `FAISS.from_texts()` to convert all text chunks into numerical vectors (embeddings) and stores them in a FAISS vector store.
        * This vector store is saved locally as a folder named `faiss_index`.
    * A success message is shown to the user.
4.  **User Asks a Question:** The user types a question into the `st.chat_input` and presses Enter.
5.  **Handling Input:** The `user_input()` function is triggered.
    * **Load Vector Store:** The application loads the previously saved `faiss_index` from the local disk using `FAISS.load_local()`.
    * **Similarity Search:** It performs a `similarity_search()` on the vector store. This compares the user's question (also converted to an embedding) against all the document chunks and retrieves the most relevant ones.
    * **Get QA Chain:** The `get_conversational_chain()` function is called.
        * It defines a specific `prompt_template` that instructs the AI to answer *only* from the provided context.
        * It initializes the `ChatGoogleGenerativeAI` model (`gemini-2.5-flash`).
        * It loads a `load_qa_chain` (of type "stuff") with the model and the custom prompt.
    * **Generate Response:** The chain is executed, passing in the relevant documents (`input_documents`) retrieved from the search and the user's `question`.
    * **Return Answer:** The chain returns the generated `output_text`.
6.  **Display:** The application displays the user's question and the assistant's response in the chat window. The conversation is saved in `st.session_state.messages` to maintain history.

## 3. Technology Used

* **Python:** The core programming language.
* **Streamlit:** For building and running the interactive web user interface.
* **Google Gemini:** The family of generative AI models used.
    * `gemini-2.5-flash`: The chat model used for generating answers (QA).
    * `text-embedding-004`: The model used for creating vector embeddings from text chunks.
* **LangChain:** A framework for developing applications powered by language models. It's used to orchestrate the entire RAG pipeline, from text splitting to prompt management and running the QA chain.
* **FAISS (Facebook AI Similarity Search):** A library for efficient similarity search and clustering of dense vectors. It's used as the vector store to house the document embeddings.
* **PyPDF2:** A Python library used to read and extract text from PDF files.

## 4. Applications

This tool can be adapted for various use cases:

* **Personal Knowledge Management:** Chat with your personal collection of e-books, articles, and notes.
* **Academic Research:** Quickly find information and get answers from multiple research papers or textbooks.
* **Business & Legal:** Query large documents like contracts, technical manuals, or financial reports to find specific clauses or information.
* **Customer Support:** Can be the backend for a chatbot that answers questions based on a company's internal knowledge base (e.g., user guides, FAQs).

## 5. Requirements

You will need the following Python libraries. You can install them using `pip`.

```

streamlit
PyPDF2
langchain
langchain-google-genai
langchain-community
langchain-classic
google-generativeai
faiss-cpu
python-dotenv

````

Create a `requirements.txt` file with the content above and run:
`pip install -r requirements.txt`

## 6. Tools and Library Significance

* **streamlit:** Provides the interactive web UI (buttons, file uploader, chat interface) with minimal code.
* **PyPDF2:** The key to unlocking the data. It reads the `.pdf` format and extracts the raw text content.
* **langchain / langchain-classic:** The "glue" for the application. It connects all the different parts.
    * `RecursiveCharacterTextSplitter`: Intelligently splits large texts to fit into the model's context.
    * `load_qa_chain`: A pre-built chain that simplifies the process of sending context and a question to an LLM.
    * `PromptTemplate`: Allows for customizing the exact instructions given to the AI, ensuring it follows rules (like not answering from outside the context).
* **langchain-google-genai / google-generativeai:** Provides the "brain."
    * `GoogleGenerativeAIEmbeddings`: Converts human text into a mathematical representation (vector) that the computer can understand and compare.
    * `ChatGoogleGenerativeAI`: The actual LLM that reads the context and "thinks" of an answer.
* **faiss-cpu:** The "database." It's a highly optimized library for storing and searching through millions of vectors almost instantly. This is what makes the "retrieval" part of RAG so fast.

## 7. How to Download and Run Locally

Follow these steps to get the application running on your local machine.

1.  **Clone the Repository:**
    ```sh
    git clone [https://github.com/manojDOX/Chat-with-multiple-PDFs-using-Gemini](https://github.com/manojDOX/Chat-with-multiple-PDFs-using-Gemini)
    cd Chat-with-multiple-PDFs-using-Gemini
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the libraries listed in section 5, then run:
    ```sh
    pip install -r requirements.txt
    ```

4.  **Get Your Google API Key:**
    * Go to the [Google AI Studio](https://aistudio.google.com/app/apikey).
    * Create an API key.

5.  **Set the API Key:**
    Open the Python script (e.g., `app.py`) and replace the placeholder API key with your own:
    ```python
    # Find this line:
    os.environ["GOOGLE_API_KEY"] = "-----" 
    
    # Replace it with your key:
    os.environ["GOOGLE_API_KEY"] = "YOUR_ACTUAL_GOOGLE_API_KEY"
    ```
    *(**Note:** For better security, it's recommended to use a `.env` file and `python-dotenv` to load the key, but for this specific script, direct replacement works.)*

6.  **Run the Streamlit App:**
    In your terminal, run:
    ```sh
    streamlit run app.py
    ```
    (Assuming your Python file is named `app.py`)

7.  **Use the App:**
    Your default web browser will open automatically, pointing to the local Streamlit app. You can now upload your PDFs and start chatting!
