import os
import tempfile
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline

# Configure environment
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.7  # Relevance threshold for retrieval
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512
TOP_K = 3  # Number of chunks to retrieve

# Store for conversation history
conversation_history = {}
current_session_id = None
current_document_store = None
current_document_name = None
FILE_EXTENSIONS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
}

class DocumentAIBot:
    def __init__(self):
        self.setup_models()
        
    def setup_models(self):
        print("Setting up models...")
        # Set up embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Set up LLM model
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).to(DEVICE)
        
        # Create text generation pipeline
        self.text_generation_pipeline = pipeline(
            "text2text-generation",
            model=self.llm_model,
            tokenizer=self.tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            device=0 if DEVICE == "cuda" else -1
        )
        
        # Create HuggingFace pipeline for LangChain
        self.llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)
        
        # Text splitter for document chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        
        print("Models loaded successfully!")
        
    def process_document(self, file_path):
        """Process a document and create a vector store."""
        print(f"Processing document: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension not in FILE_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Select appropriate loader
        loader_class = FILE_EXTENSIONS[file_extension]
        loader = loader_class(file_path)
        
        # Load and split the document
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("No content extracted from the document")
        
        print(f"Document split into {len(chunks)} chunks")
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, self.embedding_model)
        return vector_store

    def setup_retrieval_chain(self, vector_store):
        """Set up the retrieval chain with the vector store."""
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": TOP_K,
                "score_threshold": THRESHOLD
            }
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        
        return chain
    
    def get_answer(self, question, session_id, vector_store, chat_history):
        """Get answer for a question using the retrieval chain."""
        if not question.strip():
            return "Please enter a question related to the document.", chat_history
        
        # Setup retrieval chain if needed
        retrieval_chain = self.setup_retrieval_chain(vector_store)
        
        # Format chat history for the model
        formatted_chat_history = [(q, a) for q, a in chat_history]
        
        # Get response from the chain
        response = retrieval_chain(
            {"question": question, "chat_history": formatted_chat_history}
        )
        
        answer = response["answer"]
        source_documents = response.get("source_documents", [])
        
        # Format answer with source information
        if source_documents:
            source_info = "\n\nSources:"
            seen_sources = set()
            
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown source")
                page = doc.metadata.get("page", "Unknown page")
                
                source_key = f"{source}-{page}"
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    if source == "Unknown source":
                        source_info += f"\n- Document chunk (page {page})"
                    else:
                        source_info += f"\n- {os.path.basename(source)} (page {page})"
            
            answer += source_info
        
        return answer, chat_history + [(question, answer)]

def generate_session_id():
    """Generate a unique session ID."""
    import uuid
    return str(uuid.uuid4())

def process_uploaded_document(file_path):
    """Process an uploaded document and set up the session."""
    global current_session_id, current_document_store, current_document_name, conversation_history
    
    try:
        if file_path is None:
            return None, "Please upload a document first."
        
        # In newer Gradio versions, the file input with type="filepath" returns the path directly
        # No need to save the file as it's already saved by Gradio
        
        # Extract filename for display
        filename = os.path.basename(file_path)
        
        # Create document AI bot if not already created
        if not hasattr(process_uploaded_document, "bot"):
            process_uploaded_document.bot = DocumentAIBot()
        
        # Process the document
        vector_store = process_uploaded_document.bot.process_document(file_path)
        
        # Create a new session
        session_id = generate_session_id()
        conversation_history[session_id] = []
        
        # Update global variables
        current_session_id = session_id
        current_document_store = vector_store
        current_document_name = filename
        
        return [], f"Document '{filename}' processed successfully. You can now ask questions about it."
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error processing document: {str(e)}"

def clear_conversation():
    """Clear the conversation history for the current session."""
    global conversation_history, current_session_id
    
    if current_session_id and current_session_id in conversation_history:
        conversation_history[current_session_id] = []
    
    return [], f"Conversation cleared. You can continue asking questions about '{current_document_name}'."

def answer_question(question, history):
    """Answer a question about the current document."""
    global current_session_id, current_document_store, conversation_history
    
    if not current_document_store:
        return "", history + [(question, "Please upload a document first.")]
    
    if not hasattr(process_uploaded_document, "bot"):
        return "", history + [(question, "Document AI bot not initialized. Please reload the page and try again.")]
    
    try:
        # Get current chat history
        chat_history = conversation_history.get(current_session_id, [])
        
        # Get answer
        answer, updated_history = process_uploaded_document.bot.get_answer(
            question, 
            current_session_id, 
            current_document_store, 
            chat_history
        )
        
        # Update conversation history
        conversation_history[current_session_id] = updated_history
        
        # Update the display history
        history = history + [(question, answer)]
        return "", history
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "", history + [(question, f"Error generating answer: {str(e)}")]

def build_interface():
    """Build and launch the Gradio interface."""
    # Define the Gradio blocks
    with gr.Blocks(title="Document AI Chatbot") as interface:
        gr.Markdown("# 📄 Document AI Chatbot")
        gr.Markdown("Upload a document (PDF, TXT, DOCX, PPTX) and ask questions about its content.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Document upload and processing section
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".txt", ".docx", ".pptx"],
                    type="filepath"  # This returns the file path directly
                )
                
                upload_button = gr.Button("Process Document", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                clear_button = gr.Button("Clear Conversation")
                
                gr.Markdown("### System Information")
                system_info = gr.Markdown(f"""
                - Embedding Model: {EMBEDDING_MODEL}
                - Language Model: {LLM_MODEL}
                - Running on: {DEVICE}
                - Chunk Size: {CHUNK_SIZE}
                - Relevance Threshold: {THRESHOLD}
                """)
            
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True,
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Ask a question about the document",
                        placeholder="What is the main topic of this document?",
                        lines=2,
                        show_label=True
                    )
                    
                    submit_button = gr.Button("Submit", variant="primary")
        
        # Set up event handlers
        upload_button.click(
            process_uploaded_document,
            inputs=[file_input],
            outputs=[chatbot, upload_status]
        )
        
        submit_button.click(
            answer_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot]
        )
        
        question_input.submit(
            answer_question,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot]
        )
        
        clear_button.click(
            clear_conversation,
            inputs=[],
            outputs=[chatbot, upload_status]
        )
    
    return interface

# Main execution
if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )