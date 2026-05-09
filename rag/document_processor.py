import io
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

def process_document(file_bytes: bytes, filename: str) -> list[Document]:
    """Parse a file from bytes into a list of Langchain Documents."""
    docs = []
    ext = filename.split('.')[-1].lower()
    
    try:
        if ext == 'pdf':
            reader = PdfReader(io.BytesIO(file_bytes))
            text = ""
            for i, page in enumerate(reader.pages):
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text() + "\n"
            docs.append(Document(page_content=text, metadata={"source": filename}))
            
        elif ext in ['csv', 'xlsx', 'xls']:
            if ext == 'csv':
                df = pd.read_csv(io.BytesIO(file_bytes))
            else:
                df = pd.read_excel(io.BytesIO(file_bytes))
            
            # Convert tabular data into a readable text format for the LLM
            text = f"Data from {filename}:\n"
            text += df.to_string(index=False)
            docs.append(Document(page_content=text, metadata={"source": filename}))
            
        elif ext in ('txt', 'md'):
            text = file_bytes.decode('utf-8', errors='ignore')
            docs.append(Document(page_content=text, metadata={"source": filename}))

        else:
            logger.warning("unsupported_file_type", filename=filename)
            return []
            
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        logger.info("document_processed_successfully", filename=filename, chunks=len(chunks))
        return chunks
        
    except Exception as e:
        logger.error("document_processing_error", error=str(e), filename=filename)
        return []
