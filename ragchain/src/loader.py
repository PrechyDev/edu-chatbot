from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from pathlib import Path

def load_document(path: str):
    ext = Path(path).suffix.lower()
    
    try:
        if ext == ".pdf":
            return PyMuPDFLoader(path).load()
        elif ext in [".pptx", ".ppt"]:
            return UnstructuredPowerPointLoader(path).load()
        elif ext in [".docx", ".doc"]:
            return UnstructuredWordDocumentLoader(path).load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        print(f"❌ Error loading document {path}: {e}")
        return []

# docs = load_document('data/inno-list-of-schools.pdf')

# if not docs:
#     raise ValueError("No documents loaded. Please check the file path and format.")

# # Check the number of documents and total number of characters in docs
# num_docs = len(docs)
# total_chars = sum(len(doc.page_content) for doc in docs)
# print(f"Number of documents: {num_docs}")
# print(f"Total characters in docs: {total_chars}")