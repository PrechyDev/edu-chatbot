# Chunking

from typing import List, Optional
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

# Customize your chunk size here
EMBEDDING_MODEL_NAME = "thenlper/gte-large"
MAX_TOKENS = 512
OVERLAP = int(MAX_TOKENS / 10)  # 10% overlap

def split_documents(docs: List[LangchainDocument],
    chunk_size: int = MAX_TOKENS,
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """
    Token-aware document chunking using model tokenizer.
    Ensures each chunk stays within model’s max input size.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=OVERLAP,
        add_start_index=True,
        strip_whitespace=True,
        #separators=SEPARATORS,
       #is_separator_regex=True,
    )

    chunks = text_splitter.split_documents(docs)

    # Remove duplicates
    seen = set()
    unique_chunks = []
    for doc in chunks:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_chunks.append(doc)
    print(f"Total chunks: {len(unique_chunks)}")

    return unique_chunks
