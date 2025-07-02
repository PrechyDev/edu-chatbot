from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from langchain_core.documents import Document
from tqdm import tqdm
from keybert import KeyBERT


# ✅ Initialize KeyBERT (uses miniLM by default)
keyword_model = KeyBERT("paraphrase-MiniLM-L6-v2")

def extract_keywords(text, top_n=8):
    try:
        keywords = keyword_model.extract_keywords(
            text,
            stop_words="english",
            top_n=top_n,
            use_maxsum=True,
            nr_candidates=10,
            diversity=0.7
        )
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"❌ Failed to extract keywords: {e}")
        return []

def extract_simple_metadata(doc_text: str,
                            uploaded_at=None,
                            source_type="unknown",
                            institution="unknown",
                            doc_type="general",
                            origin="user_upload"):
    metadata = {
        "source_type": source_type,                # e.g. JAMB, Handbook, Circular
        "institution": institution,                # e.g. OAU, WAEC, Federal MinEd
        "doc_type": doc_type,                      # e.g. brochure, handbook, memo, press_release
        "origin": origin,                          # e.g. user_upload, scraped, API
        "language": "en",                          # can be updated later
        "keywords": extract_keywords(doc_text, top_n=3),
        "uploaded_at": uploaded_at or datetime.now(timezone.utc).isoformat()
    }
    print(f"Metadata: {metadata}")
    return metadata

def process_docs_with_metadata(
    docs,
    uploaded_at=datetime.now(timezone.utc).isoformat(),
    source_type="brochure",
    institution="unknown",
    doc_type="general",
    origin="user_upload",
    max_workers=8  # You can adjust this based on your CPU
):
    def process_one(i_doc):
        i, doc = i_doc
        try:
            meta = extract_simple_metadata(
                doc.page_content,
                uploaded_at=uploaded_at,
                source_type=source_type,
                institution=institution,
                doc_type=doc_type,
                origin=origin
            )
            return i, Document(page_content=doc.page_content, metadata=meta)
        except Exception as e:
            print(f"❌ Failed to process doc chunk {i}: {e}")
            return i, None

    results = [None] * len(docs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, (i, doc)): i for i, doc in enumerate(docs)}
        for f in tqdm(as_completed(futures), total=len(futures), desc="📄 Processing documents (parallel)"):
            i, result = f.result()
            results[i] = result

    # Filter out any None results (failed chunks)
    final_docs = [doc for doc in results if doc is not None]
    return final_docs
