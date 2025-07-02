from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.prompts import PromptTemplate

def build_rag_chain(llm, vector_store, rerank_fn):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are an intelligent assistant designed to help users answer questions about Nigerian education using trusted official documents, including brochures, handbooks, and press releases.
    
    You are provided with CONTEXT extracted from official educational documents. Each document may include metadata like:
    - Institution (e.g., OAU, UNILAG, JAMB)
    - Source type (e.g., Handbook, UTME Brochure)
    - Keywords (e.g., 'Accounting', 'Direct Entry', 'Computer Science')
    
    Use the provided context to answer the user's question as best as you can. 
    If the answer is not directly stated, but you can reasonably infer it from the context, provide your best possible answer and explain your reasoning briefly.
    If the answer truly cannot be found or inferred from the context, say: "I don't know based on the provided documents."
    
    Guidelines:
    - Be clear, factual, and concise.
    - If multiple institutions are mentioned, organize your answer accordingly.
    - When listing items (e.g., courses), format them as a clean list with bullets or commas.
    - If helpful, refer to document types or page numbers (from metadata) to strengthen the answer.
    
    Context:
    {context}
    
    User Question:
    {question}
    
    Answer:
    """
    )

    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str

    def refine_query(state: State):
        # Example: Use LLM to clarify/expand the question
        refine_prompt = PromptTemplate(
            input_variables=["question"],
            template="Rewrite the following question to be as clear and specific as possible for searching Nigerian education documents. Only return the improved question, nothing else:\n\n{question}\n\nRefined Question:"
        )
        refined = llm.invoke(refine_prompt.invoke({"question": state["question"]}))
        refined_text = refined.content.strip().splitlines()[0]  # Only take the first line
        print(f"[RAG] Refined query: {refined_text}")
        return {"question": refined_text}


    def retrieve(state: State):
        print(f"[RAG] Retrieving for query: {state['question']}")
        retrieved_docs = vector_store.similarity_search(state["question"], k=20)
        print(f"[RAG] Retrieved {len(retrieved_docs)} docs from Qdrant")
        for i, doc in enumerate(retrieved_docs[:3]):  # Show first 3 docs for brevity
            print(f"[RAG] Doc {i}: {doc.page_content[:200]}...")  # Print first 200 chars
        return {"context": retrieved_docs}

    def rerank(state: State):
        reranked = rerank_fn(state["question"], state["context"], top_k=10)
        print(f"[RAG] Reranked to {len(reranked)} docs")
        return {"context": reranked}

    def generate(state: State):
        context_text = "\n\n".join([doc.page_content for doc in state["context"]])
        print(f"[RAG] Context passed to LLM (first 500 chars): {context_text}...")
        messages = prompt.invoke({"question": state["question"], "context": context_text})
        response = llm.invoke(messages)
        print(f"[RAG] LLM response: {response.content[:200]}...")
        return {"answer": response.content}
 
    graph_builder = (
        StateGraph(State)
        .add_node("refine_query", refine_query)
        .add_node("retrieve", retrieve)
        .add_node("rerank", rerank)
        .add_node("generate", generate)
        .add_edge(START, "refine_query")
        .add_edge("refine_query", "retrieve")
        .add_edge("retrieve", "rerank")
        .add_edge("rerank", "generate")
    )

    return graph_builder.compile()

