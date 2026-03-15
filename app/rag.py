"""
rag.py - RAG Pipeline with per-session conversation memory and source citations
Written for: langchain==1.2.x / langchain-core==1.2.x / langchain-community==0.4.x
"""

import time
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

DATA_DIR        = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
FAISS_INDEX     = VECTORSTORE_DIR / "faiss_index"


def _make_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_documents() -> list[Document]:
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{DATA_DIR}'. Add medical PDFs there first."
        )
    documents = []
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(docs)
    print(f"  Loaded {len(documents)} pages from {len(pdf_files)} PDF(s).")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embed chunks in batches of 50 with 65s delay between batches.
    Required because free tier limit is 100 requests/minute.
    One-time operation — index is saved to disk after completion.
    """
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = _make_embeddings()
    batch_size = 50

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"  Embedding {len(chunks)} chunks in {total_batches} batches of {batch_size}...")
    print(f"  Estimated time: ~{total_batches * 65 // 60} minutes (free tier rate limit)")

    # First batch — creates the vectorstore
    first_batch = chunks[:batch_size]
    vs = FAISS.from_documents(first_batch, embeddings)
    print(f"  Batch 1/{total_batches} done ({len(first_batch)} chunks)")

    # Remaining batches — add to existing vectorstore
    for i in range(batch_size, len(chunks), batch_size):
        batch      = chunks[i:i + batch_size]
        batch_num  = i // batch_size + 1
        print(f"  Waiting 65s before batch {batch_num}/{total_batches}...")
        time.sleep(65)
        vs.add_documents(batch)
        print(f"  Batch {batch_num}/{total_batches} done ({len(batch)} chunks)")

    vs.save_local(str(FAISS_INDEX))
    print(f"  FAISS index saved to '{FAISS_INDEX}'.")
    return vs


def load_vectorstore() -> FAISS:
    vs = FAISS.load_local(
        str(FAISS_INDEX), _make_embeddings(),
        allow_dangerous_deserialization=True,
    )
    print("  Loaded existing FAISS index.")
    return vs


_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, "
     "rewrite the follow-up as a fully self-contained standalone question. "
     "Return ONLY the rewritten question — no preamble."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise medical assistant. "
     "Answer using ONLY the context provided below. "
     "If the answer is not in the context, say: "
     "'I don't have enough information in the provided documents.' "
     "Be concise and clinically accurate.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


class RAGPipeline:
    def __init__(self):
        self.vectorstore: Optional[FAISS]                  = None
        self.llm:         Optional[ChatGroq] = None
        self._histories:  dict[str, list]                  = {}

    def initialize(self):
        """Load FAISS index if it exists. Called at app startup."""
        print("[RAG] Initialising pipeline...")
        if FAISS_INDEX.exists():
            self.vectorstore = load_vectorstore()
        else:
            print("[RAG] No index found — POST to /ingest to build one.")

        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.3,
        )
        print("[RAG] Pipeline ready.")

    def ingest_documents(self) -> int:
        """Build FAISS index from PDFs in data/. Returns chunk count."""
        print("[RAG] Ingesting documents...")
        chunks = split_documents(load_documents())
        self.vectorstore = build_vectorstore(chunks)
        print(f"[RAG] Ingestion complete — {len(chunks)} chunks indexed.")
        return len(chunks)

    def query(self, question: str, session_id: str = "default") -> dict:
        """
        Answer a question using RAG + per-session conversation history.
        Returns: {"answer": str, "sources": [{"document": str, "page": int}]}
        """
        if self.llm is None:
            raise RuntimeError("Pipeline not initialised — call initialize() first.")
        if self.vectorstore is None:
            raise ValueError("No documents ingested yet. POST to /ingest first.")

        history   = self._histories.setdefault(session_id, [])
        retriever = self.vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # 1. Condense follow-up into standalone question
        if history:
            standalone_q = (_CONDENSE_PROMPT | self.llm | StrOutputParser()).invoke(
                {"chat_history": history, "question": question}
            )
        else:
            standalone_q = question

        # 2. Retrieve relevant chunks
        source_docs = retriever.invoke(standalone_q)
        context     = "\n\n".join(doc.page_content for doc in source_docs)

        # 3. Generate grounded answer
        answer = (_ANSWER_PROMPT | self.llm | StrOutputParser()).invoke(
            {"chat_history": history, "question": question, "context": context}
        )

        # 4. Update bounded session history (last 10 turns)
        history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        if len(history) > 20:
            self._histories[session_id] = history[-20:]

        # 5. Deduplicate citations
        seen, sources = set(), []
        for doc in source_docs:
            filename = doc.metadata.get("source", "unknown")
            page     = doc.metadata.get("page", 0) + 1
            if (filename, page) not in seen:
                seen.add((filename, page))
                sources.append({"document": filename, "page": page})

        return {"answer": answer, "sources": sources}

    def clear_memory(self, session_id: Optional[str] = None):
        """Clear history for one session (or all if session_id is None)."""
        if session_id:
            self._histories.pop(session_id, None)
        else:
            self._histories.clear()

    def rebuild_index(self):
        self.ingest_documents()


rag_pipeline = RAGPipeline()