from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ".", " ", ""]


def split_documents(
    documents: List[Document],
    chunk_size: int = 3500,
    chunk_overlap: int = 500,
) -> List[Document]:
    """Split documents with sentence-priority separators for one-line letters."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


def build_vector_store(
    chunks: List[Document],
    persist_directory: str | Path,
    embedding_model: str = "gemini-embedding-001",
) -> Chroma:
    """Create and persist Chroma vector store with Gemini embeddings."""
    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    return vector_store


def load_vector_store(
    persist_directory: str | Path,
    embedding_model: str = "gemini-embedding-001",
) -> Chroma:
    """Load an existing Chroma vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    return Chroma(
        persist_directory=str(Path(persist_directory)),
        embedding_function=embeddings,
    )
