from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient, connections

from src.ingest import looks_like_table_dump

# langchain-milvus 0.3.x reads MilvusClient._using as its ORM alias but never
# registers it in pymilvus.connections, so any code path that touches an
# *existing* collection (load_vector_store, similarity_search after reuse)
# blows up in the constructor before our post-hoc connect() can run. Patch
# MilvusClient.__init__ once so every client also registers its alias.
_PATCHED_FLAG = "_lc_alias_patched"
if not getattr(MilvusClient, _PATCHED_FLAG, False):
    _orig_mc_init = MilvusClient.__init__

    def _mc_init_with_alias(self, uri: str = "http://localhost:19530", **kwargs):
        _orig_mc_init(self, uri=uri, **kwargs)
        alias = self._using
        if alias not in [a for a, _ in connections.list_connections()]:
            connections.connect(alias=alias, uri=uri, **{
                k: v for k, v in kwargs.items()
                if k in ("user", "password", "token", "secure", "db_name")
            })

    MilvusClient.__init__ = _mc_init_with_alias
    setattr(MilvusClient, _PATCHED_FLAG, True)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", ".", " ", ""]
DEFAULT_MILVUS_URI = "http://localhost:19530"
DEFAULT_COLLECTION_NAME = "buffett_letters"
DEFAULT_EMBEDDING_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v1"


def split_documents(
    documents: List[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 0,
    drop_table_chunks: bool = True,
) -> List[Document]:
    """Split documents with sentence-priority separators for one-line letters.

    When `drop_table_chunks` is True (default), chunks classified as financial
    tables by `looks_like_table_dump` are removed. The number filtered is
    written into `_split_documents_last_filtered` for caller-side logging.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(documents)

    global _split_documents_last_filtered
    if not drop_table_chunks:
        _split_documents_last_filtered = 0
        return chunks

    kept = [c for c in chunks if not looks_like_table_dump(c.page_content)]
    _split_documents_last_filtered = len(chunks) - len(kept)
    return kept


_split_documents_last_filtered: int = 0


def collection_exists(uri: str, collection_name: str) -> bool:
    """Check whether a collection already exists on the Milvus server."""
    client = MilvusClient(uri=uri)
    try:
        return client.has_collection(collection_name)
    finally:
        client.close()


def _make_vector_store(
    uri: str, collection_name: str, embedding_model: str
) -> Milvus:
    """Build a Milvus wrapper. Alias registration is handled by the
    MilvusClient.__init__ patch at module top."""
    embeddings = NVIDIAEmbeddings(model=embedding_model, truncate="NONE")
    return Milvus(
        embedding_function=embeddings,
        collection_name=collection_name,
        connection_args={"uri": uri},
        auto_id=True,
    )


def build_vector_store(
    chunks: List[Document],
    uri: str = DEFAULT_MILVUS_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    drop_old: bool = True,
) -> Milvus:
    """Create (or recreate) a Milvus collection and ingest chunks."""
    if drop_old:
        client = MilvusClient(uri=uri)
        try:
            if client.has_collection(collection_name):
                client.drop_collection(collection_name)
        finally:
            client.close()

    vs = _make_vector_store(uri, collection_name, embedding_model)
    vs.add_documents(chunks)
    return vs


def load_vector_store(
    uri: str = DEFAULT_MILVUS_URI,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Milvus:
    """Attach to an existing Milvus collection without re-ingesting."""
    return _make_vector_store(uri, collection_name, embedding_model)
