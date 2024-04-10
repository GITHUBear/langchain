from __future__ import annotations

import logging
import uuid
import json
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

# import numpy as np
from sqlalchemy import Column, String, Table, create_engine, insert, text
from sqlalchemy.types import UserDefinedType, Float, String
from sqlalchemy.dialects.mysql import JSON, LONGTEXT, VARCHAR

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore

_LANGCHAIN_OCEANBASE_DEFAULT_EMBEDDING_DIM = 1536
_LANGCHAIN_OCEANBASE_DEFAULT_COLLECTION_NAME = "langchain_document"
_LANGCHAIN_OCEANBASE_DEFAULT_IVFFLAT_CREATION_ROW_THRESHOLD = 10000
_LANGCHAIN_OCEANBASE_DEFAULT_RWLOCK_MAX_READER = 64

Base = declarative_base()

def from_db(value):
    return [float(v) for v in value[1:-1].split(',')]

def to_db(value, dim=None):
    if value is None:
        return value

    return '[' + ','.join([str(float(v)) for v in value]) + ']'

class Vector(UserDefinedType):
    cache_ok = True
    _string = String()

    def __init__(self, dim):
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        return "VECTOR(%d)" % self.dim

    def bind_processor(self, dialect):
        def process(value):
            return to_db(value, self.dim)
        return process

    def literal_processor(self, dialect):
        string_literal_processor = self._string._cached_literal_processor(dialect)

        def process(value):
            return string_literal_processor(to_db(value, self.dim))
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            return from_db(value)
        return process

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other):
            return self.op('<->', return_type=Float)(other)

        def max_inner_product(self, other):
            return self.op('<#>', return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op('<=>', return_type=Float)(other)

class OceanBaseCollectionStat:
    def __init__(self):
        self._lock = threading.Lock()
        self.maybe_collection_not_exist = True
        self.maybe_collection_index_not_exist = True
    
    def collection_exists(self):
        with self._lock:
            self.maybe_collection_not_exist = False
    
    def collection_index_exists(self):
        with self._lock:
            self.maybe_collection_index_not_exist = False

    def collection_not_exists(self):
        with self._lock:
            self.maybe_collection_not_exist = True
    
    def collection_index_not_exists(self):
        with self._lock:
            self.maybe_collection_index_not_exist = True
    
    def get_maybe_collection_not_exist(self):
        with self._lock:
            return self.maybe_collection_not_exist

    def get_maybe_collection_index_not_exist(self):
        with self._lock:
            return self.maybe_collection_index_not_exist

class OceanBaseGlobalRWLock:
    def __init__(self, max_readers) -> None:
        self.max_readers_ = max_readers
        self.writer_entered_ = False
        self.reader_cnt_ = 0            
        self.mutex_ = threading.Lock()
        self.writer_cv_ = threading.Condition(self.mutex_)
        self.reader_cv_ = threading.Condition(self.mutex_)

    def rlock(self):
        self.mutex_.acquire()
        while self.writer_entered_ or self.max_readers_ == self.reader_cnt_:
            self.reader_cv_.wait()
        self.reader_cnt_ += 1
        self.mutex_.release()

    def runlock(self):
        self.mutex_.acquire()
        self.reader_cnt_ -= 1
        if self.writer_entered_:
            if 0 == self.reader_cnt_:
                self.writer_cv_.notify(1)
        else:
            if self.max_readers_ - 1 == self.reader_cnt_:
                self.reader_cv_.notify(1)
        self.mutex_.release()

    def wlock(self):
        self.mutex_.acquire()
        while self.writer_entered_:
            self.reader_cv_.wait()
        self.writer_entered_ = True
        while 0 < self.reader_cnt_:
            self.writer_cv_.wait()
        self.mutex_.release()

    def wunlock(self):
        self.mutex_.acquire()
        self.writer_entered_ = False
        self.reader_cv_.notifyAll()
        self.mutex_.release()
    
    class OBRLock:
        def __init__(self, rwlock) -> None:
            self.rwlock_ = rwlock
            
        def __enter__(self):
            self.rwlock_.rlock()

        def __exit__(self, exc_type, exc_value, traceback):
            self.rwlock_.runlock()

    class OBWLock:
        def __init__(self, rwlock) -> None:
            self.rwlock_ = rwlock

        def __enter__(self):
            self.rwlock_.wlock()

        def __exit__(self, exc_type, exc_value, traceback):
            self.rwlock_.wunlock()

    def reader_lock(self):
        return self.OBRLock(self)
    
    def writer_lock(self):
        return self.OBWLock(self)

ob_grwlock = OceanBaseGlobalRWLock(_LANGCHAIN_OCEANBASE_DEFAULT_RWLOCK_MAX_READER)

class OceanBase(VectorStore):
    def __init__(
        self,
        connection_string: str,
        embedding_function: Embeddings,
        embedding_dimension: int = _LANGCHAIN_OCEANBASE_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_OCEANBASE_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        engine_args: Optional[dict] = None,
        delay_table_creation: bool = True,
        enable_index: bool = False,
        th_create_ivfflat_index: int = _LANGCHAIN_OCEANBASE_DEFAULT_IVFFLAT_CREATION_ROW_THRESHOLD,
        sql_logger: Optional[logging.Logger] = None,
        collection_stat: Optional[OceanBaseCollectionStat] = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.delay_table_creation = delay_table_creation
        self.th_create_ivfflat_index = th_create_ivfflat_index
        self.enable_index = enable_index
        self.sql_logger = sql_logger
        self.collection_stat = collection_stat
        self.__post_init__(engine_args)
    
    def __post_init__(
        self,
        engine_args: Optional[dict] = None,
    ) -> None:
        _engine_args = engine_args or {}
        if (
            "pool_recycle" not in _engine_args
        ):
            _engine_args[
                "pool_recycle"
            ] = 3600
        self.engine = create_engine(self.connection_string, **_engine_args)
        self.create_collection()
    
    @property
    def embeddings(self) -> Embeddings:
        return self.embedding_function

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._euclidean_relevance_score_fn
    
    def create_collection(self) -> None:
        if self.pre_delete_collection:
            self.delete_collection()
        if not self.delay_table_creation and (self.collection_stat is None or self.collection_stat.get_maybe_collection_not_exist()):
            self.create_table_if_not_exists()
            if self.collection_stat is not None:
                self.collection_stat.collection_exists()
    
    def delete_collection(self) -> None:
        drop_statement = text(f"DROP TABLE IF EXISTS {self.collection_name}")
        if self.sql_logger is not None:
            self.sql_logger.debug(f"Trying to delete collection: {drop_statement}")
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(drop_statement)
                if self.collection_stat is not None:
                    self.collection_stat.collection_not_exists()
                    self.collection_stat.collection_index_not_exists()
    
    def create_table_if_not_exists(self) -> None:
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS `{self.collection_name}` (
                id VARCHAR(40) NOT NULL, 
                embedding VECTOR({self.embedding_dimension}), 
                document LONGTEXT, 
                metadata JSON, 
                PRIMARY KEY (id)
            )
        """
        if self.sql_logger is not None:
            self.sql_logger.debug(f"Trying to create table: {create_table_query}")
        with self.engine.connect() as conn:
            with conn.begin():
                # Create the table
                conn.execute(text(create_table_query))
    
    def create_collection_ivfflat_index_if_not_exists(self) -> None:
        create_index_query = f"""
            CREATE INDEX IF NOT EXISTS `embedding_idx` on `{self.collection_name}` (
                embedding l2
            ) using ivfflat with (lists=20)
        """
        with ob_grwlock.writer_lock():
            with self.engine.connect() as conn:
                with conn.begin():
                    # Create Ivfflat Index
                    if self.sql_logger is not None:
                        self.sql_logger.debug(f"Trying to create ivfflat index: {create_index_query}")
                    conn.execute(text(create_index_query))

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        
        embeddings = self.embedding_function.embed_documents(list(texts))

        if len(embeddings) == 0:
            return ids

        if not metadatas:
            metadatas = [{} for _ in texts]
        
        if self.delay_table_creation and (self.collection_stat is None or self.collection_stat.get_maybe_collection_not_exist()):
            self.embedding_dimension = len(embeddings[0])
            self.create_table_if_not_exists()
            self.delay_table_creation = False
            if self.collection_stat is not None:
                self.collection_stat.collection_exists()

        chunks_table = Table(
            self.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", LONGTEXT, nullable=True),
            Column("metadata", JSON, nullable=True),  # filter
            keep_existing=True,
        )

        row_count_query = f"""
            SELECT COUNT(*) as count FROM `{self.collection_name}`
        """
        chunks_table_data = []
        # try:
        with self.engine.connect() as conn:
            with conn.begin():
                for document, metadata, chunk_id, embedding in zip(
                    texts, metadatas, ids, embeddings
                ):
                    chunks_table_data.append(
                        {
                            "id": chunk_id,
                            "embedding": embedding,
                            "document": document,
                            "metadata": metadata,
                        }
                    )

                    # Execute the batch insert when the batch size is reached
                    if len(chunks_table_data) == batch_size:
                        with ob_grwlock.reader_lock():
                            if self.sql_logger is not None:
                                insert_sql_for_log = str(insert(chunks_table).values(chunks_table_data))
                                self.sql_logger.debug(
                                    f"""Trying to insert vectors: 
                                        {insert_sql_for_log}""")
                            conn.execute(insert(chunks_table).values(chunks_table_data))
                        # Clear the chunks_table_data list for the next batch
                        chunks_table_data.clear()

                # Insert any remaining records that didn't make up a full batch
                if chunks_table_data:
                    with ob_grwlock.reader_lock():
                        if self.sql_logger is not None:
                            insert_sql_for_log = str(insert(chunks_table).values(chunks_table_data))
                            self.sql_logger.debug(
                                f"""Trying to insert vectors: 
                                    {insert_sql_for_log}""")
                        conn.execute(insert(chunks_table).values(chunks_table_data))
                
                # if self.sql_logger is not None:
                #     self.sql_logger.debug(f"Get the number of vectors: {row_count_query}")
                if self.enable_index and (self.collection_stat is None or self.collection_stat.get_maybe_collection_index_not_exist()):
                    with ob_grwlock.reader_lock():
                        row_cnt_res = conn.execute(text(row_count_query))
                    for row in row_cnt_res:
                        if row.count > self.th_create_ivfflat_index:
                            self.create_collection_ivfflat_index_if_not_exists()
                            if self.collection_stat is not None:
                                self.collection_stat.collection_index_exists()

        # except Exception as e:
        #     print(f"OceanBase add_text failed: {str(e)}")

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
        **kwargs: Any,
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        try:
            from sqlalchemy.engine import Row
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )
        
        # filter is not support in OceanBase.

        embedding_str = to_db(embedding, self.embedding_dimension)
        sql_query = f"""
            SELECT document, metadata, embedding <-> '{embedding_str}' as distance
            FROM {self.collection_name}
            ORDER BY embedding <-> '{embedding_str}'
            LIMIT :k
        """
        sql_query_str_for_log = f"""
            SELECT document, metadata, embedding <-> '?' as distance
            FROM {self.collection_name}
            ORDER BY embedding <-> '?'
            LIMIT {k}
        """

        params = {"k": k}
        try:
            with ob_grwlock.reader_lock():
                with self.engine.connect() as conn:
                    if self.sql_logger is not None:
                        self.sql_logger.debug(f"Trying to do similarity search: {sql_query_str_for_log}")
                    results: Sequence[Row] = conn.execute(text(sql_query), params).fetchall()
            
            documents_with_scores = [
                (
                    Document(
                        page_content=result.document,
                        metadata=json.loads(result.metadata),
                    ),
                    result.distance if self.embedding_function is not None else None,
                )
                for result in results
            ]
            return documents_with_scores
        except Exception as e:
            self.logger.error("similarity_search_with_score_by_vector failed:", str(e))
            return []
        
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding_function.embed_query(query)
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        return docs
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            raise ValueError("No ids provided to delete.")

        # Define the table schema
        chunks_table = Table(
            self.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", LONGTEXT, nullable=True),
            Column("metadata", JSON, nullable=True),  # filter
            keep_existing=True,
        )

        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    delete_condition = chunks_table.c.id.in_(ids)
                    delete_stmt = chunks_table.delete().where(delete_condition)
                    with ob_grwlock.reader_lock():
                        if self.sql_logger is not None:
                            self.sql_logger.debug(f"Trying to delete vectors: {str(delete_stmt)}")
                        conn.execute(delete_stmt)
                    return True
        except Exception as e:
            self.logger.error("Delete operation failed:", str(e))
            return False

    @classmethod
    def from_texts(
        cls: Type[OceanBase],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        embedding_dimension: int = _LANGCHAIN_OCEANBASE_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_OCEANBASE_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> OceanBase:
        connection_string = cls.get_connection_string(kwargs)

        store = cls(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding,
            embedding_dimension=embedding_dimension,
            pre_delete_collection=pre_delete_collection,
            engine_args=engine_args,
        )

        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store
    
    @classmethod
    def get_connection_string(cls, kwargs: Dict[str, Any]) -> str:
        connection_string: str = get_from_dict_or_env(
            data=kwargs,
            key="connection_string",
            env_key="OB_CONNECTION_STRING",
        )

        if not connection_string:
            raise ValueError(
                "OceanBase connection string is required"
                "Either pass it as a parameter"
                "or set the OB_CONNECTION_STRING environment variable."
            )

        return connection_string

    @classmethod
    def from_documents(
        cls: Type[OceanBase],
        documents: List[Document],
        embedding: Embeddings,
        embedding_dimension: int = _LANGCHAIN_OCEANBASE_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _LANGCHAIN_OCEANBASE_DEFAULT_COLLECTION_NAME,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> OceanBase:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        connection_string = cls.get_connection_string(kwargs)

        kwargs["connection_string"] = connection_string

        return cls.from_texts(
            texts=texts,
            pre_delete_collection=pre_delete_collection,
            embedding=embedding,
            embedding_dimension=embedding_dimension,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            engine_args=engine_args,
            **kwargs,
        )

    @classmethod
    def connection_string_from_db_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"