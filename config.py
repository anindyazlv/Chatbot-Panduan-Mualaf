from llama_index.embeddings.huggingface import HuggingFaceEmbedding

LLM_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4
SIM_CUTOFF = 0.80
INDEX_ROOT = 'indexes'
DATA_DIR = 'data'
CLEANING_VERSION = 'c2'

PDF_NAME = 'panduan_mualaf.pdf'
EDISI = 'Content Association (1445 H)'
FALLBACK = ('Maaf, informasi tersebut tidak ditemukan pada buku panduan. '
            'Silakan tanyakan kepada ustaz atau pendamping mualaf Anda.')


def get_embedding():
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        query_instruction='query: ',
        text_instruction='passage: ',
    )
