from llama_index.embeddings.huggingface import HuggingFaceEmbedding

LLM_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
EMBEDDING_MODEL = 'intfloat/multilingual-e5-base'
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4
SIM_CUTOFF = 0.820
INDEX_ROOT = 'indexes'
DATA_DIR = 'data'
CLEANING_VERSION = 'c3'

PDF_NAME = 'guide_book_for_new_muslims.pdf'
EDITION = 'Second Edition 2014'
FALLBACK = (
    'Sorry, that information was not found in the guidebook. '
    'Please ask your Islamic Teacher or Mentor.'
)


def get_embedding():
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        query_instruction='query: ',
        text_instruction='passage: ',
    )
