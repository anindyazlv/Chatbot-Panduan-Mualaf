import json
import pathlib

import pymupdf
import streamlit as st
import torch
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

st.set_page_config(page_title='Mualaf Guide Chatbot', layout='wide')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SYSTEM_PROMPT = (
    'You are an assistant that answers beginner questions about the basics of '
    'Islam based on the guidebook provided.'
)

QA_TEMPLATE = '''Answer the question ONLY based on the context below.
Do not add evidence, rulings, or opinions that are not in the context.
If the context contains steps, write them in order.
If the answer is not in the context, answer exactly: __FALLBACK__
Answer in the same language as the question, concisely and clearly.
Summarize in your own words in 3 to 5 sentences; do not copy the context verbatim.

Context:
---------------------
{context_str}
---------------------
Question: {query_str}
Answer:'''.replace('__FALLBACK__', config.FALLBACK)


# LLM: loaded only when the user sends a question, then cached.
@st.cache_resource(show_spinner='Loading language model...')
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)

    if DEVICE == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.float16,
            device_map='auto',
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

    model.eval()

    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=384,
        generate_kwargs={'do_sample': False, 'use_cache': True},
        system_prompt=SYSTEM_PROMPT,
        tokenizer=tokenizer,
        model=model,
    )

# Embedding model
@st.cache_resource(show_spinner='Loading embedding model...')
def load_embedding():
    return config.get_embedding()


Settings.embed_model = load_embedding()
Settings.chunk_size = config.CHUNK_SIZE
Settings.chunk_overlap = config.CHUNK_OVERLAP


# Index
def list_versions():
    root = pathlib.Path(config.INDEX_ROOT)
    names = [p.parent.name for p in root.glob('v*/manifest.json')]
    return sorted(names, key=lambda v: int(v[1:]))


@st.cache_resource(show_spinner='Loading index...')
def load_index(version):
    ctx = StorageContext.from_defaults(
        persist_dir=str(pathlib.Path(config.INDEX_ROOT) / version)
    )
    return load_index_from_storage(ctx)


# PDF page rendering
@st.cache_data(show_spinner=False)
def render_page(file_name, page_no):
    if not file_name or page_no is None:
        return None

    try:
        pdf_path = pathlib.Path(config.DATA_DIR) / file_name

        with pymupdf.open(pdf_path) as doc:
            page_index = int(page_no) - 1

            if page_index < 0 or page_index >= len(doc):
                return None

            return doc[page_index].get_pixmap(dpi=110).tobytes('png')

    except Exception as e:
        print(f'Failed to render PDF page: {e}')
        return None


# Retrieval and query engine
def cutoff_filter(cutoff):
    return SimilarityPostprocessor(similarity_cutoff=cutoff)


def build_engine(index, llm, top_k, cutoff):
    engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        node_postprocessors=[cutoff_filter(cutoff)],
        response_mode='compact',
    )
    engine.update_prompts({
        'response_synthesizer:text_qa_template': PromptTemplate(QA_TEMPLATE)
    })
    return engine


# Source display
def show_sources(nodes, expanded):
    with st.expander('Sources used', expanded=expanded):
        for i, node in enumerate(nodes, start=1):
            meta = node.node.metadata
            edition = meta.get('edition')
            page = meta.get('pdf_page')
            score = f'{node.score:.3f}' if node.score is not None else '-'

            st.markdown(
                f'**Source {i}: {edition}, PDF page {page}** (score {score})'
            )
            st.write(node.node.get_content())

            image = render_page(meta.get('file_name'), page)

            if image:
                st.image(
                    image,
                    caption='Original page (full Arabic text)',
                    width=480,
                )


# Sidebar
with st.sidebar:
    st.header('Settings')

    versions = list_versions()
    version = (
        st.selectbox('Index version', versions[::-1]) if versions else None
    )

    if version:
        path = pathlib.Path(config.INDEX_ROOT) / version / 'manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))

        st.caption(
            f"{manifest['pages_indexed']} pages indexed, "
            f"created {manifest['created_at']}"
        )

    use_llm = st.checkbox('Compose answer with language model', value=True)
    top_k = st.slider('Number of chunks (top-k)', 1, 8, config.TOP_K)
    cutoff = st.slider(
        'Similarity cutoff', 0.0, 1.0, config.SIM_CUTOFF, 0.05
    )


# Main UI
st.title('Mualaf Guide Chatbot')

st.warning(
    'This chatbot only summarizes the Guide Book for New Muslims. '
    'It is not a fatwa and does not replace your islamic teacher or mentor.'
)

index = load_index(version) if version else None

if index is None:
    st.info('No index yet. Run build_index.py first.')
    st.stop()


# Chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])


# User query
question = st.chat_input('Ask about the basics of Islam, e.g. how to perform wudhu...')

if question:
    st.session_state.messages.append({'role': 'user', 'content': question})

    with st.chat_message('user'):
        st.write(question)

    with st.chat_message('assistant'):
        nodes = []
        answer = config.FALLBACK

        if use_llm:
            try:
                with st.spinner('Loading model and composing answer...'):
                    llm = load_llm()
                    engine = build_engine(index, llm, top_k, cutoff)
                    response = engine.query(question)

                nodes = response.source_nodes
                answer = str(response).strip() if nodes else config.FALLBACK

            except Exception as e:
                st.error('Failed to load the model or process the question.')
                st.exception(e)
                st.warning(
                    'Try disabling the language model option in the sidebar '
                    'to check whether retrieval still works.'
                )

        else:
            try:
                with st.spinner('Searching the book...'):
                    found = index.as_retriever(
                        similarity_top_k=top_k
                    ).retrieve(question)
                    nodes = cutoff_filter(cutoff).postprocess_nodes(found)

                answer = (
                    'Here are the most relevant excerpts from the guidebook:'
                    if nodes
                    else config.FALLBACK
                )

            except Exception as e:
                st.error('Failed to search the index.')
                st.exception(e)

        st.write(answer)

        if nodes:
            show_sources(nodes, expanded=not use_llm)

    st.session_state.messages.append({'role': 'assistant', 'content': answer})
