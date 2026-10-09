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

st.set_page_config(page_title='Chatbot Panduan Mualaf', layout='wide')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SYSTEM_PROMPT = (
    'Anda adalah asisten yang menjawab pertanyaan pemula tentang dasar-dasar '
    'Islam berdasarkan buku panduan yang diberikan.'
)

QA_TEMPLATE = '''Jawab pertanyaan HANYA berdasarkan konteks di bawah.
Jangan menambah dalil, hukum, atau pendapat yang tidak ada pada konteks.
Jika konteks berisi langkah-langkah, tuliskan secara berurutan.
Jika jawaban tidak ada pada konteks, jawab persis: __FALLBACK__
Jawab dalam bahasa yang sama dengan pertanyaan, ringkas, dan jelas.

Konteks:
---------------------
{context_str}
---------------------
Pertanyaan: {query_str}
Jawaban:'''.replace('__FALLBACK__', config.FALLBACK)


# ============================================================
# MODEL LLM
# ============================================================

@st.cache_resource(show_spinner='Memuat model bahasa...')
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL)

    dtype = torch.float16 if DEVICE == 'cuda' else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL,
        dtype=dtype
    ).to(DEVICE)

    model.eval()

    return HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={
            'do_sample': False
        },
        system_prompt=SYSTEM_PROMPT,
        tokenizer=tokenizer,
        model=model,
    )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner='Memuat model embedding...')
def load_embedding():
    return config.get_embedding()


Settings.embed_model = load_embedding()
Settings.chunk_size = config.CHUNK_SIZE
Settings.chunk_overlap = config.CHUNK_OVERLAP


# ============================================================
# INDEX
# ============================================================

def list_versions():
    root = pathlib.Path(config.INDEX_ROOT)

    names = [
        p.parent.name
        for p in root.glob('v*/manifest.json')
    ]

    return sorted(
        names,
        key=lambda v: int(v[1:])
    )


@st.cache_resource(show_spinner='Memuat indeks...')
def load_index(version):
    ctx = StorageContext.from_defaults(
        persist_dir=str(
            pathlib.Path(config.INDEX_ROOT) / version
        )
    )

    return load_index_from_storage(ctx)


# ============================================================
# PDF PAGE RENDERING
# ============================================================

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

            return doc[page_index].get_pixmap(
                dpi=110
            ).tobytes("png")

    except Exception as e:
        print(f'Gagal merender halaman PDF: {e}')
        return None


# ============================================================
# RETRIEVAL
# ============================================================

def cutoff_filter(cutoff):
    return SimilarityPostprocessor(
        similarity_cutoff=cutoff
    )


# ============================================================
# QUERY ENGINE
# ============================================================

def build_engine(index, llm, top_k, cutoff):
    engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        node_postprocessors=[
            cutoff_filter(cutoff)
        ],
        response_mode='compact',
    )

    engine.update_prompts({
        'response_synthesizer:text_qa_template':
            PromptTemplate(QA_TEMPLATE)
    })

    return engine


# ============================================================
# SOURCE DISPLAY
# ============================================================

def show_sources(nodes, expanded):
    with st.expander(
        'Sumber yang digunakan',
        expanded=expanded
    ):
        for i, node in enumerate(nodes, start=1):
            meta = node.node.metadata

            edisi = meta.get('edisi')
            hal = meta.get('hal_pdf')

            score = (
                f'{node.score:.3f}'
                if node.score is not None
                else '-'
            )

            st.markdown(
                f'**Sumber {i}: {edisi}, '
                f'halaman berkas PDF {hal}** '
                f'(skor {score})'
            )

            st.write(
                node.node.get_content()
            )

            image = render_page(
                meta.get('file_name'),
                hal
            )

            if image:
                st.image(
                    image,
                    caption='Halaman asli (teks Arab utuh)',
                    width=480
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header('Pengaturan')

    versions = list_versions()

    version = (
        st.selectbox(
            'Versi indeks',
            versions[::-1]
        )
        if versions
        else None
    )

    if version:
        path = (
            pathlib.Path(config.INDEX_ROOT)
            / version
            / 'manifest.json'
        )

        manifest = json.loads(
            path.read_text(
                encoding='utf-8'
            )
        )

        n_pages = manifest['pages_indexed']
        created = manifest['created_at']

        st.caption(
            f'{n_pages} halaman terindeks, '
            f'dibuat {created}'
        )

    use_llm = st.checkbox(
        'Susun jawaban dengan model bahasa',
        value=True
    )

    top_k = st.slider(
        'Jumlah potongan (top-k)',
        1,
        8,
        config.TOP_K
    )

    cutoff = st.slider(
        'Ambang kemiripan',
        0.0,
        1.0,
        config.SIM_CUTOFF,
        0.05
    )


# ============================================================
# MAIN UI
# ============================================================

st.title('Chatbot Panduan Mualaf')

st.warning(
    'Chatbot ini hanya merangkum buku Panduan Ringkas untuk Mualaf. '
    'Bukan fatwa dan tidak menggantikan ustaz atau pendamping Anda.'
)


# ============================================================
# LOAD INDEX
# ============================================================

index = (
    load_index(version)
    if version
    else None
)

if index is None:
    st.info(
        'Belum ada indeks. '
        'Jalankan build_index.py terlebih dahulu.'
    )
    st.stop()


# ============================================================
# LOAD LLM AT STARTUP
# ============================================================

llm = None

if use_llm:
    try:
        llm = load_llm()

    except Exception as e:
        st.error(
            'Model bahasa gagal dimuat. '
            'Silakan periksa traceback di bawah.'
        )
        st.exception(e)

        st.warning(
            'Chatbot tetap dapat digunakan tanpa model bahasa '
            'dengan menonaktifkan opsi "Susun jawaban dengan model bahasa".'
        )

        use_llm = False


# ============================================================
# CHAT HISTORY
# ============================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])


# ============================================================
# USER QUERY
# ============================================================

question = st.chat_input(
    'Tanyakan dasar-dasar Islam, misalnya tata cara wudu...'
)


if question:

    st.session_state.messages.append({
        'role': 'user',
        'content': question
    })

    with st.chat_message('user'):
        st.write(question)

    with st.chat_message('assistant'):

        try:

            if use_llm and llm is not None:

                with st.spinner(
                    'Mencari di buku dan menyusun jawaban...'
                ):

                    engine = build_engine(
                        index,
                        llm,
                        top_k,
                        cutoff
                    )

                    response = engine.query(question)

                nodes = response.source_nodes

                answer = (
                    str(response).strip()
                    if nodes
                    else config.FALLBACK
                )

            else:

                with st.spinner(
                    'Mencari informasi di buku...'
                ):

                    found = index.as_retriever(
                        similarity_top_k=top_k
                    ).retrieve(question)

                    nodes = cutoff_filter(
                        cutoff
                    ).postprocess_nodes(found)

                answer = (
                    'Berikut kutipan paling relevan '
                    'dari buku panduan:'
                    if nodes
                    else config.FALLBACK
                )

            st.write(answer)

            if nodes:
                show_sources(
                    nodes,
                    expanded=not use_llm
                )

        except Exception as e:

            st.error(
                'Terjadi error saat memproses pertanyaan.'
            )

            st.exception(e)

            answer = config.FALLBACK

            st.write(answer)

    st.session_state.messages.append({
        'role': 'assistant',
        'content': answer
    })
