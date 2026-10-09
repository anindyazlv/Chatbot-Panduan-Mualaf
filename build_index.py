import sys

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader

import config
from preprocess import clean_page


def load_pdf(path):
    reader = PyMuPDFReader()
    docs = []
    for page_no, page in enumerate(reader.load(file_path=str(path)), start=1):
        text = clean_page(page.text)
        if not text:
            continue
        meta = {'file_name': path.name, 'edisi': config.EDISI, 'hal_pdf': page_no}
        docs.append(Document(
            text=text, metadata=meta,
            excluded_embed_metadata_keys=list(meta),
            excluded_llm_metadata_keys=list(meta)))
    return docs


def main(version):
    pdf = pathlib.Path(config.DATA_DIR) / config.PDF_NAME
    if not pdf.exists():
        sys.exit(f'Berkas tidak ditemukan: {pdf}')

    Settings.llm = None  # pengindeksan tidak memerlukan model bahasa
    Settings.embed_model = config.get_embedding()
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    docs = load_pdf(pdf)
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    out = pathlib.Path(config.INDEX_ROOT) / version
    index.storage_context.persist(persist_dir=str(out))
    manifest = {
        'version': version,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'embedding_model': config.EMBEDDING_MODEL,
        'chunk_size': config.CHUNK_SIZE,
        'chunk_overlap': config.CHUNK_OVERLAP,
        'cleaning_version': config.CLEANING_VERSION,
        'pages_indexed': len(docs),
        'pdf': {pdf.name: hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]},
    }
    (out / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Indeks tersimpan di', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'v1')
