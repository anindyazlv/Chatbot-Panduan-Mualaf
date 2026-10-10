import datetime
import hashlib
import json
import pathlib
import re
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

        meta = {
            'file_name': path.name,
            'edition': config.EDITION,
            'pdf_page': page_no,
        }

        docs.append(
            Document(
                text=text,
                metadata=meta,
                excluded_embed_metadata_keys=list(meta),
                excluded_llm_metadata_keys=list(meta),
            )
        )

    return docs


def get_next_version():
    """
    Find the latest index version in INDEX_ROOT and return the next one.

    Examples:
    no index yet   -> v1
    v1 exists      -> v2
    v1, v2 exist   -> v3
    """
    index_root = pathlib.Path(config.INDEX_ROOT)
    index_root.mkdir(parents=True, exist_ok=True)

    versions = []

    for folder in index_root.iterdir():
        if not folder.is_dir():
            continue

        match = re.fullmatch(r'v(\d+)', folder.name)

        if match:
            versions.append(int(match.group(1)))

    next_number = max(versions) + 1 if versions else 1

    return f'v{next_number}'


def main():
    pdf = pathlib.Path(config.DATA_DIR) / config.PDF_NAME

    if not pdf.exists():
        sys.exit(f'File not found: {pdf}')

    # Determine version automatically
    version = get_next_version()
    print(f'Creating new index: {version}')

    # LlamaIndex settings (indexing does not need a language model)
    Settings.llm = None
    Settings.embed_model = config.get_embedding()
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP

    # Load PDF
    print('Reading PDF...')
    docs = load_pdf(pdf)
    print(f'{len(docs)} pages processed successfully.')

    if not docs:
        sys.exit('No pages were processed.')

    # Create vector index
    print('Creating vector index...')
    index = VectorStoreIndex.from_documents(docs, show_progress=True)

    # Create version folder and save index
    out = pathlib.Path(config.INDEX_ROOT) / version
    out.mkdir(parents=True, exist_ok=False)
    index.storage_context.persist(persist_dir=str(out))

    # Create manifest
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
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    print()
    print('Index created successfully.')
    print('Version  :', version)
    print('Location :', out)


if __name__ == '__main__':
    main()
