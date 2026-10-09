import json
import pathlib
import re
import sys

from llama_index.core import Settings, StorageContext, load_index_from_storage
import config


def norm(text):
    return re.sub(r'\s+', ' ', text.lower())

if len(sys.argv) < 2:
    sys.exit(
        'Usage: python eval_retrieval.py <version>\n'
        'Contoh: python eval_retrieval.py v1'
    )

version = sys.argv[1]
index_path = pathlib.Path(config.INDEX_ROOT) / version

if not index_path.exists():
    sys.exit(f'Indeks tidak ditemukan: {index_path}')

Settings.llm = None
Settings.embed_model = config.get_embedding()

index = load_index_from_storage(
    StorageContext.from_defaults(
        persist_dir=str(index_path)
    )
)

retriever = index.as_retriever(similarity_top_k=5)

qa_path = pathlib.Path('qa_set.json')

if not qa_path.exists():
    sys.exit(f'File evaluasi tidak ditemukan: {qa_path}')

qa_set = json.loads(
    qa_path.read_text(encoding='utf-8')
)

in_scope = []
out_scope = []
ranks = []

for item in qa_set:
    question = item['question']
    keywords = item.get('keywords')

    nodes = retriever.retrieve(question)

    top = nodes[0].score if nodes else 0.0

    # Out-of-scope question
    if keywords is None:
        out_scope.append(top)
        continue

    # In-scope question
    in_scope.append(top)

    rank = next(
        (
            r
            for r, node in enumerate(nodes, 1)
            if any(
                norm(keyword) in norm(node.node.get_content())
                for keyword in keywords
            )
        ),
        None
    )

    ranks.append(rank)

n = len(ranks)

if n == 0:
    print('Tidak ada pertanyaan in-scope yang dapat dievaluasi.')
    sys.exit(0)

for k in (1, 3, 5):
    hit = sum(
        1
        for rank in ranks
        if rank is not None and rank <= k
    ) / n

    print(f'Hit@{k}: {hit:.2%}')

mrr = sum(
    1 / rank
    for rank in ranks
    if rank is not None
) / n

print(f'MRR: {mrr:.3f}')
print('Skor teratas rata-rata, dalam cakupan:',
    round(sum(in_scope) / len(in_scope), 3)
    if in_scope else 'N/A')
print('Skor teratas rata-rata, luar cakupan:',
    round(sum(out_scope) / len(out_scope), 3)
    if out_scope else 'N/A')
