import json
import pathlib
import re
import sys

from llama_index.core import Settings, StorageContext, load_index_from_storage

import config


def norm(text):
    return re.sub(r'\s+', ' ', text.lower())


Settings.llm = None
Settings.embed_model = config.get_embedding()
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=f'{config.INDEX_ROOT}/{sys.argv[1]}'))
retriever = index.as_retriever(similarity_top_k=5)
qa_set = json.loads(pathlib.Path('qa_set.json').read_text(encoding='utf-8'))

in_scope, out_scope, ranks = [], [], []
for item in qa_set:
    nodes = retriever.retrieve(item['question'])
    top = nodes[0].score if nodes else 0.0
    if item['keywords'] is None:
        out_scope.append(top)
        continue
    in_scope.append(top)
    rank = next((r for r, n in enumerate(nodes, 1)
                 if any(norm(k) in norm(n.node.get_content()) for k in item['keywords'])), None)
    ranks.append(rank)

n = len(ranks)
for k in (1, 3, 5):
    hit = sum(1 for r in ranks if r and r <= k) / n
    print(f'Hit@{k}: {hit:.2%}')
print(f'MRR: {sum(1 / r for r in ranks if r) / n:.3f}')
print('Skor teratas rata-rata, dalam cakupan:', round(sum(in_scope) / len(in_scope), 3))
print('Skor teratas rata-rata, luar cakupan:', round(sum(out_scope) / max(len(out_scope), 1), 3))