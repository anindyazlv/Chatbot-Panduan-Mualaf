import json
import pathlib
import re
import sys

from llama_index.core import Settings, StorageContext, load_index_from_storage

import config


def norm(text):
    return re.sub(r'\s+', ' ', text.lower())


# Arguments
if len(sys.argv) < 2:
    sys.exit(
        'Usage: python eval_retrieval.py <version>\n'
        'Example: python eval_retrieval.py v1'
    )

version = sys.argv[1]
index_path = pathlib.Path(config.INDEX_ROOT) / version

if not index_path.exists():
    sys.exit(f'Index not found: {index_path}')

# Load embedding and index
Settings.llm = None
Settings.embed_model = config.get_embedding()

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir=str(index_path))
)
retriever = index.as_retriever(similarity_top_k=config.TOP_K)

# Load evaluation dataset
qa_path = pathlib.Path('qa_set.json')

if not qa_path.exists():
    sys.exit(f'Evaluation file not found: {qa_path}')

qa_set = json.loads(qa_path.read_text(encoding='utf-8'))

# Retrieval evaluation
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
        None,
    )
    ranks.append(rank)

# Metrics
n = len(ranks)

if n == 0:
    print('No in-scope questions available to evaluate.')
    sys.exit(0)

for k in (1, 3, config.TOP_K):
    hit = sum(1 for rank in ranks if rank is not None and rank <= k) / n
    print(f'Hit@{k}: {hit:.2%}')

mrr = sum(1 / rank for rank in ranks if rank is not None) / n
print(f'MRR: {mrr:.3f}')

print(
    'Average top score, in-scope:',
    round(sum(in_scope) / len(in_scope), 3) if in_scope else 'N/A',
)
print(
    'Average top score, out-of-scope:',
    round(sum(out_scope) / len(out_scope), 3) if out_scope else 'N/A',
)

# Similarity cutoff check
print(
    f'In-scope passing cutoff ({config.SIM_CUTOFF}):',
    f'{sum(s >= config.SIM_CUTOFF for s in in_scope) / len(in_scope):.2%}',
)
print(
    f'Out-of-scope rejected by cutoff ({config.SIM_CUTOFF}):',
    f'{sum(s < config.SIM_CUTOFF for s in out_scope) / len(out_scope):.2%}'
    if out_scope else 'N/A',
)

# Cutoff sweep
for c in (0.82, 0.825, 0.83, 0.835, 0.84):
    kept = sum(s >= c for s in in_scope) / len(in_scope)
    rejected = sum(s < c for s in out_scope) / len(out_scope) if out_scope else 0
    print(f'cutoff {c:.3f}: in-scope kept {kept:.0%}, out-of-scope rejected {rejected:.0%}')

print('Max out-of-scope score:', round(max(out_scope), 3))
print('Min in-scope score:', round(min(in_scope), 3))
