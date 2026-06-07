""" Set up RAPTOR to be used for slicing down eval set for BM25 search.
Using the following notebook as guide: https://github.com/parthsarthi03/raptor/blob/master/demo.ipynb"""

import os
import re
import sys
from pathlib import Path

# Use the RAPTOR repo cloned in this project (not the unrelated PyPI "raptor" package).
_raptor_repo = Path(__file__).resolve().parent.parent / "raptor"
if _raptor_repo.is_dir():
    _raptor_repo_str = str(_raptor_repo)
    if _raptor_repo_str not in sys.path:
        sys.path.insert(0, _raptor_repo_str)

from raptor import BaseSummarizationModel, BaseEmbeddingModel, RetrievalAugmentationConfig, RetrievalAugmentation
from transformers import pipeline
from config import GPT4RecModelConfig
from sentence_transformers import SentenceTransformer
import torch

os.environ["OPENAI_API_KEY"] = "not_used"

ITEM_MARKER_RE = re.compile(r"\[ITEM\s+(\d+)\]")

def catalog_document_for_raptor(item_text_by_item_id):
    parts = []
    for iid in sorted(item_text_by_item_id.keys()):
        parts.append(f"[ITEM {iid}]\n{item_text_by_item_id[iid]}")
    return "\n\n".join(parts)


def node_to_item_id_from_tree(RA):
    mapping = {}
    if RA is None or RA.tree is None:
        return mapping
    for node in RA.tree.leaf_nodes.values():
        match = ITEM_MARKER_RE.search(node.text)
        if match:
            mapping[node.index] = int(match.group(1))
    return mapping


def raptor_leaf_item_ids(RA, query, node_to_item_id, max_items=50):
    if not node_to_item_id or RA.retriever is None:
        return []
    try:
        _, layer_info = RA.retrieve(
            query,
            start_layer=0,
            num_layers=1,
            collapse_tree=False,
            return_layer_information=True,
        )
    except Exception:
        return []

    item_ids = []
    seen = set()
    for entry in layer_info:
        node_idx = entry.get("node_index")
        iid = node_to_item_id.get(node_idx)
        if iid is None or iid in seen:
            continue
        seen.add(iid)
        item_ids.append(iid)
        if len(item_ids) >= max_items:
            break
    return item_ids


def summary_to_items_from_tree(RA):
    """Map each layer-1 summary node index to all item ids in its subtree.

    Walks each layer-1 node's children down to layer-0 leaves and extracts
    every ``[ITEM <id>]`` marker via ``.findall`` so multi-item chunked leaves
    are fully captured (RAPTOR's default ``split_text`` can place several
    items into one leaf).
    """
    mapping = {}
    if RA is None or RA.tree is None:
        return mapping

    tree = RA.tree
    cache = {}

    def items_under(node_idx):
        if node_idx in cache:
            return cache[node_idx]
        node = tree.all_nodes.get(node_idx)
        if node is None:
            cache[node_idx] = []
            return cache[node_idx]
        if not node.children:
            iids = [int(m) for m in ITEM_MARKER_RE.findall(node.text)]
            cache[node_idx] = iids
            return iids
        collected = []
        seen = set()
        for child_idx in node.children:
            for iid in items_under(child_idx):
                if iid in seen:
                    continue
                seen.add(iid)
                collected.append(iid)
        cache[node_idx] = collected
        return collected

    for node in tree.layer_to_nodes.get(1, []):
        mapping[node.index] = items_under(node.index)
    return mapping


def raptor_top_summary_items(RA, query, summary_to_items, top_summaries=15, max_items=None):
    """Retrieve the top-N layer-1 summaries by query similarity, union their items.

    Args:
        RA: built/loaded RetrievalAugmentation instance.
        query: free-text query (e.g. a beam-search candidate from GPT4Rec).
        summary_to_items: mapping produced by ``summary_to_items_from_tree``.
        top_summaries: how many layer-1 summary nodes to take per query.
        max_items: optional hard cap on the returned item count.

    Returns:
        Ordered list of item ids (dedup'd, summaries in order of similarity).
    """
    if not summary_to_items or RA is None or RA.retriever is None:
        return []
    try:
        _, layer_info = RA.retrieve(
            query,
            start_layer=1,
            num_layers=1,
            collapse_tree=False,
            return_layer_information=True,
        )
    except Exception:
        return []
    if not layer_info:
        return []

    collected = []
    seen = set()
    for entry in layer_info[:top_summaries]:
        node_idx = entry.get("node_index")
        if node_idx is None:
            continue
        for iid in summary_to_items.get(node_idx, []):
            if iid in seen:
                continue
            seen.add(iid)
            collected.append(iid)
            if max_items is not None and len(collected) >= max_items:
                return collected
    return collected


class SBertEmbeddingModel(BaseEmbeddingModel): # default embeddings used in demo notebook
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)


class GPT4RecSummarizationModel(BaseSummarizationModel):
    """Local summarizer for RAPTOR tree build. Uses DistilBART-CNN, which is
    instruction-free and actually compresses text. Independent from the
    DistilGPT-2 model GPT4Rec uses for query generation."""

    SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

    def __init__(self, config: GPT4RecModelConfig = None):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device} for RAPTOR summarization ({self.SUMMARIZER_MODEL})")
        self.summarization_pipeline = pipeline(
            "summarization",
            model=self.SUMMARIZER_MODEL,
            device=self.device,
        )

    def summarize(self, context, max_tokens=100):
        out = self.summarization_pipeline(
            context,
            max_length=int(max_tokens),
            min_length=max(8, int(max_tokens) // 4),
            do_sample=False,
            truncation=True,
        )
        return out[0]["summary_text"].strip()

def initialize_raptor(summarization_model,embedding_model,leaf_top_k=30):
    ra_config = RetrievalAugmentationConfig(
        summarization_model=summarization_model,
        embedding_model=embedding_model,
        tr_top_k=leaf_top_k,
    )
    RA = RetrievalAugmentation(config=ra_config)
    return RA

def build_raptor_tree(RA: RetrievalAugmentation, document: str):
    tree = RA.add_documents(document)

def save_raptor_tree(RA, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    RA.save(str(path))

def load_raptor_tree(path: str, embedding_model: SBertEmbeddingModel, leaf_top_k: int):
    config = RetrievalAugmentationConfig(
        embedding_model=embedding_model,
        tr_top_k=leaf_top_k,
    )
    RA = RetrievalAugmentation(config=config, tree=path) # load pickled tree from the path
    return RA
