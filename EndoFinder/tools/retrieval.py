import json
import numpy as np
import torch
import faiss
import dataclasses
from sklearn import preprocessing

from pathlib import Path
import sys
# Make sure EndoFinder is in PYTHONPATH.
base_path = str(Path(__file__).resolve().parent.parent.parent)
if base_path not in sys.path:
    sys.path.append(base_path)

from EndoFinder.datasets.matching import retrieval_eval

def l2_normalize(embedding):
    return preprocessing.normalize(embedding, norm='l2', axis=1)

@dataclasses.dataclass
class Embeddings:
    ids: np.ndarray
    embeddings: np.ndarray

@dataclasses.dataclass
class GroundTruthMatch:
    query: str
    db: str

def get_ground_truth(query_idx, ref_idx):

    gt_pairs = []
    for i in range(len(query_idx)):
        gt_pairs.append(GroundTruthMatch(str(query_idx[i]), str(ref_idx[i])))

    return gt_pairs

def get_Embeddings(ids, embeddings):

    return Embeddings(ids=ids, embeddings=embeddings)

def evaluate(
    queries: Embeddings,
    refs: Embeddings,
    gt,
    mode,
):

    metrics = retrieval_eval(
        queries.ids,
        queries.embeddings,
        refs.ids,
        refs.embeddings,
        gt,
        mode,
    )

    return metrics

def evaluate_inference(query_embeddings, ref_embeddings, query_names, ref_names, mode='cosine'):
    query_embeddings, ref_embeddings = np.array(query_embeddings), np.array(ref_embeddings)

    # query_embeddings = l2_normalize(query_embeddings)
    # ref_embeddings = l2_normalize(ref_embeddings)

    real_labels = np.array([ref_names.index(query_name) for query_name in query_names]).astype(int) #真实标签

    gt = get_ground_truth(np.arange(0, len(query_names)), real_labels)

    metrics = evaluate(get_Embeddings(ids=np.arange(0, len(query_names)), embeddings=query_embeddings), 
                        get_Embeddings(ids=np.arange(0, len(query_names)), embeddings=ref_embeddings),
                        gt=gt,
                        mode=mode)
    return metrics
