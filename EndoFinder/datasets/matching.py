from typing import Dict
import numpy as np
from .metrics import PredictedMatch, evaluate, Metrics, print_metrics
from sklearn.metrics.pairwise import cosine_similarity

def hamming_distance(x, y):
    """haimming distance"""
    return np.sum(x != y, axis=-1)

def match_and_prediction(query_embeddings, ref_embeddings, query_ids, ref_ids, mode='cosine'):

    if mode == 'cosine':
        sim = cosine_similarity(query_embeddings, ref_embeddings)
        max_argmin = np.argmax(sim, axis=1)

    elif mode == 'hamming':
        sim = np.zeros((query_embeddings.shape[0], ref_embeddings.shape[0]))
        for i in range(query_embeddings.shape[0]):
            sim[i, :] = 1-hamming_distance(query_embeddings[i], ref_embeddings)/query_embeddings.shape[1]
        max_argmin = np.argmax(sim, axis=1)
    else:
        raise ValueError("Mode should be 'cosine' or 'hamming'")

    predictions = []
    for i in range(query_embeddings.shape[0]):

        predictions.append(PredictedMatch(str(query_ids[i]), str(ref_ids[max_argmin[i]]), sim[i, max_argmin[i]]))
    
    return predictions
    
def retrieval_eval(
        query_ids,
        query_embeddings,
        ref_ids,
        ref_embeddings,
        gt,
        mode,
    ) -> Dict[str, float]:
        
        predictions = match_and_prediction(query_embeddings, ref_embeddings, query_ids, ref_ids, mode)

        results: Metrics = evaluate(gt, predictions)
        return {
            "uAP": results.average_precision,
            "accuracy-at-1": results.recall_at_rank1,
            "recall-at-p90": results.recall_at_p90 or 0.0,
        }