
from typing import Optional

import k2
import torch


def dpwfst(
    features: torch.Tensor,
    codebook: torch.Tensor,
    lmbda: float,
    num_neighbors: Optional[int] = None,
):
    assert features.device == codebook.device
    device = features.device

    if num_neighbors is None:
        num_neighbors = codebook.shape[0]

    distances = torch.cdist(features, codebook, p=2.0) ** 2
    top_k_distances, top_k_indices = torch.topk(
        distances, k=num_neighbors, dim=1, largest=False
    )

    arcs1, labels1, scores1 = build_initial_transition_arcs(
        top_k_distances, top_k_indices
    )
    arcs2, labels2, scores2 = build_intermediate_transition_arcs(
        top_k_distances, top_k_indices, lmbda
    )
    arcs3, labels3, scores3 = build_final_transition_arcs(
        T=distances.shape[0], k=num_neighbors, device=device
    )

    arcs = torch.cat([arcs1, arcs2, arcs3], dim=0)
    labels = torch.cat([labels1, labels2, labels3], dim=0)
    scores = torch.cat([scores1, scores2, scores3], dim=0)

    # Create FSA
    fsa = k2.Fsa(arcs)

    # Now add the true float scores
    fsa.scores = scores

    # Define the auxilary labels. This can be anything. Simply using labels for now.
    fsa.aux_labels = labels

    fsa_vec = k2.create_fsa_vec([fsa])

    # Shortest path
    best = k2.shortest_path(fsa_vec, use_double_scores=True)
    units = best.aux_labels[:-1]

    return units


def build_initial_transition_arcs(top_k_distances, top_k_indices):
    T, k = top_k_distances.shape
    device = top_k_indices.device
    src = torch.zeros(k, dtype=torch.int32, device=device)
    dest = torch.arange(1, k + 1, dtype=torch.int32, device=device)
    labels = top_k_indices[0].to(torch.int32)
    scores = -top_k_distances[0]
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, labels, scores


def build_intermediate_transition_arcs(top_k_distances, top_k_indices, lmbda):
    T, k = top_k_indices.shape
    device = top_k_indices.device
    src = (
        torch.arange(1, (T - 1) * k + 1, dtype=torch.int32, device=device)
        .view(T - 1, k)
        .unsqueeze(-1)
        .expand(T - 1, k, k)
        .flatten()
    )
    dest = (
        torch.arange(k + 1, T * k + 1, dtype=torch.int32, device=device)
        .view(T - 1, k)
        .unsqueeze(1)
        .expand(T - 1, k, k)
        .flatten()
    )
    labels = (
        top_k_indices[1:].to(torch.int32).unsqueeze(1).expand(T - 1, k, k).flatten()
    )
    quant_scores = -top_k_distances[1:].unsqueeze(1).expand(T - 1, k, k).flatten()
    duration_scores = (
        top_k_indices[:-1].unsqueeze(-1).expand(T - 1, k, k).flatten()
        == top_k_indices[1:].unsqueeze(1).expand(T - 1, k, k).flatten()
    ).to(torch.float32) * lmbda
    scores = quant_scores + duration_scores
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, labels, scores


def build_final_transition_arcs(T, k, device):
    src = torch.arange((T - 1) * k + 1, T * k + 1, dtype=torch.int32, device=device)
    dest = torch.full_like(src, T * k + 1, dtype=torch.int32, device=device)
    labels = torch.full_like(src, -1, dtype=torch.int32, device=device)
    scores = torch.zeros_like(src, dtype=torch.float32, device=device)
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, labels, scores
