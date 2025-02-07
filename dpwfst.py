from typing import List, Optional, Union

import k2
import torch


def dpwfst(
    features: Union[torch.Tensor, List[torch.Tensor]],
    codebook: torch.Tensor,
    lmbda: float,
    num_neighbors: Optional[int] = None,
    deduplicate: Optional[bool] = False,
):
    if isinstance(features, torch.Tensor):
        # if unbatched, make batched
        features = [features]

    # build fsa for each set of features
    fsas = [
        build_fsa(features, codebook, num_neighbors, lmbda, deduplicate)
        for features in features
    ]
    # create fsa vector so that we can process in parallel
    fsa_vec = k2.create_fsa_vec(fsas)
    # Find the shortest path(s) with tropical semiring
    best = k2.shortest_path(fsa_vec, use_double_scores=True)
    # Split back into individual FSAs
    split_fsas = [best[i] for i in range(len(features))]
    # Get units for each FSA
    units = [fsa.aux_labels[fsa.aux_labels != -1] for fsa in split_fsas]

    return units


def build_fsa(features, codebook, num_neighbors, lmbda, deduplicate):
    assert features.device == codebook.device
    device = features.device

    if num_neighbors is None:
        num_neighbors = codebook.shape[0]

    distances = torch.cdist(features, codebook, p=2.0) ** 2  # uses squared euclidean
    top_k_distances, top_k_indices = torch.topk(
        distances, k=num_neighbors, dim=1, largest=False
    )

    arcs1, out_labels1, scores1 = build_initial_transition_arcs(
        top_k_distances, top_k_indices
    )
    arcs2, out_labels2, scores2 = build_intermediate_transition_arcs(
        top_k_distances, top_k_indices, lmbda, deduplicate
    )
    arcs3, out_labels3, scores3 = build_final_transition_arcs(
        T=distances.shape[0], k=num_neighbors, device=device
    )

    arcs = torch.cat([arcs1, arcs2, arcs3], dim=0)
    out_labels = torch.cat([out_labels1, out_labels2, out_labels3], dim=0)
    scores = torch.cat([scores1, scores2, scores3], dim=0)

    # Create FSA (includes dummy int32 scores)
    fsa = k2.Fsa(arcs)
    # Now add the true float scores
    fsa.scores = scores
    # Attach the auxilary labels. This can be anything.
    # If you want you could attach the codebook tensors here.
    # This would then give you the quantized features as output.
    # For now, to obtain units, we can use the output labels of the WFST.
    fsa.aux_labels = out_labels

    return fsa


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


def build_intermediate_transition_arcs(
    top_k_distances, top_k_indices, lmbda, deduplicate=False
):
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
    repeats_unit = (
        top_k_indices[:-1].unsqueeze(-1).expand(T - 1, k, k).flatten()
        == top_k_indices[1:].unsqueeze(1).expand(T - 1, k, k).flatten()
    )
    aux_labels = labels.clone()
    if deduplicate:
        # then use epsilon transitions if the same unit is repeated
        aux_labels[repeats_unit] = -1
    quant_penalty = top_k_distances[1:].unsqueeze(1).expand(T - 1, k, k).flatten()
    duration_bonus = repeats_unit.to(torch.float32) * lmbda
    scores = -quant_penalty + duration_bonus
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, aux_labels, scores


def build_final_transition_arcs(T, k, device):
    src = torch.arange((T - 1) * k + 1, T * k + 1, dtype=torch.int32, device=device)
    dest = torch.full_like(src, T * k + 1, dtype=torch.int32, device=device)
    labels = torch.full_like(src, -1, dtype=torch.int32, device=device)
    scores = torch.zeros_like(src, dtype=torch.float32, device=device)
    dummy_scores = torch.zeros_like(scores, dtype=torch.int32, device=device)
    arcs = torch.stack([src, dest, labels, dummy_scores], dim=1)
    return arcs, labels, scores
