from typing import Optional

import torch


def calculate_relevance_mass_accuracy(
    relevance: torch.Tensor,
    gt: torch.Tensor,
    interpolate: bool = False,
    relevance_pooling_type: Optional[str] = None,
):
    if interpolate:
        relevance = torch.nn.functional.interpolate(
            relevance, gt.size()[1:], mode="bilinear"
        )
    if relevance_pooling_type is not None:
        relevance = pool_relevance(relevance, relevance_pooling_type)
    in_mask_relevance = torch.sum(relevance * gt)
    return in_mask_relevance / torch.sum(relevance)


def calculate_relevance_rank_accuracy(
    relevance: torch.Tensor,
    gt: torch.Tensor,
    interpolate: bool = False,
    relevance_pooling_type: Optional[str] = None,
):
    if interpolate:
        relevance = torch.nn.functional.interpolate(
            relevance, gt.size()[1:], mode="bilinear"
        )
    if relevance_pooling_type is not None:
        relevance = pool_relevance(relevance, relevance_pooling_type)
    relevance = relevance.flatten()
    gt = gt.flatten()
    gt_size = torch.sum(gt, dtype=torch.int)
    top_k = torch.argsort(relevance)[-gt_size:]
    score_mask = torch.zeros_like(relevance)
    score_mask[top_k] = 1

    return torch.sum(gt * score_mask) / gt_size


def pool_relevance(relevance, relevance_pooling_type: str):
    if relevance_pooling_type == "pos_sum":
        relevance = torch.sum(torch.nn.functional.relu(relevance), dim=0)
    elif relevance_pooling_type == "sum_pos":
        relevance = torch.nn.functional.relu(torch.sum(relevance, dim=0))
    else:
        raise Exception(
            f"Unknown relevance_pooling_type '{relevance_pooling_type}'"
        )
    return relevance
