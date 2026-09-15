import torch.nn.functional as F

def bpr_loss(positive_scores, negative_scores):
    diff = positive_scores[:, None] - negative_scores
    return -F.logsigmoid(diff).mean()
