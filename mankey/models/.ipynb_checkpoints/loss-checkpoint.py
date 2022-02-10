import torch
from torch.nn.modules.loss import _Loss

def of_l1_loss(kpt_of_pred, kpt_of_gt, normalize=True, reduce=False):
    '''
    :param kpt_of_pred:  [bs, n_pts, c]
    :param kpt_of_gt:    [bs, n_pts, c]
    '''
    bs, n_pts, c = kpt_of_pred.size()
    diff = kpt_of_pred - kpt_of_gt
    abs_diff = torch.abs(diff)
    in_loss = abs_diff

    if normalize:
        in_loss = torch.sum(in_loss.view(bs, -1), 1) / (n_pts * c)
        
    if reduce:
        torch.mean(in_loss)

    return in_loss


class OFLoss(_Loss):
    def __init__(self):
        super(OFLoss, self).__init__(True)

    def forward(self, kpt_of_pred, kpt_of_gt, reduce=False):
        l1_loss = of_l1_loss(kpt_of_pred, kpt_of_gt, reduce=False)

        return l1_loss