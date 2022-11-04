import torch
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = "mean"

    def forward(self, pred, true):
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = alpha_factor * modulating_factor
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
class OC_Cost:
    def __init__(self, lm=0.5):
        self.lm = lm
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.focal_loss = FocalLoss(gamma=1.5, alpha=0.25)
    def getCcls(self, truth, pred):
        
        ccls = self.focal_loss(pred, truth)
        return ccls


    def getGIOU(self, truth, pred, reduction="mean"):
        lti = torch.min(truth[:, None, :2], pred[:, :2])
        rbi = torch.max(truth[:, None, 2:], pred[:, 2:])

        whi = (rbi - lti).clamp(min=0)  # [N,M,2]
        c_area = whi[:, :, 0] * whi[:, :, 1]

        intersect, union = self.getIntersectUnion(truth, pred)

        iou = intersect / (union)
        giou = iou - ((c_area - union) / c_area)

        if reduction=="mean":
            giou = giou.mean()
        elif reduction=="sum":
            giou = giou.sum()

        return giou

    def getIntersectUnion(self, truth, pred):
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(truth.T)
        area2 = box_area(pred.T)

        intersect = (torch.min(truth[:, None, 2:], pred[:, 2:]) - torch.max(truth[:, None, :2], pred[:, :2])).clamp(0).prod(2)

        union = area1[:, None] + area2 - intersect

        return intersect, union

    def getOneCost(self, truth_bbox, pred_bbox, truth_cls, pred_cls):
        #giou = self.getGIOU(truth_bbox, pred_bbox, reduction="mean")
        #ccls = self.getCcls(truth_cls, pred_cls)
        giou = self.kl_loss(truth_bbox.sigmoid(), pred_bbox.sigmoid())
        ccls = self.kl_loss(truth_cls.sigmoid(), pred_cls.sigmoid())
        cost = (self.lm * giou/2) + ((1 - self.lm) * ccls)
        return cost / truth_bbox.shape[0]

