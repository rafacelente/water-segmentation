from typing import Optional
import segmentation_models_pytorch as smp
from torchmetrics import Metric
import torch

class SegmentationIOU(Metric):
    def __init__(
            self, 
            reduction: Optional[str] = "micro-imagewise",
            activation: callable = torch.sigmoid,
            mask_threshold: float = 0.5,
        ):
        super().__init__()
        self.reduction = reduction
        self.activation = activation
        self.mask_threshold = mask_threshold
        self.add_state("tp", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor([]), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.tensor([]), dist_reduce_fx="sum")

    def update(
            self, 
            preds: torch.Tensor, 
            target: torch.Tensor
        ):
        """
            Update the confusion matrix with the predictions and target
            Args:
                preds: Predictions from the model
                target: Target values
        """
        prob_mask = self.activation(preds)
        pred_mask = (prob_mask > self.mask_threshold).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), target.long(), mode="binary")
        #print(f"tp: {tp},\n fp: {fp},\n fn: {fn},\n tn: {tn}\n\n")
        self.tp = torch.cat((self.tp, tp))
        self.fp = torch.cat((self.fp, fp))
        self.fn = torch.cat((self.fn, fn))
        self.tn = torch.cat((self.tn, tn))

    def compute(self):
        """
            Compute the IOU score
            Returns:
                IOU score
        """
        return smp.metrics.iou_score(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction)