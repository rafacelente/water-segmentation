import pytorch_lightning as pl
import torch.nn as nn
import segmentation_models_pytorch as smp
from metrics import SegmentationIOU
import torch

class UNETModule(pl.LightningModule):
    def __init__(
        self,
        model="unet",
        encoder="resnet152",
        encoder_weights="imagenet",
        loss_fn="dice",
        ):
        super(UNETModule, self).__init__()
        if model == "unet":
            self.model = smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
        elif model == "deeplabv3":
            self.model = smp.DeepLabV3(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
        elif model == "FPN":
            self.model = smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
        elif model == "unetpp":
            self.model = smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=3, classes=1)
        else:
            raise ValueError(f"Model type {model} not supported")
        
        self.cross_entroy = False
        if loss_fn == "dice":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss_fn == "jaccard":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.BINARY_MODE, from_logits=True)
        elif loss_fn == "crossentropy":
            self.cross_entropy = True
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError(f"Loss function {loss_fn} not supported")
            
        self.validation_iou = SegmentationIOU(
            reduction="micro",#"micro-imagewise",
            activation=torch.sigmoid,
            mask_threshold=0.5
        )
        self.training_iou = SegmentationIOU(
            reduction="micro",#"micro-imagewise",
            activation=torch.sigmoid,
            mask_threshold=0.5,
            
        )
        self.test_iou = SegmentationIOU(
            reduction="micro",#"micro-imagewise",
            activation=torch.sigmoid,
            mask_threshold=0.5,
            
        )

    @classmethod
    def from_byol(cls, byol_module):
        unet = cls()
        unet.model.encoder.load_state_dict(byol_module.backbone.state_dict())
        return unet

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage):
        """
            Shared step for training, validation and test steps
            Based on https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb

            Args:
                batch: Tuple of (image, mask)
                batch_idx: Index of the batch
            Returns:
                Dict with loss, true positive, false positive, false negative and true negative
        """
        #x, y = batch # image, mask
        x = batch["image"].float()
        y = batch["mask"].float()
        logits = self(x)
        assert logits.shape == y.shape, f"Output shape {logits.shape} does not match target shape {y.shape}"
        assert y.max() <= 1 and y.min() >= 0, f"Target mask should be binary, but got {y.min()} and {y.max()}"
        if self.cross_entropy:
            preds = torch.sigmoid(logits)
            loss = self.loss_fn(preds, y)
        else:
            loss = self.loss_fn(logits, y)
            
        return loss, logits, y

    def on_train_epoch_start(self):
        self.training_iou.reset()
    
    def training_step(self, batch, batch_idx):
        loss, logits, masks = self.shared_step(batch, "train")
        self.training_iou.update(logits, masks)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "loss":loss,
            "tp": self.training_iou.tp,
            "fp": self.training_iou.fp,
            "fn": self.training_iou.fn,
            "tn": self.training_iou.tn
        }
    
    def on_train_epoch_end(self):
        iou = self.training_iou.compute()
        self.log("train_iou", iou, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self.validation_iou.reset()
    
    def validation_step(self, batch, batch_idx):
        loss, logits, masks = self.shared_step(batch, "val")
        self.validation_iou.update(logits, masks)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "loss":loss,
            "tp": self.validation_iou.tp,
            "fp": self.validation_iou.fp,
            "fn": self.validation_iou.fn,
            "tn": self.validation_iou.tn
        }
    
    def on_validation_epoch_end(self):
        iou = self.validation_iou.compute()
        self.log("val_iou", iou, on_epoch=True, prog_bar=True)

    def on_test_epoch_start(self):
        self.test_iou.reset()
    
    def test_step(self, batch, batch_idx):
        loss, logits, masks = self.shared_step(batch, "test")
        self.test_iou.update(logits, masks)
        return {
            "loss":loss,
            "tp": self.test_iou.tp,
            "fp": self.test_iou.fp,
            "fn": self.test_iou.fn,
            "tn": self.test_iou.tn
        }
    
    def on_test_epoch_end(self):
        iou = self.test_iou.compute()
        self.log("test_iou", iou, on_epoch=True, prog_bar=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        filename, x = batch
        x = x.float()
        logits = self(x)
        preds = (torch.sigmoid(logits) > 0.6).float()
        return logits, preds, filename
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': {
               'scheduler': scheduler,
               'monitor': 'train_loss',
               'interval': 'epoch',
               'frequency': 1,
           }
        }