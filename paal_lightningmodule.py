import pytorch_lightning as pl
import torch
from torch import nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torchvision.models import resnet18
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from typing import Any
from custom.mednext import mednext_base

class ActiveLearningSegmentationModule(pl.LightningModule):
    def __init__(self, 
                 lr=1e-4, 
                 num_classes=3):
        super().__init__()
        self.save_hyperparameters()
        
        # Segmentation model
        self.seg_net = mednext_base(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=3,
            deep_supervision=True,
            use_grad_checkpoint=True
        )

        self.predictor = self._get_predictor('ap18')
        self.channels = 1
        self.num_classes = num_classes

        # Accuracy predictor (ResNet18-based regression)
        self.predictor = resnet18(pretrained=False)
        self.predictor.conv1 = nn.Conv2d(1 + num_classes, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.predictor.fc = nn.Linear(512, 1)

        # Loss functions
        self.seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.pred_loss_fn = nn.MSELoss()
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

        # AMP scaler
        self.scaler = GradScaler()

    def forward(self, x):
        return self.seg_net(x)

    def configure_optimizers(self):
        seg_optimizer = torch.optim.Adam(self.seg_net.parameters(), lr=self.hparams.lr)
        pred_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.hparams.lr)
        return [seg_optimizer, pred_optimizer]

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, labels = batch["image"], batch["label"]

        # Segmentation training
        if optimizer_idx == 0:
            seg_logits = self.seg_net(images)
            seg_loss = self.seg_loss_fn(seg_logits, labels)
            self.log("train_seg_loss", seg_loss, prog_bar=True)
            return seg_loss

        # Predictor training
        if optimizer_idx == 1:
            with torch.no_grad():
                seg_logits = self.seg_net(images)
                preds = torch.argmax(F.softmax(seg_logits, dim=1), dim=1, keepdim=True)
                dice_scores = self.dice_metric(preds, labels.unsqueeze(1)).detach()

            predictor_input = torch.cat([images, F.softmax(seg_logits, dim=1)], dim=1).mean(dim=2)
            dice_pred = self.predictor(predictor_input).squeeze(1)

            pred_loss = self.pred_loss_fn(dice_pred, dice_scores)
            self.log("train_pred_loss", pred_loss, prog_bar=True)
            return pred_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        seg_logits = self.seg_net(images)
        seg_loss = self.seg_loss_fn(seg_logits, labels)

        preds = torch.argmax(F.softmax(seg_logits, dim=1), dim=1, keepdim=True)
        dice_actual = self.dice_metric(preds, labels.unsqueeze(1))

        predictor_input = torch.cat([images, F.softmax(seg_logits, dim=1)], dim=1).mean(dim=2)
        dice_pred = self.predictor(predictor_input).squeeze(1)
        pred_loss = self.pred_loss_fn(dice_pred, dice_actual)

        self.log_dict({
            "val_seg_loss": seg_loss,
            "val_pred_loss": pred_loss,
            "val_dice_actual": dice_actual.mean(),
            "val_dice_pred": dice_pred.mean()
        }, prog_bar=True)

    def _get_predictor(self, predictor_name):
        if predictor_name.startswith('ap'):
            import model.predictor as predictor
            predictor = predictor.__dict__[predictor_name](
                input_channels=self.channels + self.num_classes, 
                num_classes=self.num_classes,
                final_drop=0.5)
        
        return predictor

    @torch.no_grad()
    def predict_accuracy(self, unlabeled_loader: Any):
        self.seg_net.eval()
        self.predictor.eval()
        scores = []

        for batch in unlabeled_loader:
            images = batch["image"].cuda()
            seg_logits = self.seg_net(images)
            seg_probs = F.softmax(seg_logits, dim=1)

            predictor_input = torch.cat([images, seg_probs], dim=1).mean(dim=2)
            dice_pred = self.predictor(predictor_input).squeeze(1)
            scores.extend(dice_pred.cpu().numpy())

        return scores
    
