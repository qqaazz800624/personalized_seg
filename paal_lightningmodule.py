import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torchvision.models import resnet18

class PAALLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Segmentation Network (e.g., MONAI 3D UNet)
        from monai.networks.nets import UNet
        self.seg_net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,  # background, liver, liver_tumor
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )

        # Accuracy Predictor (ResNet18-based regression model)
        from torchvision.models import resnet18
        self.predictor = resnet18(pretrained=False)
        self.predictor.conv1 = nn.Conv2d(
            in_channels=4,  # CT image (1 channel) + segmentation output (3 channels)
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.predictor.fc = nn.Linear(512, 1)  # Output Dice regression

        # Loss functions
        from monai.losses import DiceCELoss
        from monai.metrics import DiceMetric
        self.seg_loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
        self.predictor_loss_fn = nn.MSELoss()
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        return self.seg_net(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]

        # Segmentation model forward
        seg_logits = self.seg_net(images)
        seg_loss = self.seg_loss_fn(seg_logits, labels)

        # Compute actual dice scores for predictor
        with torch.no_grad():
            preds = torch.argmax(F.softmax(seg_logits, dim=1), dim=1, keepdim=True)
            actual_dice = self.dice_metric(preds, labels.unsqueeze(1)).detach()
        
        # Prepare predictor input
        seg_probs = F.softmax(seg_logits, dim=1)
        predictor_input = torch.cat([images, seg_probs], dim=1)  # B,4,D,H,W

        # Global average pooling across depth (reduce dimension for predictor)
        predictor_input_2d = predictor_input.mean(dim=2)  # [B,4,H,W]

        # Predictor output
        dice_pred = self.predictor(predictor_input).squeeze(1)

        # Predictor Loss
        pred_loss = self.predictor_loss_fn(dice_pred, actual_dice)

        # Segmentation Loss
        seg_loss = self.seg_loss_fn(seg_logits, labels)

        # Combined Loss
        total_loss = seg_loss + pred_loss

        # Logging
        self.log("train_seg_loss", seg_loss, prog_bar=True)
        self.log("train_pred_loss", pred_loss, prog_bar=True)
        self.log("train_total_loss", total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        seg_logits = self(images)

        # Segmentation validation
        seg_loss = self.seg_loss_fn(seg_logits, labels)
        preds = torch.argmax(F.softmax(seg_logits, dim=1), dim=1, keepdim=True)
        dice_actual = self.dice_metric(preds, labels.unsqueeze(1))

        # Predictor validation
        seg_probs = F.softmax(seg_logits, dim=1)
        predictor_input = torch.cat([images, seg_probs], dim=1).mean(dim=2)
        dice_pred = self.predictor(predictor_input).squeeze(1)

        pred_loss = self.predictor_loss_fn(dice_pred, dice_actual)

        self.log("val_seg_loss", seg_loss, prog_bar=True)
        self.log("val_pred_loss", pred_loss, prog_bar=True)
        self.log("val_dice_actual", dice_actual.mean(), prog_bar=True)
        self.log("val_dice_pred", dice_pred.mean(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
