# Import the required libraries
import torch
import timm, time
import torch.nn as nn
from transformers import AutoModel
from utils.env import *
from utils.loss import ContrastiveLoss
from collections import OrderedDict

import torchmetrics
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_metric_learning import losses
import torch.nn.functional as F
from torch import optim

from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision

# Language model
class LanguageModel(nn.Module):

    def __init__(self, model_name, input_size = 768, output_size = 512, dropout=0.4):
        super(LanguageModel, self).__init__()
        
        self.model_name = model_name
        self.model_path = model_modelpath_mapping[self.model_name]
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout

        # Instantiating Pre trained model object 
        self.model = AutoModel.from_pretrained(self.model_path)
        # for name, p in self.model.named_parameters():
        #     p.requires_grad = False

        # Layers
        # the first dense layer will have 768 neurons if base model is used and 
        # 1024 neurons if large model is used

        self.dense = nn.Linear(self.input_size, self.output_size)

    def forward(self, input_ids, attention_masks = None, token_type_ids = None):

        x = self.model(input_ids = input_ids, attention_mask = attention_masks,
                            token_type_ids = token_type_ids).pooler_output
        x = self.dense(x)
        return x

# TODO: Try later
# Use for non-BERT models
class NonPoolerTransformer(nn.Module):

    def __init__(self):
        super(NonPoolerTransformer, self).__init__()
        
        # Instantiating Pre trained model object 
        self.model_layer = AutoModel.from_pretrained(model_path)

        # Layers
        # the first dense layer will have 768 if base model is used and 
        # 1024 if large model is used

        self.dense_layer_1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.4)
        self.dense_layer_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(0.2)
        self.cls_layer = nn.Linear(128, 1, bias = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input_ids, attention_masks=None):

        hidden_state = self.model_layer(input_ids=input_ids, attention_mask=attention_masks)[0]
        pooled_output = hidden_state[:, 0]

        x = self.dense_layer_1(pooled_output)
        x = self.dropout(x)
        x_1 = self.dense_layer_2(x)
        x_2 = self.dropout_2(x_1)

        logits = self.cls_layer(x_2)
        output = self.sigmoid(logits)

        return output

class VisionModel(nn.Module):

    def __init__(self, model_name, hidden_size = 2048, output_size = 512, pretrained = True):
        super(VisionModel, self).__init__()

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.pretrained = pretrained

        model = timm.create_model(self.model_name, pretrained = self.pretrained)
        dense = nn.Linear(model.fc.in_features, self.output_size)
        model.reset_classifier(0)
        self.model = nn.Sequential(OrderedDict([
            ('backbone', model),
            ('dense', dense)
        ]))

    def forward(self, x):
        x = self.model(x)
        return x

class DualEncoder(pl.LightningModule):

    def __init__(self, vision_model_name, language_model_name, language_input_size = 768, 
                vision_hidden_size = 2048, output_size = 512, vision_learning_rate=1e-2, 
                language_learning_rate = 1e-5, dropout = 0.4, pretrained = True, weight_decay=1e-4,
                warmup_epochs = 2):
        super().__init__()

        # 'save_hyperparameters' saves the values of anything in the __init__ for us to the checkpoint.
        # This is a useful feature.

        self.save_hyperparameters()

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.language_input_size = language_input_size
        self.vision_hidden_size = vision_hidden_size
        self.output_size = output_size
        self.vision_learning_rate = vision_learning_rate
        self.language_learning_rate = language_learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.pretrained = pretrained
        self.warmup_epochs = warmup_epochs

        self.loss_cls = ContrastiveLoss()

        self.vision_model = VisionModel(self.vision_model_name, hidden_size = self.vision_hidden_size, 
                                        output_size = self.output_size, pretrained = self.pretrained)
        self.language_model = LanguageModel(self.language_model_name, input_size = language_input_size, 
                                            output_size = self.output_size, dropout = self.dropout)
    
        self.accuracy = torchmetrics.Accuracy()
    
    def on_epoch_start(self):
        print('\n')

    def forward(self, image, text_input_ids, attention_masks = None, token_type_ids = None):
        
        image_features = self.vision_model(image)
        text_features = self.language_model(text_input_ids, attention_masks = attention_masks, token_type_ids = token_type_ids)

        return image_features, text_features
    
    def training_step(self, batch, batch_idx, optimizer_idx):
    # def training_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])
        self.vision_scheduler.step()
        self.language_scheduler.step()
        # image_features, text_features = self.forward(batch[0], batch[1], batch[2], batch[3])
        # loss = self.loss_cls(image_features, text_features, batch[4])
        # print('Loss:', loss)

        # Logging training loss on each training step and also on each epoch
        # self.log('train_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])
        # image_features, text_features = self.forward(batch[0], batch[1], batch[2], batch[3])
        # loss = self.loss_cls(image_features, text_features, batch[4])
        # print("Val loss: ", loss)

        # Logging training loss on each training step and also on each epoch
        # self.log('val_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def test_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])

        # Logging training loss on each training step and also on each epoch
        # self.log('test_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        vision_optimizer = optim.Adam(self.vision_model.parameters(), lr=self.hparams.vision_learning_rate, weight_decay=self.weight_decay)
        language_optimizer = optim.Adam(self.language_model.parameters(), lr=self.hparams.language_learning_rate, weight_decay=self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.hparams.language_learning_rate, weight_decay=self.weight_decay)
        # return optimizer
        self.vision_scheduler = CosineAnnealingLR(vision_optimizer, T_max = self.warmup_epochs)
        self.language_scheduler = CosineAnnealingLR(language_optimizer, T_max = self.warmup_epochs)
        return [vision_optimizer, language_optimizer]


from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch.hub import load_state_dict_from_url
# from torch.utils import _log_api_usage_once

# from .._internally_replaced_utils import load_state_dict_from_url
# from ..utils import _log_api_usage_once


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        alpha = alpha.view((alpha.shape[0], 1, alpha.shape[1], 1))
        beta = beta.view((beta.shape[0], 1, 1, beta.shape[1]))
        gamma = gamma.view((gamma.shape[0], gamma.shape[1], 1, 1))

        x = x + alpha * x + beta * x + gamma * x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

        # x = self.fc(x)

        # return x

    def forward(self, x: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor) -> Tensor:
        return self._forward_impl(x, alpha, beta, gamma)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)



class SpatialInformationAggregatorModule(pl.LightningModule):

    def __init__(self, dual_encoder, height = 7, width = 7, num_channels = 2048, 
                output_size = 512, pretrained = True, learning_rate = 1e-4, weight_decay = 1e-4):
        super().__init__()

        # self.save_hyperparameters()

        self.pretrained = pretrained
        self.object_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = self.pretrained, 
                                        trainable_backbone_layers=0, rpn_post_nms_top_n_test=50)

        self.object_detection_model.eval()

        self.dual_encoder = dual_encoder
        self.dual_encoder.eval()

        for _, p in self.dual_encoder.named_parameters():
            p.requires_grad = False
        for _, p in self.object_detection_model.named_parameters():
            p.requires_grad = False

        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.input_size = self.object_detection_model.roi_heads.box_predictor.cls_score.in_features
        self.output_size = output_size

        self.roi_linear = nn.Linear(self.input_size, self.output_size)
        self.alpha_linear = nn.Linear(self.output_size, self.height)
        self.beta_linear = nn.Linear(self.output_size, self.width)
        self.gamma_linear = nn.Linear(self.output_size, self.num_channels)
        
        self.guide_vision_model = resnet101(pretrained = self.pretrained)
        self.image_linear = nn.Linear(self.num_channels, self.output_size)

        self.loss_cls = ContrastiveLoss()

        self.ROI_features = [] 
        def save_features(model, input, output):
            dim0 = output.shape[0] // 50
            output = output.view((dim0, 50, output.shape[1])) # 8 x 50 x 1024
            self.ROI_features.append(output.data) 

         # you can also hook layers inside the roi_heads
        layer_to_hook = 'roi_heads.box_head.fc7'
        # layer_to_hook = 'roi_heads.box_head'
        for name, layer in self.object_detection_model.named_modules():
            if name == layer_to_hook:
                layer.register_forward_hook(save_features)
    
    def forward(self, image, text_input_ids, attention_masks = None, token_type_ids = None):

        text_features = self.dual_encoder.language_model(text_input_ids, attention_masks = attention_masks, token_type_ids = token_type_ids)
        text_features = F.normalize(text_features, p = 2, dim = 1)

        self.ROI_features.clear()      
        _ = self.object_detection_model(image)
        roi_features = self.ROI_features[0].to(device=image.device)

        # Life lesson: Always read the parameters/documentation of models/scripts you are picking up from the internet. 
        # They might be generic enough that you don't have to waste a day :)
        
        aggregated_features = []
        for idx, roi_feature in enumerate(roi_features):
            roi_feature = self.roi_linear(roi_feature)
            roi_feature = F.normalize(roi_feature, p = 2, dim = 1)
            weights = torch.matmul(roi_feature, text_features[idx])
            aggregated_feature = torch.sum(roi_feature * weights.view((-1, 1)), dim = 0)
            aggregated_features.append(aggregated_feature)
        aggregated_features = torch.stack(aggregated_features) # 16 x 512

        alpha = self.alpha_linear(aggregated_features)
        beta = self.beta_linear(aggregated_features)
        gamma = self.gamma_linear(aggregated_features)

        image_features = self.guide_vision_model(image, alpha, beta, gamma)
        image_features = F.normalize(image_features, p = 2, dim = 1) # 16 x 2048

        image_features = self.image_linear(image_features) # 16 x 2048

        return image_features, text_features

    def training_step(self, batch, batch_idx):
        self.dual_encoder.eval()
        self.object_detection_model.eval()
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])

        return loss
    
    def validation_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])
        # image_features, text_features = self.forward(batch[0], batch[1], batch[2], batch[3])
        # loss = self.loss_cls(image_features, text_features, batch[4])
        # print("Val loss: ", loss)

        # Logging training loss on each training step and also on each epoch
        # self.log('val_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss
    
    def test_step(self, batch, batch_idx):
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])

        # Logging training loss on each training step and also on each epoch
        # self.log('test_loss', loss, on_step=True, on_epoch=True, logger=False)

        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        # vision_optimizer = optim.Adam(self.vision_model.parameters(), lr=self.hparams.vision_learning_rate, weight_decay=self.weight_decay)
        # language_optimizer = optim.Adam(self.language_model.parameters(), lr=self.hparams.language_learning_rate, weight_decay=self.weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=self.hparams.language_learning_rate, weight_decay=self.weight_decay)
        optimizer = optim.Adam([{"params": self.roi_linear.parameters()},
                                {"params": self.alpha_linear.parameters()},
                                {"params": self.beta_linear.parameters()},
                                {"params": self.gamma_linear.parameters()},
                                {"params": self.image_linear.parameters()}], lr = self.learning_rate, weight_decay = self.weight_decay)
        # return optimizer
        
        # optimizer = optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)

        # self.vision_scheduler = CosineAnnealingLR(vision_optimizer, T_max = self.warmup_epochs)
        # self.language_scheduler = CosineAnnealingLR(language_optimizer, T_max = self.warmup_epochs)
        # return [vision_optimizer, language_optimizer]

        return optimizer