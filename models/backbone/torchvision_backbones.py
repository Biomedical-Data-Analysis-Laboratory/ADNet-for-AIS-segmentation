import torch.nn as nn
import torchvision


class TVDeeplabRes101Encoder(nn.Module):
    """
    FCN-Resnet101 backbone from torchvision deeplabv3
    No ASPP is used as we found empirically it hurts performance
    """

    def __init__(self, use_coco_init, dataset, aux_dim_keep=64, use_aspp=False):
        super().__init__()
        self.dataset = dataset

        _model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=use_coco_init, progress=True,
                                                                     num_classes=21, aux_loss=None)
        if use_coco_init: print("###### NETWORK: Using ms-coco initialization ######")

        if "CTP" in self.dataset:
            for param in _model.parameters(): param.requires_grad = False

        _model_list = list(_model.children())

        self.layers_pre = nn.Conv2d(30,3,kernel_size=1, stride=1, bias=True)
        self.aux_dim_keep = aux_dim_keep
        self.backbone = _model_list[0]
        self.localconv = nn.Conv2d(2048, 256, kernel_size=1, stride=1, bias=False)  # reduce feature map dimension
        self.asppconv = nn.Conv2d(256, 256, kernel_size=1, bias=False)

        _aspp = _model_list[1][0]
        _conv256 = _model_list[1][1]
        self.aspp_out = nn.Sequential(*[_aspp, _conv256])
        self.use_aspp = use_aspp
        self.features = {}

        self.layers_pre.register_forward_hook(self.get_features("layers_pre"))

    def forward(self, x_in, low_level):
        """
        Args:
            low_level: whether returning aggregated low-level features in FCN
        """
        if "CTP" in self.dataset: x_in = self.layers_pre(x_in)

        fts = self.backbone(x_in)
        if self.use_aspp:
            fts256 = self.aspp_out(fts['out'])
            high_level_fts = fts256
        else:
            fts2048 = fts['out']
            high_level_fts = self.localconv(fts2048)

        if low_level:
            low_level_fts = fts['aux'][:, : self.aux_dim_keep]
            return high_level_fts, low_level_fts
        else:
            return high_level_fts

    def get_features(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()

        return hook