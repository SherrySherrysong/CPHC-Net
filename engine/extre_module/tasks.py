import yaml
from torch import nn

from HCA import Conv, HCA
from MCE import MCE
from PDAF import GateNCS, PDAF

from ..core import register
from ..deim.dfine_decoder import DFINETransformer
from ..deim.hybrid_encoder import ConvNormLayer_fuse, SCDown
from .fusion import Concat, DownsampleConv, FullPAD_Tunnel


@register()
class DEIM_MG(nn.Module):
    """Build the CPHC-Net backbone, encoder, and decoder from YAML."""

    __share__ = ["num_classes", "eval_spatial_size"]

    def __init__(self, yaml_path, num_classes=3, eval_spatial_size=(640, 640)):
        super().__init__()
        if tuple(eval_spatial_size) != (640, 640):
            raise ValueError("CPHC-Net expects 640 x 640 inputs for its positional embeddings.")
        with open(yaml_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)
        self.backbone, self.encoder, self.decoder, self.save = parse_model(
            config, ch=3, nc=num_classes, eval_spatial_size=eval_spatial_size
        )

    def forward(self, x, targets=None):
        y = []
        for m in (*self.backbone, *self.encoder):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers  
            x = m(x)  # run    
            y.append(x if m.i in self.save else None)  # save output
        
        x = self.decoder([y[j] for j in self.decoder.f], targets)  
        return x

    def deploy(self):
        self.eval()     
        for m in self.modules():
            if m is not self and hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self    
  

def parse_module(index, source, name, args, channels, num_classes, eval_spatial_size):
    layers = {
        cls.__name__: cls for cls in (
            Conv, PDAF, MCE, HCA, DownsampleConv, FullPAD_Tunnel,
            ConvNormLayer_fuse, Concat, GateNCS, SCDown, DFINETransformer,
        )
    }
    layer = nn.Upsample if name == "nn.Upsample" else layers[name]
    if layer is DFINETransformer:
        output_channels = args["hidden_dim"]
        module = layer(
            **args,
            feat_channels=[channels[i] for i in source],
            num_classes=num_classes,
            eval_spatial_size=eval_spatial_size,
        )
    elif layer in (Concat, FullPAD_Tunnel):
        output_channels = sum(channels[i] for i in source) if layer is Concat else channels[source[0]]
        module = layer(*args)
    elif layer is HCA:
        output_channels = args[0]
        module = layer(channels[source[1]], *args)
    elif layer in (nn.Upsample, DownsampleConv):
        output_channels = channels[source]
        module = layer(*args) if layer is nn.Upsample else layer(output_channels, *args)
    else:
        output_channels = args[0]
        module = layer(channels[source], *args)

    module.i, module.f = index, source
    return module, output_channels


def parse_model(config, ch, nc, eval_spatial_size):
    channels = [ch]
    save = set()
    sections = {}
    index = 0
    for section in ("backbone", "encoder", "decoder"):
        layers = []
        for source, name, args in config[section]:
            layer, output_channels = parse_module(
                index, source, name, args, channels, nc, eval_spatial_size
            )
            sources = [source] if isinstance(source, int) else source
            save.update(i % index for i in sources if i != -1)
            if index == 0:
                channels.clear()
            channels.append(output_channels)
            layers.append(layer)
            index += 1
        sections[section] = layers

    decoder, = sections["decoder"]
    return nn.Sequential(*sections["backbone"]), nn.Sequential(*sections["encoder"]), decoder, save
