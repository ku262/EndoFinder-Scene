import enum
import torch

from pathlib import Path
import sys
# Make sure vs is in PYTHONPATH.
base_path = str(Path(__file__).resolve().parent.parent.parent)
if base_path not in sys.path:
    sys.path.append(base_path)

from EndoFinder.util.pos_embed import interpolate_pos_embed
from EndoFinder.models import models_vit
from EndoFinder import models


class LoadModelSetting(enum.Enum):
    vit_large_patch16 = enum.auto()
    sscd = enum.auto()
    sscd_pretrained = enum.auto()
    EndoFinder_S = enum.auto()
    dino = enum.auto()
    
    def get_model(self, model_path):
        config = self._get_config(self)

        if config == "dino":
            global_pool = False
            embed_dim = 512 #ignore
            model = models_vit.__dict__["vit_large_patch16"](
                num_classes=embed_dim,
                global_pool=global_pool,
            )
            state_dict = torch.load(model_path, map_location="cpu")
            state_dict = state_dict['teacher']
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(msg)

            return model

        if config == "vit_large_patch16":

            global_pool = False
            embed_dim = 512 #ignore
            model = models_vit.__dict__[config](
                num_classes=embed_dim,
                global_pool=global_pool,
            )
            checkpoint = torch.load(model_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % model_path)
            checkpoint_model = checkpoint['model']
            state_dict = model.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            # print(msg)

            return model

        if config == "sscd":

            model = torch.jit.load("pretrain_pth/model_torchscript.pt")
            return model

        if config == "sscd_pretrained":

            model = torch.jit.load("pretrain_pth/sscd_imagenet_mixup.torchscript.pt")
            return model

        if config == "EndoFinder_S":

            model = models.model.__dict__["mae_vit_large_patch16"](norm_pix_loss=False)
            checkpoint = torch.load(model_path, map_location='cpu')
            print("Load pre-trained checkpoint from: %s" % model_path)
            checkpoint_model = checkpoint
            state_dict = model.state_dict()

            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

            # load pre-trained model
            msg = model.load_state_dict(checkpoint_model, strict=False)
            # print(msg)
            return model

    def _get_config(self, value):
        return {
            self.vit_large_patch16: "vit_large_patch16",
            self.sscd: "sscd",
            self.sscd_pretrained: "sscd_pretrained",
            self.EndoFinder_S: "EndoFinder_S",
            self.dino: "dino"
        }[value]    