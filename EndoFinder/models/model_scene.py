# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from EndoFinder.util.pos_embed import get_2d_sincos_pos_embed
from .layers import Transformer

# from mae.lib.distributed_util import cross_gpu_batch

class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class HashLayer(nn.Module):

    def threshold_function(self, tensor):
        # 将大于等于0的元素变为1，小于0的元素变为-1
        return torch.where(tensor >= 0, torch.tensor(1).to(tensor), torch.tensor(-1).to(tensor))

    def forward(self, x):
        return self.threshold_function(x)

class SceneTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=512, num_heads=16,
                 decoder_embed_dim=512, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # patch encoder specifics
        self.num_patches = (img_size//patch_size)**2 
        self.patch_size = patch_size

        self.scene_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.embeddings = L2Norm()
        self.hash_layer = HashLayer()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Scene encoder specifics
        self.scene_transform = Transformer(embed_dim, depth=1, heads=num_heads, dim_head=embed_dim//num_heads,
                                       mlp_dim=embed_dim*2, selfatt=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))    

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.decoder_transformer = Transformer(decoder_embed_dim, depth=2, heads=decoder_num_heads, dim_head=embed_dim // 12,
                                mlp_dim=embed_dim * 2, selfatt=False, kv_dim=embed_dim)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.scene_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def scene_encoder(self, x):
        
        scene_token = self.scene_token
        x = torch.cat((scene_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.scene_transform(x) #[256,2,1024]

        return x

    def get_scene_token(self, slsr, mean=True):
        if mean:
            slsr = torch.mean(slsr, dim=1)  
        else:
            slsr = slsr[:, :1, :]
            slsr = slsr.flatten(0, 1)   
        # slsr = self.head(slsr)
        slsr = self.embeddings(slsr)#进行L2Norm
        return slsr

    def forward_decoder(self, x, y):
        # embed tokens
        
        y = self.decoder_embed(y)

        mask_tokens = self.mask_token.repeat(y.shape[0], self.num_patches, 1)
        
        y = torch.cat([y, mask_tokens], dim=1)

        # add pos embed
        y = y + self.decoder_pos_embed

        y = self.decoder_transformer(y, x)

        # predictor projection
        y = self.decoder_pred(y[:, 1:, :])

        return y

    def forward_mse_loss(self, imgs, pred):
        """
        imgs: [N, 1, 3, H, W]
        pred: [N, L, p*p*3]
        """
        imgs = imgs.flatten(0, 1)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = loss.mean()  # mean loss 

        return loss

    def get_similarity(self, embeddings, instance_labels):
        """Compute a cross-GPU embedding similarity matrix.

        Embeddings are gathered differentiably, via an autograd function.

        Returns a tuple of similarity, match_matrix, and indentity tensors,
        defined as follows, where N is the batch size (including copies), and
        W is the world size:
          similarity (N, W*N), float: embedding inner-product similarity between
              this GPU's embeddings, and all embeddings in the global
              (cross-GPU) batch.
          match_matrix (N, W*N), bool: cell [i][j] is True when batch[i] and
              global_batch[j] share input content (take any content from the
              same original image), including trivial pairs (comparing a
              copy with itself).
          identity (N, W*N), bool: cell [i][j] is True only for trivial pairs
              (comparing a copy with itself). This identifies the "diagonal"
              in the global (virtual) W*N x W*N similarity matrix. Since each
              GPU has only a slice of this matrix (to avoid N^2 memory use),
              the "diagonal" requires a bit of logic to identify.
        """
        """
        input:
            (no mixup)
            embeddings [B*2, dims]
            instance_labels [B*2]

        """
        # print(instance_labels.shape)
        # print(all_instance_labels.shape)
        # exit(-1)
        N = embeddings.size(0)

        similarity = embeddings.matmul(embeddings.transpose(0, 1))
        
        # In the non-mixup case, instance_labels are instance ID long ints.
        # We broadcast a `==` operation to translate this to a match matrix.
        match_matrix = instance_labels.unsqueeze(1) == instance_labels.unsqueeze(0)

        identity = torch.zeros_like(match_matrix)
        identity[:, : N] = torch.eye(N).to(identity) #对角线为True
        return similarity, match_matrix, identity

    def forward_sscd_loss(self, embeddings, instance_labels, entropy_weight=15, infonce_temperature=0.05):

        similarity, match_matrix, identity = self.get_similarity(
            embeddings, instance_labels
        )

        non_matches = match_matrix == 0
        nontrivial_matches = match_matrix * (~identity)#非自身与自身的匹配的

        # InfoNCE loss
        small_value = torch.tensor(-100.0).to(
            similarity
        )  # any value > max L2 normalized distance

        max_non_match_sim, _ = torch.where(non_matches, similarity, small_value).max(
            dim=1, keepdim=True
        ) #[B*2, 1] 不匹配中相似性最大的取出

        logits = (similarity / infonce_temperature).exp()#放大相似性高的数值

        partitions = logits + ((non_matches * logits).sum(dim=1) + 1e-6).unsqueeze(1)
        probabilities = logits / partitions


        infonce_loss = (
            -probabilities.log() * nontrivial_matches
        ).sum() / similarity.size(0)

        components = {"InfoNCE": infonce_loss}
        loss = infonce_loss
        if entropy_weight:
            # Differential entropy regularization loss.
            closest_distance = (2 - (2 * max_non_match_sim)).clamp(min=1e-6).sqrt()
            entropy_loss = -closest_distance.log().mean() * entropy_weight
            components["entropy"] = entropy_loss

            loss = infonce_loss + entropy_loss
        else:
            entropy_loss = infonce_loss
        # Log stats and loss components.
        with torch.no_grad():
            stats = {
                "positive_sim": (similarity * nontrivial_matches).sum()
                / nontrivial_matches.sum(),
                "negative_sim": (similarity * non_matches).sum() / non_matches.sum(),
                "nearest_negative_sim": max_non_match_sim.mean(),
                "center_l2_norm": embeddings.mean(dim=0).pow(2).sum().sqrt(),
                "InfoNCE": infonce_loss,
                "entropy": entropy_loss,
                "sscd_loss": loss
            }

        return loss, stats

    def forward(self, inputs, target, target_img, instance_ids, entropy_weight=15):

        latent_scene = self.scene_encoder(inputs)
        embedding = self.get_scene_token(latent_scene, mean=False)
        # embedding = self.scene_fc(latent_x)

        pred = self.forward_decoder(embedding.unsqueeze(1), target)  # [N, L, p*p*3]
        mse_loss = self.forward_mse_loss(target_img, pred)
        sscd_loss, stats = self.forward_sscd_loss(embedding, instance_ids, entropy_weight=entropy_weight)
    
        return mse_loss, pred, sscd_loss, stats

    def forward_inference(self, inputs):
        
        x = self.scene_encoder(inputs)

        x = self.get_scene_token(x, mean=False)

        # x = self.scene_fc(x)

        # x = self.hash_layer(x)

        return x
        

def scene_transformer(**kwargs):
    model = SceneTransformer( patch_size=16, embed_dim=512, num_heads=16,
        decoder_embed_dim=512, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
scene_encoder_decoder = scene_transformer  # decoder: 512 dim, 8 blocks
