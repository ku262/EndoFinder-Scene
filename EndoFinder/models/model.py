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
from timm.models.vision_transformer import PatchEmbed, Block

from EndoFinder.util.pos_embed import get_2d_sincos_pos_embed
from timm.models.layers import to_2tuple
from .layers import Transformer, FeedForward


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x)

class HashLayer(nn.Module):

    def threshold_function(self, tensor):
        # 将大于等于0的元素变为1，小于0的元素变为-1
        return torch.where(tensor >= 0, torch.tensor(1).to(tensor), torch.tensor(-1).to(tensor))

    def forward(self, x):
        return self.threshold_function(x)

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., num_classes=512, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # patch encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)#224/16 * 224/16 , embed_dim [B, 196, embed_dim]
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scene_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # self.head = FeedForward(embed_dim, embed_dim*2)
        

        self.embeddings = L2Norm()
        self.hash_layer = HashLayer()
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Scene encoder specifics
        self.scene_transform = Transformer(embed_dim, depth=1, heads=num_heads, dim_head=embed_dim//num_heads,
                                       mlp_dim=embed_dim*2, selfatt=True) #2
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.decoder_transformer = Transformer(decoder_embed_dim, depth=2, heads=decoder_num_heads, dim_head=embed_dim // 12,
                                mlp_dim=embed_dim * 2, selfatt=False, kv_dim=embed_dim)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
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
        p = self.patch_embed.patch_size[0]
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
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def patch_encoder_target(self, target, mask_ratio):
        
        # target : [B, 1, 3, H, W]
        y = target.flatten(0, 1)
        y = self.patch_embed(y)
        y = y + self.pos_embed[:, 1:, :]
        y, mask, ids_restore = self.random_masking(y, mask_ratio)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        y = torch.cat((cls_token.expand(y.shape[0], -1, -1), y), dim=1)

        for blk in self.blocks:
            y = blk(y)
        y = self.norm(y)

        return y, mask, ids_restore

    def patch_encoder_scene(self, inputs):

        # inputs : [B, N, 3, H, W]
        B, N = inputs.shape[:2]
        x = inputs.flatten(0, 1)
        
        # embed patches
        x = self.patch_embed(x) #[B*N, L, dim]
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        x = x[:, :1, :]
        x = x.reshape(B, N, -1)

        return x

    def scene_encoder(self, x):
        
        scene_token = self.scene_token
        x = torch.cat((scene_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.scene_transform(x) #[256,2,1024]

        return x

    def get_scene_token(self, slsr, mean=True):
        if mean:
            slsr = torch.mean(slsr[:, 1:, :], dim=1)  
        else:
            slsr = slsr[:, :1, :]
            slsr = slsr.flatten(0, 1)   
        slsr = self.embeddings(slsr)#进行L2Norm
        return slsr

    def forward_decoder(self, x, y, ids_restore):
        # embed tokens
        
        y = self.decoder_embed(y)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(y.shape[0], ids_restore.shape[1] + 1 - y.shape[1], 1)
        y_ = torch.cat([y[:, 1:, :], mask_tokens], dim=1)  # no cls token
        y_ = torch.gather(y_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, y.shape[2]))  # unshuffle
        y = torch.cat([y[:, :1, :], y_], dim=1)

        # add pos embed
        y = y + self.decoder_pos_embed
        
        y = y[:, 1:, :]
        
        y = self.decoder_transformer(y, x)

        # predictor projection
        y = self.decoder_pred(y)

        return y

    def forward_mse_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        imgs = imgs.flatten(0, 1)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
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
        N = embeddings.size(0)

        similarity = embeddings.matmul(embeddings.transpose(0, 1))
        
        # In the non-mixup case, instance_labels are instance ID long ints.
        # We broadcast a `==` operation to translate this to a match matrix.
        match_matrix = instance_labels.unsqueeze(1) == instance_labels.unsqueeze(0)

        identity = torch.zeros_like(match_matrix)
        identity[:, : N] = torch.eye(N).to(identity) 
        return similarity, match_matrix, identity

    def forward_sscd_loss(self, embeddings, instance_labels, entropy_weight=15, infonce_temperature=0.05):

        similarity, match_matrix, identity = self.get_similarity(
            embeddings, instance_labels
        )

        non_matches = match_matrix == 0
        nontrivial_matches = match_matrix * (~identity)

        # InfoNCE loss
        small_value = torch.tensor(-100.0).to(
            similarity
        )  # any value > max L2 normalized distance

        max_non_match_sim, _ = torch.where(non_matches, similarity, small_value).max(
            dim=1, keepdim=True
        ) #[B*2, 1] 

        logits = (similarity / infonce_temperature).exp()

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

    def forward(self, inputs, target, instance_ids, mask_ratio=0.75, entropy_weight=15):
        latent_x = self.patch_encoder_scene(inputs)
        latent_y, mask, ids_restore = self.patch_encoder_target(target, mask_ratio)
        latent_scene = self.scene_encoder(latent_x)
        embedding = self.get_scene_token(latent_scene, mean=False)

        pred = self.forward_decoder(embedding.unsqueeze(1), latent_y, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_mse_loss(target, pred, mask)
        sscd_loss, stats = self.forward_sscd_loss(embedding, instance_ids, entropy_weight=entropy_weight)
    
        return loss, pred, mask, sscd_loss, stats

    def forward_inference(self, inputs):

        x = self.patch_encoder_scene(inputs)
        
        x = self.scene_encoder(x)

        x = self.get_scene_token(x, mean=False)

        # x = self.hash_layer(x)

        return x

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, num_classes=512, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, num_classes=512, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, num_classes=512, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
