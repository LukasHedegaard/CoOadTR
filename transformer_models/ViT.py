from typing import Tuple
import torch
from torch.functional import Tensor
import torch.nn as nn
from .Transformer import TransformerModel, CoTransformerModel
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
import continual as co
from continual_transformers import RecyclingPositionalEncoding

__all__ = ["ViT_B16", "ViT_B32", "ViT_L16", "ViT_L32", "ViT_H14"]


def CoVisionTransformer(
    args,
    img_dim,
    patch_dim,
    out_dim,
    embedding_dim,
    num_heads,
    num_layers,
    hidden_dim,
    dropout_rate=0.0,
    attn_dropout_rate=0.0,
    use_representation=True,
    conv_patch_representation=False,
    positional_encoding_type="learned",
    with_camera=True,
    with_motion=True,
    num_channels=3072,
):

    assert embedding_dim % num_heads == 0
    assert img_dim % patch_dim == 0

    num_patches = int(img_dim // patch_dim)
    seq_length = num_patches  # no class token
    flatten_dim = patch_dim * patch_dim * num_channels

    linear_encoding = co.Linear(flatten_dim, embedding_dim, channel_dim=1)
    assert "recycling" in positional_encoding_type
    position_encoding = RecyclingPositionalEncoding(
        embedding_dim,
        args.num_embeddings,
        learned="learned" in positional_encoding_type,
        forward_update_index_steps=1,
    )
    print("position encoding :", positional_encoding_type)

    pe_dropout = nn.Dropout(p=dropout_rate)

    encoder = CoTransformerModel(
        embedding_dim,
        num_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        attn_dropout_rate,
    )
    pre_head_ln = co.Lambda(nn.LayerNorm(embedding_dim), takes_time=False)
    mlp_head = co.Linear(hidden_dim, out_dim, channel_dim=1)

    return co.Sequential(
        linear_encoding,
        position_encoding,
        pe_dropout,
        encoder,
        pre_head_ln,
        mlp_head,
    )


class VisionTransformer_v3(nn.Module):
    def __init__(
        self,
        args,
        img_dim,
        patch_dim,
        out_dim,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned",
        with_camera=True,
        with_motion=True,
        num_channels=3072,
    ):
        super(VisionTransformer_v3, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.with_camera = with_camera
        self.with_motion = with_motion
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        # num_channels = img_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        # self.num_patches = int((img_dim // patch_dim) ** 2)
        self.num_patches = int(img_dim // patch_dim)
        self.seq_length = self.num_patches  # + 1 # Remove class token
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print("position encoding :", positional_encoding_type)

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        use_representation = False  # False
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, inputs: Tuple[Tensor, Tensor]):
        sequence_input_rgb, sequence_input_flow = inputs
        if self.with_camera and self.with_motion:
            x = torch.cat((sequence_input_rgb, sequence_input_flow), 2)
        elif self.with_camera:
            x = sequence_input_rgb
        elif self.with_motion:
            x = sequence_input_flow

        x = self.linear_encoding(x)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # [128, 33, 1024]
        x = self.mlp_head(x)
        # x = F.log_softmax(x, dim=-1)

        return x[:, -1]  # x

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ["SAME", "VALID"]
        if padding_type == "SAME":
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


def ViT_B16(dataset="imagenet"):
    if dataset == "imagenet":
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif "cifar" in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer_v3(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_B32(dataset="imagenet"):
    if dataset == "imagenet":
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif "cifar" in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer_v3(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L16(dataset="imagenet"):
    if dataset == "imagenet":
        img_dim = 224
        out_dim = 1000
        patch_dim = 16
    elif "cifar" in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer_v3(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_L32(dataset="imagenet"):
    if dataset == "imagenet":
        img_dim = 224
        out_dim = 1000
        patch_dim = 32
    elif "cifar" in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer_v3(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )


def ViT_H14(dataset="imagenet"):
    if dataset == "imagenet":
        img_dim = 224
        out_dim = 1000
        patch_dim = 14
    elif "cifar" in dataset:
        img_dim = 32
        out_dim = 10
        patch_dim = 4

    return VisionTransformer_v3(
        img_dim=img_dim,
        patch_dim=patch_dim,
        out_dim=out_dim,
        num_channels=3,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    )
