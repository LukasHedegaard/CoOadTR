from torch import nn
from .Attention import SelfAttention
from continual_transformers import CoSiTransformerEncoder, CoReSiTransformerEncoder


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def CoTransformerModel(
    dim,
    depth,
    heads,
    mlp_dim,
    dropout_rate=0.1,
    attn_dropout_rate=0.1,
    sequence_len=64,
):
    assert depth in {1, 2}

    if depth == 1:
        return CoSiTransformerEncoder(
            sequence_len=sequence_len,
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout_rate,
            in_proj_bias=False,
            query_index=-1,
            ff_hidden_dim=mlp_dim,
            ff_activation=nn.GELU(),
            device=None,
            dtype=None,
        )

    # depth == 2
    return CoReSiTransformerEncoder(
        sequence_len=sequence_len,
        embed_dim=dim,
        num_heads=heads,
        dropout=dropout_rate,
        in_proj_bias=False,
        query_index=-1,
        ff_hidden_dim=mlp_dim,
        ff_activation=nn.GELU(),
        device=None,
        dtype=None,
    )


def _register_ptflops():
    try:
        from ptflops import flops_counter as fc

        def self_attention_counter_hook(module, input, output):
            flops = 0

            q = input[0]
            k = input[0]
            v = input[0]
            batch_size = q.shape[1]

            num_heads = module.num_heads
            embed_dim = module.qkv.in_features
            kdim = embed_dim
            vdim = embed_dim

            # initial projections
            flops = (
                q.shape[0] * q.shape[2] * embed_dim
                + k.shape[0] * k.shape[2] * kdim
                + v.shape[0] * v.shape[2] * vdim
            )
            if module.qkv.bias is not None:
                flops += (q.shape[0] + k.shape[0] + v.shape[0]) * embed_dim

            # attention heads: scale, matmul, softmax, matmul
            head_dim = embed_dim // num_heads
            head_flops = (
                q.shape[0] * head_dim
                + head_dim * q.shape[0] * k.shape[0]
                + q.shape[0] * k.shape[0]
                + q.shape[0] * k.shape[0] * head_dim
            )

            flops += num_heads * head_flops

            # final projection, bias is always enabled
            flops += q.shape[0] * embed_dim * (embed_dim + 1)

            flops *= batch_size
            module.__flops__ += int(flops)

        fc.MODULES_MAPPING[SelfAttention] = self_attention_counter_hook

    except ModuleNotFoundError:  # pragma: no cover
        pass
    except Exception as e:  # pragma: no cover
        print(f"Failed to add flops_counter_hook: {e}")


_register_ptflops()
