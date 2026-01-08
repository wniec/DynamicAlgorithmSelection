import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


# implements skip-connection module / short-cut module
class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class PositionalEncoding:
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def get_PE(self, seq_len):
        return self.encoding[:seq_len, :]


# implements Normalization module
class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalization(x)


# implements the original Multi-head Self-Attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None, val_dim=None, key_dim=None):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # todo randn?rand
        self.W_query = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.randn(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.randn(n_heads, input_dim, val_dim))

        # self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        # self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.randn(n_heads, val_dim, embed_dim))
            # self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

    def forward(self, q, h=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, batch_size, n_query, key_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)

        # Calculate keys and values (n_heads, batch_size, graph_size, key_size or val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(
            compatibility, dim=-1
        )  # (n_heads, batch_size, n_query, graph_size)

        heads = torch.matmul(attn, V)  # (n_heads, batch_size, n_query, val_size)

        out = torch.mm(
            heads.permute(1, 2, 0, 3)
            .contiguous()
            .view(-1, self.n_heads * self.val_dim),
            # (batch_size * n_query, n_heads * val_size)
            self.W_out.view(-1, self.embed_dim),  # (n_heads * val_size, embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


# implements the encoder
class MultiHeadEncoder(nn.Module):
    def __init__(self, n_heads, embed_dim, feed_forward_hidden, normalization):
        super(MultiHeadEncoder, self).__init__()
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
            n_heads,
            embed_dim,
            normalization=normalization,
        )

        self.FFandNorm_sublayer = FFandNormsubLayer(
            embed_dim,
            feed_forward_hidden,
            normalization=normalization,
        )

    def forward(self, input):
        out = self.MHA_sublayer(input)
        return self.FFandNorm_sublayer(out)


# implements the encoder (DAC-Att sublayer)
class MultiHeadAttentionsubLayer(nn.Module):
    def __init__(
        self,
        n_heads,
        embed_dim,
        normalization,
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()

        self.MHA = MultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim)

        self.Norm = Normalization(embed_dim)

    def forward(self, x):
        # Attention
        out = self.MHA(x)

        # Residual connection and Normalization
        return self.Norm(out + x)


# implements the encoder (FFN sublayer)
class FFandNormsubLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        feed_forward_hidden,
        normalization,
    ):
        super(FFandNormsubLayer, self).__init__()

        self.FF = (
            nn.Sequential(
                nn.Linear(embed_dim, feed_forward_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(feed_forward_hidden, embed_dim),
            )
            if feed_forward_hidden > 0
            else nn.Linear(embed_dim, embed_dim)
        )

        self.Norm = Normalization(embed_dim)

    def forward(self, x):
        # FF
        out = self.FF(x)

        # Residual connection and Normalization
        return self.Norm(out + x)


class EmbeddingNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)

    def forward(self, x):
        h_em = self.embedder(x)
        return h_em


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


# neural feature extractor
class Feature_Extractor(nn.Module):
    def __init__(
        self,
        node_dim=2,  # todo: node_dim should be win * 2
        hidden_dim=16,
        n_heads=1,
        ffh=16,
        n_layers=1,
        use_positional_encoding=True,
        is_mlp=False,
    ):
        super(Feature_Extractor, self).__init__()
        # bs * dim * pop_size * 2  ->  bs * dim * pop_size * hidden_dim
        self.embedder = EmbeddingNet(node_dim=node_dim, embedding_dim=hidden_dim)
        # positional_encoding, we only add PE at before the dimension encoder
        # since each dimension should be regarded as different parts, their orders matter.
        self.is_mlp = is_mlp
        if not self.is_mlp:
            self.use_PE = use_positional_encoding
            if self.use_PE:
                self.position_encoder = PositionalEncoding(hidden_dim, 512)

            # bs * dim * pop_size * hidden_dim -> bs * dim * pop_size * hidden_dim
            self.dimension_encoder = mySequential(
                *(
                    MultiHeadEncoder(
                        n_heads=n_heads,
                        embed_dim=hidden_dim,
                        feed_forward_hidden=ffh,
                        normalization="n2",
                    )
                    for _ in range(n_layers)
                )
            )
            # the gradients are predicted by attn each dimension of a single individual.
            # bs * pop_size * dim * 128 -> bs * pop_size * dim * 128
            self.individual_encoder = mySequential(
                *(
                    MultiHeadEncoder(
                        n_heads=n_heads,
                        embed_dim=hidden_dim,
                        feed_forward_hidden=ffh,
                        normalization="n1",
                    )
                    for _ in range(n_layers)
                )
            )
        else:
            self.mlp = nn.Linear(hidden_dim, hidden_dim)
            self.acti = nn.ReLU()
        # print('------------------------------------------------------------------')
        # print('The feature extractor has been successfully initialized...')
        # print(self.get_parameter_number())
        # print('------------------------------------------------------------------')
        self.is_train = False

    def set_on_train(self):
        self.is_train = True

    def set_off_train(self):
        self.is_train = False

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # return 'Total {} parameters, of which {} are trainable.'.format(total_num, trainable_num)
        return total_num

    # todo: add a dimension representing the window
    # xs(bs * pop_size * dim) are the candidates,
    # ys(bs * pop_size) are the corresponding obj values, they are np.array
    def forward(self, xs, ys):
        if self.is_train:
            return self._run(xs, ys)
        else:
            with torch.no_grad():
                return self._run(xs, ys)

    def _run(self, xs, ys):
        _, d = xs.shape
        xs = xs[None, :, :]  # 1 * n * d
        ys = ys[None, :]
        y_ = (ys - ys.min(-1)[:, None]) / (
            ys.max(-1)[:, None] - ys.min(-1)[:, None] + 1e-12
        )
        ys = y_[:, :, None]

        # pre-processing data as the form of per_dimension_feature bs * d * n * 2
        a_x = xs[:, :, :, None]
        a_y = np.repeat(ys, d, -1)[:, :, :, None]
        raw_feature = np.concatenate([a_x, a_y], axis=-1).transpose(
            (0, 2, 1, 3)
        )  # bs * d * n * 2
        h_ij = self.embedder(
            torch.tensor(raw_feature, dtype=torch.float32)
        )  # bs * dim * pop_size * 2  ->  bs * dim * pop_size * hd
        bs, dim, pop_size, node_dim = h_ij.shape
        # resize h_ij as (bs*dim) * pop_size * hd
        h_ij = h_ij.view(-1, pop_size, node_dim)
        if not self.is_mlp:
            o_ij = self.dimension_encoder(h_ij).view(
                bs, dim, pop_size, node_dim
            )  # bs * dim * pop_size * 128 -> bs * dim * pop_size * hd
            # resize o_ij, to make each dimension of the single individual into as a group
            o_i = o_ij.permute(0, 2, 1, 3).contiguous().view(-1, dim, node_dim)
            if self.use_PE:
                o_i = o_i + self.position_encoder.get_PE(dim) * 0.5
            out = self.individual_encoder(o_i).view(
                bs, pop_size, dim, node_dim
            )  # (bs * pop_size) * dim * 128 -> bs * pop_size * dim * hidden_dim
            # bs * pop_size * hidden_dim
            out = torch.mean(out, -2)
            return out
        else:
            out = self.mlp(self.acti(h_ij)).view(bs, dim, pop_size, node_dim)
            out = torch.mean(out, -3)
            return out
