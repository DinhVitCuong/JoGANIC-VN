import torch
import torch.nn as nn
from .modules import MultiHeadAttention, GehringLinear

class _DecoderBlock(nn.Module):
    """Single hybrid block attending to image, text and named entity contexts."""

    def __init__(self, embed_dim: int, img_dim: int, text_dim: int, ne_dim: int,
                 num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn_img = MultiHeadAttention(embed_dim, num_heads, kdim=img_dim, vdim=img_dim, dropout=dropout)
        self.attn_txt = MultiHeadAttention(embed_dim, num_heads, kdim=text_dim, vdim=text_dim, dropout=dropout)
        self.attn_ent = MultiHeadAttention(embed_dim, num_heads, kdim=ne_dim, vdim=ne_dim, dropout=dropout)
        self.proj = GehringLinear(embed_dim * 3, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            GehringLinear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            GehringLinear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, contexts):
        q = self.ln1(x)
        img, _ = self.attn_img(q, contexts['image'], contexts['image'],
                               key_padding_mask=contexts.get('image_mask'))
        txt, _ = self.attn_txt(q, contexts['text'], contexts['text'],
                               key_padding_mask=contexts.get('text_mask'))
        ent, _ = self.attn_ent(q, contexts['entity'], contexts['entity'],
                               key_padding_mask=contexts.get('entity_mask'))
        concat = torch.cat([img, txt, ent], dim=-1)
        out = self.proj(concat)
        x = x + self.dropout(out)
        x = self.ln2(x)
        x = x + self.dropout(self.ff(x))
        return x

class TemplateGuidedDecoder(nn.Module):
    """Hybrid decoder guided by template component probabilities."""

    def __init__(self, vocab_size: int, embed_dim: int = 1024,
                 img_dim: int = 2048, text_dim: int = 1024, ne_dim: int = 1024,
                 num_heads: int = 8, ff_dim: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.shared = nn.ModuleList([
            _DecoderBlock(embed_dim, img_dim, text_dim, ne_dim, num_heads, ff_dim, dropout)
            for _ in range(3)
        ])
        self.components = nn.ModuleList([
            _DecoderBlock(embed_dim, img_dim, text_dim, ne_dim, num_heads, ff_dim, dropout)
            for _ in range(5)
        ])
        self.output = GehringLinear(embed_dim, vocab_size)

    def forward(self, tgt_tokens: torch.Tensor, contexts: dict, alpha: torch.Tensor):
        """Generate logits for ``tgt_tokens`` conditioned on contexts and template weights ``alpha``.

        Parameters
        ----------
        tgt_tokens: ``(B, T)`` token indices of partial caption.
        contexts: dictionary containing ``image``, ``text`` and ``entity`` keys with
            shapes ``(B, L, C)``. Masks can be provided with ``*_mask`` keys.
        alpha: ``(B, 5)`` component weights between 0 and 1.
        """
        x = self.embed_tokens(tgt_tokens).transpose(0, 1)  # T,B,E
        for block in self.shared:
            x = block(x, contexts)
        comp_out = []
        for block in self.components:
            comp_out.append(block(x, contexts))
        B = tgt_tokens.size(0)
        final = 0
        for i, out in enumerate(comp_out):
            w = alpha[:, i].view(1, B, 1)
            final = final + w * out
        final = final / len(comp_out)
        final = final.transpose(0, 1)
        logits = self.output(final)
        return logits
