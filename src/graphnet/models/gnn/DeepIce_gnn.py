import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import math

from graphnet.models.components.layers_IceMix import Extractor, Rel_ds, Block_rel, Block, ExtractorV11Scaled
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.gnn.gnn import GNN

from timm.models.layers import trunc_normal_

from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils import to_dense_batch


class DeepIceModel(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=12,
        use_checkpoint=False,
        head_size=32,
        depth_rel=4,
        n_rel=1,
        **kwargs,
    ):
        super().__init__()
        self.extractor = Extractor(dim_base, dim)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [Block_rel(dim=dim, num_heads=dim // head_size) for i in range(depth_rel)]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        #self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)
        self.n_rel = n_rel

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0.mask
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf

        for i, blk in enumerate(self.sandwich):
            x = blk(x, attn_mask, rel_pos_bias)
            if i + 1 == self.n_rel:
                rel_pos_bias = None

        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        #x = self.proj_out(x[:, 0])  # cls token
        return x[:, 0]
    
    
class EncoderWithDirectionReconstructionV22(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=8,
        use_checkpoint=False,
        head_size=64,
        **kwargs,
    ):
        super().__init__()
        self.extractor = ExtractorV11Scaled(dim_base, dim // 2)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
            ]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        #self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.local_root = DynEdge(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        )
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0.mask
        graph_featutre = torch.concat(
            [
                x0.pos[mask],
                x0.time[mask].view(-1, 1),
                x0.auxiliary[mask].view(-1, 1),
                x0.qe[mask].view(-1, 1),
                x0.charge[mask].view(-1, 1),
                x0.ice_properties[mask],
            ],
            dim=1,
        )
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        batch_index = mask.nonzero()[:, 0]
        edge_index = knn_graph(x=graph_featutre[:, :3], k=8, batch=batch_index).to(
            mask.device
        )
        graph_featutre = self.local_root(
            graph_featutre, edge_index, batch_index, x0.n_pulses
        )
        graph_featutre, _ = to_dense_batch(graph_featutre, batch_index)

        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        x = torch.cat([x, graph_featutre], 2)

        for blk in self.sandwich:
            x = blk(x, attn_mask, rel_pos_bias)
            rel_pos_bias = None
        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        #x = self.proj_out(x[:, 0])  # cls token
        return x[:, 0]
    
    
class EncoderWithDirectionReconstructionV23(nn.Module):
    def __init__(
        self,
        dim=384,
        dim_base=128,
        depth=8,
        use_checkpoint=False,
        head_size=64,
        **kwargs,
    ):
        super().__init__()
        self.extractor = ExtractorV11Scaled(dim_base, dim // 2)
        self.rel_pos = Rel_ds(head_size)
        self.sandwich = nn.ModuleList(
            [
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
                Block_rel(dim=dim, num_heads=dim // head_size),
            ]
        )
        self.cls_token = nn.Linear(dim, 1, bias=False)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=dim // head_size,
                    mlp_ratio=4,
                    drop_path=0.0 * (i / (depth - 1)),
                    init_values=1,
                )
                for i in range(depth)
            ]
        )
        #self.proj_out = nn.Linear(dim, 3)
        self.use_checkpoint = use_checkpoint
        self.local_root = DynEdge(
            9,
            post_processing_layer_sizes=[336, dim // 2],
            dynedge_layer_sizes=[(128, 256), (336, 256), (336, 256), (336, 256)],
        )
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token.weight, std=0.02)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def forward(self, x0):
        mask = x0.mask
        graph_featutre = torch.concat(
            [
                x0.pos[mask],
                x0.time[mask].view(-1, 1),
                x0.auxiliary[mask].view(-1, 1),
                x0.qe[mask].view(-1, 1),
                x0.charge[mask].view(-1, 1),
                x0.ice_properties[mask],
            ],
            dim=1,
        )
        Lmax = mask.sum(-1).max()
        x = self.extractor(x0, Lmax)
        rel_pos_bias, rel_enc = self.rel_pos(x0, Lmax)
        # nbs = get_nbs(x0, Lmax)
        mask = mask[:, :Lmax]
        batch_index = mask.nonzero()[:, 0]
        edge_index = knn_graph(x=graph_featutre[:, :4], k=8, batch=batch_index).to(
            mask.device
        )
        graph_featutre = self.local_root(
            graph_featutre, edge_index, batch_index, x0.n_pulses
        )
        graph_featutre, _ = to_dense_batch(graph_featutre, batch_index)

        B, _ = mask.shape
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        x = torch.cat([x, graph_featutre], 2)

        for blk in self.sandwich:
            x = blk(x, attn_mask, rel_pos_bias)
            # rel_pos_bias = None
        mask = torch.cat(
            [torch.ones(B, 1, dtype=mask.dtype, device=mask.device), mask], 1
        )
        attn_mask = torch.zeros(mask.shape, device=mask.device)
        attn_mask[~mask] = -torch.inf
        cls_token = self.cls_token.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls_token, x], 1)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, None, attn_mask)
            else:
                x = blk(x, None, attn_mask)

        #x = self.proj_out(x[:, 0])  # cls token
        return x[:, 0]
    


    