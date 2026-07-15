from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from lara_align.features import PetriTraceFeatures


@dataclass
class LARAForwardOutput:
    place_embeddings: torch.Tensor
    transition_embeddings: torch.Tensor
    event_embeddings: torch.Tensor
    transition_region_probs: torch.Tensor
    event_region_probs: torch.Tensor
    sync_logits: torch.Tensor
    log_move_logits: torch.Tensor
    model_move_logits: torch.Tensor
    local_sketch_logits: torch.Tensor
    local_lower_bounds: torch.Tensor
    local_upper_bounds: torch.Tensor
    local_uncertainty: torch.Tensor


class TypedGraphTransformerLayer(nn.Module):
    """Graph transformer block with typed Petri-net arc message passing."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_edge_types: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.relation_linears = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_edge_types)]
        )
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.norm_ff = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        if node_embeddings.numel() == 0:
            return node_embeddings

        attended, _ = self.attention(
            node_embeddings.unsqueeze(0),
            node_embeddings.unsqueeze(0),
            node_embeddings.unsqueeze(0),
            need_weights=False,
        )
        attended = attended.squeeze(0)
        messages = self._typed_messages(node_embeddings, edge_index, edge_type)
        hidden = self.norm_attention(node_embeddings + self.dropout(attended + messages))
        hidden = self.norm_ff(hidden + self.dropout(self.ff(hidden)))
        return hidden

    def _typed_messages(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        messages = torch.zeros_like(node_embeddings)
        if edge_index.numel() == 0:
            return messages

        sources, targets = edge_index
        degree = torch.zeros(
            (node_embeddings.shape[0], 1),
            dtype=node_embeddings.dtype,
            device=node_embeddings.device,
        )
        for relation, linear in enumerate(self.relation_linears):
            mask = edge_type == relation
            if not torch.any(mask):
                continue
            relation_sources = sources[mask]
            relation_targets = targets[mask]
            relation_messages = linear(node_embeddings[relation_sources])
            messages.index_add_(0, relation_targets, relation_messages)
            degree.index_add_(
                0,
                relation_targets,
                torch.ones(
                    (relation_targets.shape[0], 1),
                    dtype=node_embeddings.dtype,
                    device=node_embeddings.device,
                ),
            )
        return messages / degree.clamp_min(1.0)


class PetriTraceEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        graph_layers: int = 3,
        trace_layers: int = 2,
        hash_vocab_size: int = 8192,
        dropout: float = 0.1,
        max_trace_len: int = 4096,
        num_edge_types: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.place_projection = nn.Linear(5, hidden_dim)
        self.transition_projection = nn.Linear(6, hidden_dim)
        self.label_embedding = nn.Embedding(hash_vocab_size, hidden_dim)
        self.node_type_embedding = nn.Embedding(2, hidden_dim)
        self.graph_layers = nn.ModuleList(
            [
                TypedGraphTransformerLayer(
                    hidden_dim,
                    num_heads,
                    num_edge_types=num_edge_types,
                    dropout=dropout,
                )
                for _ in range(graph_layers)
            ]
        )
        trace_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.trace_encoder = nn.TransformerEncoder(trace_layer, num_layers=trace_layers)
        self.position_embedding = nn.Embedding(max_trace_len, hidden_dim)
        self.event_to_transition_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transition_to_event_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_events = nn.LayerNorm(hidden_dim)
        self.cross_norm_transitions = nn.LayerNorm(hidden_dim)

    def forward(
        self, features: PetriTraceFeatures
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        place_embeddings = self.place_projection(features.place_features)
        transition_embeddings = self.transition_projection(features.transition_features)
        transition_embeddings = transition_embeddings + self.label_embedding(
            features.transition_label_ids
        )

        place_type = torch.zeros(
            features.num_places,
            dtype=torch.long,
            device=place_embeddings.device,
        )
        transition_type = torch.ones(
            features.num_transitions,
            dtype=torch.long,
            device=transition_embeddings.device,
        )
        nodes = torch.cat(
            [
                place_embeddings + self.node_type_embedding(place_type),
                transition_embeddings + self.node_type_embedding(transition_type),
            ],
            dim=0,
        )
        for layer in self.graph_layers:
            nodes = layer(nodes, features.edge_index, features.edge_type)

        encoded_places = nodes[: features.num_places]
        encoded_transitions = nodes[features.num_places :]
        encoded_events = self._encode_trace(features.event_label_ids)

        if encoded_events.numel() > 0 and encoded_transitions.numel() > 0:
            event_context, _ = self.event_to_transition_attention(
                encoded_events.unsqueeze(0),
                encoded_transitions.unsqueeze(0),
                encoded_transitions.unsqueeze(0),
                need_weights=False,
            )
            transition_context, _ = self.transition_to_event_attention(
                encoded_transitions.unsqueeze(0),
                encoded_events.unsqueeze(0),
                encoded_events.unsqueeze(0),
                need_weights=False,
            )
            encoded_events = self.cross_norm_events(
                encoded_events + event_context.squeeze(0)
            )
            encoded_transitions = self.cross_norm_transitions(
                encoded_transitions + transition_context.squeeze(0)
            )

        return encoded_places, encoded_transitions, encoded_events

    def _encode_trace(self, event_label_ids: torch.Tensor) -> torch.Tensor:
        if event_label_ids.numel() == 0:
            return torch.empty(
                (0, self.hidden_dim),
                dtype=self.label_embedding.weight.dtype,
                device=event_label_ids.device,
            )
        positions = torch.arange(
            event_label_ids.shape[0], dtype=torch.long, device=event_label_ids.device
        ).clamp(max=self.position_embedding.num_embeddings - 1)
        event_embeddings = self.label_embedding(event_label_ids) + self.position_embedding(
            positions
        )
        return self.trace_encoder(event_embeddings.unsqueeze(0)).squeeze(0)


class LearnedRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_regions: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transition_router = nn.Linear(hidden_dim, num_regions)
        self.event_router = nn.Linear(hidden_dim, num_regions)

    def forward(
        self, transition_embeddings: torch.Tensor, event_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition_probs = F.softmax(
            self.transition_router(self.dropout(transition_embeddings)), dim=-1
        )
        if event_embeddings.numel() == 0:
            event_probs = torch.empty(
                (0, transition_probs.shape[-1]),
                dtype=transition_probs.dtype,
                device=transition_probs.device,
            )
        else:
            event_probs = F.softmax(
                self.event_router(self.dropout(event_embeddings)), dim=-1
            )
        return transition_probs, event_probs


class LocalAlignmentExperts(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_regions: int,
        sketches_per_region: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_regions = num_regions
        self.sketches_per_region = sketches_per_region
        self.summary_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.sketch_head = nn.Linear(hidden_dim, sketches_per_region)
        self.lower_bound_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.upper_bound_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())

    def forward(
        self,
        transition_embeddings: torch.Tensor,
        event_embeddings: torch.Tensor,
        transition_region_probs: torch.Tensor,
        event_region_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        transition_summary = _weighted_region_summary(
            transition_embeddings, transition_region_probs, self.num_regions
        )
        event_summary = _weighted_region_summary(
            event_embeddings, event_region_probs, self.num_regions
        )
        summary = self.summary_projection(torch.cat([transition_summary, event_summary], dim=-1))
        sketch_logits = self.sketch_head(summary)
        lower_bounds = self.lower_bound_head(summary).squeeze(-1)
        upper_bounds = lower_bounds + self.upper_bound_head(summary).squeeze(-1)
        uncertainty = self.uncertainty_head(summary).squeeze(-1)
        return sketch_logits, lower_bounds, upper_bounds, uncertainty


class RecomposerHeads(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sync_bilinear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.log_head = nn.Linear(hidden_dim, 1)
        self.model_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        transition_embeddings: torch.Tensor,
        event_embeddings: torch.Tensor,
        compatibility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        transition_embeddings = self.dropout(transition_embeddings)
        event_embeddings = self.dropout(event_embeddings)
        if event_embeddings.numel() == 0 or transition_embeddings.numel() == 0:
            sync_logits = torch.empty(
                (event_embeddings.shape[0], transition_embeddings.shape[0]),
                dtype=transition_embeddings.dtype,
                device=transition_embeddings.device,
            )
        else:
            projected_events = self.sync_bilinear(event_embeddings)
            sync_logits = projected_events @ transition_embeddings.T
            sync_logits = sync_logits / transition_embeddings.shape[-1] ** 0.5
            sync_logits = sync_logits.masked_fill(~compatibility, -1e9)

        log_move_logits = self.log_head(event_embeddings).squeeze(-1)
        model_move_logits = self.model_head(transition_embeddings).squeeze(-1)
        return sync_logits, log_move_logits, model_move_logits


class LARANeuralModel(nn.Module):
    """Neural LARA foundation model.

    This module learns graph/trace representations, trace-dependent latent
    regions, local sketch scores, and global move scores. It intentionally does
    not certify optimality; certification belongs to `CertifyingAlignmentSystem`.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        graph_layers: int = 3,
        trace_layers: int = 2,
        num_regions: int = 8,
        sketches_per_region: int = 4,
        hash_vocab_size: int = 8192,
        dropout: float = 0.1,
        num_edge_types: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = PetriTraceEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            graph_layers=graph_layers,
            trace_layers=trace_layers,
            hash_vocab_size=hash_vocab_size,
            dropout=dropout,
            num_edge_types=num_edge_types,
        )
        self.router = LearnedRouter(hidden_dim, num_regions, dropout=dropout)
        self.local_experts = LocalAlignmentExperts(
            hidden_dim,
            num_regions=num_regions,
            sketches_per_region=sketches_per_region,
            dropout=dropout,
        )
        self.recomposer = RecomposerHeads(hidden_dim, dropout=dropout)

    def forward(self, features: PetriTraceFeatures) -> LARAForwardOutput:
        place_embeddings, transition_embeddings, event_embeddings = self.encoder(features)
        transition_region_probs, event_region_probs = self.router(
            transition_embeddings, event_embeddings
        )
        (
            local_sketch_logits,
            local_lower_bounds,
            local_upper_bounds,
            local_uncertainty,
        ) = self.local_experts(
            transition_embeddings,
            event_embeddings,
            transition_region_probs,
            event_region_probs,
        )
        sync_logits, log_move_logits, model_move_logits = self.recomposer(
            transition_embeddings,
            event_embeddings,
            features.compatibility,
        )
        return LARAForwardOutput(
            place_embeddings=place_embeddings,
            transition_embeddings=transition_embeddings,
            event_embeddings=event_embeddings,
            transition_region_probs=transition_region_probs,
            event_region_probs=event_region_probs,
            sync_logits=sync_logits,
            log_move_logits=log_move_logits,
            model_move_logits=model_move_logits,
            local_sketch_logits=local_sketch_logits,
            local_lower_bounds=local_lower_bounds,
            local_upper_bounds=local_upper_bounds,
            local_uncertainty=local_uncertainty,
        )


def _weighted_region_summary(
    embeddings: torch.Tensor,
    assignment_probs: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    if embeddings.numel() == 0:
        return torch.zeros(
            (num_regions, embeddings.shape[-1]),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
    weights = assignment_probs.T
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights @ embeddings / denominator
