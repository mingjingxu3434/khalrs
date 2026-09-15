import torch
from torch import nn
import torch.nn.functional as F
from .kg import RelationProjectionKG
from .modules import HierarchicalAttentionEncoder, CandidateKnowledgeFusion, CurriculumEncoder, AdaptiveDecoder

class KHALRS(nn.Module):
    def __init__(self, num_users, num_entities, num_relations, num_semesters, neighbor_ids, neighbor_mask, cfg):
        super().__init__()
        d = cfg["embedding_dim"]
        self.cfg = cfg
        self.kg = RelationProjectionKG(num_entities, num_relations, d, cfg.get("relation_dim", d))
        self.hae = HierarchicalAttentionEncoder(
            d, cfg.get("attention_hidden_dim", d), cfg["max_sessions"],
            cfg["max_items_per_session"], cfg["dropout"], num_users
        )
        self.ckf = CandidateKnowledgeFusion(d, cfg["num_heads"], cfg["dropout"])
        self.curriculum = CurriculumEncoder(d, cfg["semester_dim"], num_semesters, cfg["dropout"])
        self.decoder = AdaptiveDecoder(d, cfg["dropout"], cfg.get("decoder_activation", "gelu"))
        self.register_buffer("neighbor_ids", neighbor_ids.long(), persistent=False)
        self.register_buffer("neighbor_mask", neighbor_mask.bool(), persistent=False)

    def encode_user(self, batch):
        hist = self.kg.entity(batch["history_items"])
        dom = self.kg.entity(batch["domains"])
        hu, attn = self.hae(
            batch["user_id"], hist, batch["item_mask"], batch["session_mask"],
            dom, batch["domain_mask"]
        )
        courses = self.kg.entity(batch["courses"])
        curr = self.curriculum(batch["semester_id"], courses, batch["course_mask"])
        return hu, curr, attn

    def score_candidates(self, batch, candidate_ids, encoded=None, return_aux=False):
        if candidate_ids.dim() == 1:
            candidate_ids = candidate_ids[:, None]
        hu, curr, attn = encoded if encoded is not None else self.encode_user(batch)
        nids = self.neighbor_ids[candidate_ids]
        nmask = self.neighbor_mask[candidate_ids]
        fused, e_kg, cross = self.ckf(
            hu, self.kg.entity(candidate_ids), self.kg.entity(nids), nmask
        )
        scores = self.decoder(fused, e_kg, curr)
        if return_aux:
            return scores, {"user_state": hu, "curriculum": curr, "hae_attention": attn, "cross_attention": cross}
        return scores

    def forward(self, batch):
        cand = torch.cat([batch["positive"][:,None], batch["negatives"]], 1)
        scores = self.score_candidates(batch, cand)
        return scores[:,0], scores[:,1:]

    def curriculum_loss(self, batch, curr, margin):
        emb = self.kg.entity(batch["courses"])
        cos = F.cosine_similarity(curr[:,None], emb, dim=-1)
        loss = F.relu(margin - cos) * batch["course_mask"].to(cos.dtype)
        return loss.sum() / batch["course_mask"].sum().clamp_min(1)

    def l2_penalty(self):
        return sum((p**2).sum() for p in self.parameters() if p.requires_grad)

    def load_kg_state(self, path, freeze=False):
        x = torch.load(path, map_location="cpu")
        self.kg.load_state_dict(x["kg"] if "kg" in x else x)
        if freeze:
            for p in self.kg.parameters():
                p.requires_grad_(False)
