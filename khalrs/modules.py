import math
import torch
from torch import nn
import torch.nn.functional as F

def masked_softmax(logits, mask, dim=-1):
    mask = mask.bool()
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)

class AdditiveAttention(nn.Module):
    def __init__(self, d, d_a):
        super().__init__()
        self.proj = nn.Linear(d, d_a)
        self.out = nn.Linear(d_a, 1, bias=False)
    def forward(self, x, mask, extra=None):
        h = self.proj(x)
        if extra is not None:
            h = h + extra
        logits = self.out(torch.tanh(h)).squeeze(-1)
        w = masked_softmax(logits, mask, dim=-1)
        return (w.unsqueeze(-1) * x).sum(dim=-2), w

class HierarchicalAttentionEncoder(nn.Module):
    def __init__(self, d, d_a, max_sessions, max_items, dropout, num_users):
        super().__init__()
        self.item_proj = nn.Linear(d, d)
        self.position = nn.Embedding(max_items, d)
        self.item_ln = nn.LayerNorm(d)
        self.item_attn = AdditiveAttention(d, d_a)
        self.session_proj = nn.Linear(d, d)
        self.recency = nn.Embedding(max_sessions, d)
        self.session_ln = nn.LayerNorm(d)
        self.user_context = nn.Embedding(num_users, d)
        self.session_h = nn.Linear(d, d_a)
        self.session_c = nn.Linear(d, d_a, bias=False)
        self.session_out = nn.Linear(d_a, 1, bias=False)
        self.domain_p = nn.Linear(d, d_a)
        self.domain_d = nn.Linear(d, d_a, bias=False)
        self.domain_out = nn.Linear(d_a, 1, bias=False)
        self.user_out = nn.Linear(2 * d, d)
        self.user_ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_ids, history_emb, item_mask, session_mask, domain_emb, domain_mask):
        b, s, l, d = history_emb.shape
        pos = torch.arange(l, device=history_emb.device)
        x = self.item_ln(self.item_proj(history_emb) + self.position(pos)[None, None])
        x = self.dropout(x)

        z, alpha = self.item_attn(x.reshape(b*s, l, d), item_mask.reshape(b*s, l))
        z = z.reshape(b, s, d)

        rec = torch.arange(s, device=z.device)
        hs = self.session_ln(self.session_proj(z) + self.recency(rec)[None])
        hs = self.dropout(hs)
        cu = self.user_context(user_ids)
        slogits = self.session_out(torch.tanh(
            self.session_h(hs) + self.session_c(cu).unsqueeze(1)
        )).squeeze(-1)
        beta = masked_softmax(slogits, session_mask, dim=-1)
        pu = (beta.unsqueeze(-1) * hs).sum(1)

        dlogits = self.domain_out(torch.tanh(
            self.domain_p(pu).unsqueeze(1) + self.domain_d(domain_emb)
        )).squeeze(-1)
        delta = masked_softmax(dlogits, domain_mask, dim=-1)
        ou = (delta.unsqueeze(-1) * domain_emb).sum(1)

        hu = self.user_ln(self.user_out(torch.cat([pu, ou], dim=-1)))
        hu = self.dropout(hu)
        return hu, {"item": alpha.reshape(b, s, l), "session": beta, "domain": delta}

class CandidateKnowledgeFusion(nn.Module):
    def __init__(self, d, num_heads, dropout):
        super().__init__()
        assert d % num_heads == 0
        self.d = d
        self.h = num_heads
        self.dh = d // num_heads
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d)
        self.kg_ln = nn.LayerNorm(d)
        self.g1 = nn.Linear(2*d, d)
        self.g2 = nn.Linear(d, d)
        self.fused_ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, user_state, candidate_emb, neighbor_emb, neighbor_mask):
        b, c, k, d = neighbor_emb.shape
        q = self.q(user_state).view(b, self.h, self.dh)[:, None, :, :]  # B,1,H,Dh
        kk = self.k(neighbor_emb).view(b,c,k,self.h,self.dh).permute(0,1,3,2,4)
        vv = self.v(neighbor_emb).view(b,c,k,self.h,self.dh).permute(0,1,3,2,4)
        logits = (q.unsqueeze(3) * kk).sum(-1) / math.sqrt(self.dh)
        mask = neighbor_mask[:, :, None, :]
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        a = torch.softmax(logits, -1) * mask.to(logits.dtype)
        a = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
        ctx = (a.unsqueeze(-1) * vv).sum(3).reshape(b, c, d)
        e_kg = self.kg_ln(candidate_emb + self.out(ctx))
        e_kg = self.dropout(e_kg)
        u = user_state[:, None].expand(-1, c, -1)
        gate = torch.sigmoid(self.g2(F.gelu(self.g1(torch.cat([u, e_kg], -1)))))
        fused = self.fused_ln(gate*u + (1-gate)*e_kg)
        return fused, e_kg, a

class CurriculumEncoder(nn.Module):
    def __init__(self, d, semester_dim, num_semesters, dropout):
        super().__init__()
        self.semester = nn.Embedding(num_semesters, semester_dim)
        self.fc1 = nn.Linear(semester_dim + d, 2*d)
        self.fc2 = nn.Linear(2*d, d)
        self.ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
    def forward(self, semester_ids, course_emb, course_mask):
        w = course_mask.to(course_emb.dtype).unsqueeze(-1)
        mean_course = (course_emb*w).sum(1) / w.sum(1).clamp_min(1.0)
        r = torch.cat([self.semester(semester_ids), mean_course], -1)
        return self.ln(self.fc2(self.dropout(F.gelu(self.fc1(r)))))

class AdaptiveDecoder(nn.Module):
    def __init__(self, d, dropout, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(5*d, 2*d)
        self.fc2 = nn.Linear(2*d, d)
        self.fc3 = nn.Linear(d, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation.lower() == "gelu" else F.relu
    def forward(self, fused, e_kg, curr):
        curr = curr[:, None].expand_as(fused)
        phi = torch.cat([fused, e_kg, curr, fused*e_kg, fused*curr], -1)
        h = self.dropout(self.act(self.fc1(phi)))
        h = self.dropout(self.act(self.fc2(h)))
        return self.fc3(h).squeeze(-1)
