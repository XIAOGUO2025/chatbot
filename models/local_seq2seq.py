# models/local_seq2seq.py
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import LOCAL_CONF_THRESHOLD, LOCAL_MODEL_PATH

torch.manual_seed(42)
random.seed(42)

DATA = [
    ("你好", "你好呀！我是小智。"), ("你是谁", "我是小智。"),
    ("谢谢", "不客气！"), ("再见", "再见！"),
    ("讲个笑话", "程序员最讨厌的两件事：写注释，和别人不写注释。"),
    ("晚安", "晚安，做个好梦。"), ("早上好", "早上好！今天也要元气满满哦。"),
]

class Vocab:
    PAD, SOS, EOS, UNK = 0, 1, 2, 3
    def __init__(self):
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
    def add(self, s):
        for ch in s:
            if ch not in self.stoi: self.stoi[ch] = len(self.itos); self.itos.append(ch)
    def enc(self, s, sos=False, eos=False):
        ids = [self.stoi.get(ch, self.UNK) for ch in s]
        if sos: ids = [self.SOS] + ids
        if eos: ids = ids + [self.EOS]
        return ids
    def dec(self, ids): return "".join(self.itos[i] for i in ids if i > self.UNK)
    def __len__(self): return len(self.itos)

VOCAB = Vocab()
for a, b in DATA: VOCAB.add(a); VOCAB.add(b)

class Seq2Seq(nn.Module):
    def __init__(self, vsize, emb=128, hid=256):
        super().__init__()
        self.emb = nn.Embedding(vsize, emb, padding_idx=0)
        self.enc = nn.GRU(emb, hid, batch_first=True)
        self.dec = nn.GRU(emb, hid, batch_first=True)
        self.attn = nn.Linear(hid, hid)
        self.out = nn.Linear(hid * 2, vsize)
    def encode(self, src, lens):
        e = self.emb(src)
        packed = nn.utils.rnn.pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        out, h = self.enc(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, h
    def step(self, token, h, enc_out, mask):
        o, h = self.dec(self.emb(token), h)
        o = o.squeeze(1)
        score = torch.bmm(enc_out, self.attn(o).unsqueeze(2)).squeeze(2)
        score = score.masked_fill(mask == 0, -1e9)
        w = F.softmax(score, dim=1)
        ctx = torch.bmm(w.unsqueeze(1), enc_out).squeeze(1)
        return self.out(torch.cat([o, ctx], dim=1)), h
    def decode(self, tgt_in, enc_out, enc_h, mask):
        h, outs = enc_h, []
        for t in range(tgt_in.size(1)):
            logit, h = self.step(tgt_in[:, t:t + 1], h, enc_out, mask)
            outs.append(logit)
        return torch.stack(outs, dim=1)

def train_step(model, opt, batch):
    src = [torch.tensor(VOCAB.enc(a, eos=True)) for a, _ in batch]
    tgt = [torch.tensor(VOCAB.enc(b, sos=True, eos=True)) for _, b in batch]
    lens = torch.tensor([len(s) for s in src], dtype=torch.long)
    src_pad = nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    mask = (src_pad != 0).float()
    enc_out, enc_h = model.encode(src_pad, lens)
    logits = model.decode(tgt_pad[:, :-1], enc_out, enc_h, mask)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_pad[:, 1:].reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.item()

def train_local(epochs=100, batch_size=8, lr=1e-3):
    model = Seq2Seq(len(VOCAB), emb=128, hid=256)
    # 【核心优化】检查是否已有本地模型文件
    if os.path.exists(LOCAL_MODEL_PATH):
        print(f"  [本地模型] 发现已保存的权重 {LOCAL_MODEL_PATH}，直接秒加载！")
        model.load_state_dict(torch.load(LOCAL_MODEL_PATH, map_location="cpu"))
        return model

    print("  [本地模型] 首次运行，正在训练...")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        random.shuffle(DATA)
        for i in range(0, len(DATA), batch_size):
            train_step(model, opt, DATA[i:i + batch_size])
    
    torch.save(model.state_dict(), LOCAL_MODEL_PATH)
    print(f"  [本地模型] 训练完成，已保存至 {LOCAL_MODEL_PATH}，下次启动将极速响应！")
    return model

@torch.no_grad()
def local_chat_with_conf(text, model, max_len=40):
    if not text.strip(): return "你好像什么都没说。", 1.0
    model.eval()
    src = torch.tensor([VOCAB.enc(text, eos=True)])
    enc_out, h = model.encode(src, torch.tensor([src.size(1)]))
    mask = (src != 0).float()
    cur, ids, probs = torch.tensor([[Vocab.SOS]]), [], []
    for _ in range(max_len):
        logit, h = model.step(cur, h, enc_out, mask)
        p = F.softmax(logit, dim=-1)
        nxt = p.argmax(-1).item()
        if nxt == Vocab.EOS: break
        ids.append(nxt); probs.append(p.max().item()); cur = torch.tensor([[nxt]])
    if not ids: return "(无输出)", 0.0
    conf = sum(probs) / len(probs)
    ans = VOCAB.dec(ids)
    if conf < LOCAL_CONF_THRESHOLD:
        return "这个问题我不太有把握，换个问法或者告诉我更多背景？", conf
    return ans, conf