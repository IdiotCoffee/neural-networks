import torch
from torch import nn
from torch.nn import functional as F

batch_size = 4  # how many independent sequences do we process in parallel?
block_size = 8  # what is the max context length we use for predictions
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_interval = 300
eval_iters = 200
n_emb = 32
learning_rate = 1e-3


print("initialized device to cuda, expect faster training!")
with open("fairy_tales.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset:", len(text))

# print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("".join(chars))
# print(vocab_size)

# TODO: use better tokenizers - tiktoken

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# print(encode("hello, world!"))
# print(decode(encode("hello, world!")))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
max_iters = 15000
train_data[: block_size + 1]

torch.manual_seed(1337)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _logits, loss = model(X, Y)

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])

    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # print(y)
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch()
print("inputs:", xb.shape, "\toutputs:", yb.shape)

print("---------------------")

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)


        v = self.value(x)
        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token will directly read off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        # self.sa_head = Head(n_emb)
        self.sa_heads = MultiHeadAttention(4, n_emb//4)
        self.lm_head = nn.Linear(n_emb, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # crop idx to the last block_size tokens to fit position embeddings
        idx_cond = idx[:, -block_size:]
        token_emb = self.token_embedding_table(idx_cond)
        pos_emb = self.position_embedding_table(
            torch.arange(idx_cond.shape[1], device=device)
        )
        x = token_emb + pos_emb
        # x = self.sa_head(x)
        x = self.sa_heads(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _loss = self(idx)  # get the predictions
            logits = logits[:, -1, :]  # focus only on the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
logits, loss = m(xb, yb)

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generate_tokens = m.generate(idx=idx, max_new_tokens=100)[0].tolist()
print("tokens generated: ", decode(generate_tokens))

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

batch_size = 32
for steps in range(max_iters):
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {steps}: training loss: {losses['train']:.4f} val loss: {losses['val']:.4f}"
        )

    xb, yb = get_batch()
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # if steps % 100 == 0:
    # print(f"{steps=} - {loss.item()=}")

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generate_tokens = m.generate(idx=idx, max_new_tokens=1000)[0].tolist()
print("tokens generated: ", decode(generate_tokens))

# now, let's tackle "self attention"
# what we are doing is basically allowing tokens at a spot t to communicate with all
# other tokens BEFORE IT.


"""
X_bag_of_words = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #(t tokens before it, C dimensions of info)
        X_bag_of_words[b, t] = torch.mean(xprev, 0)
"""

# now we see a single Head perform self-attention:
# head_size = 16
