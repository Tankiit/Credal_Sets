"""Learned edit operators and held-out factor readers (PyTorch).

Both operators see the same thing: a standardized source latent state and the
requested action (one-hot target level of the edited factor). Neither sees the
target image at inference. They start at the identity map.
"""
import copy
import torch
from torch import nn


class Standardizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.float())
        self.register_buffer('std', std.float().clamp_min(1e-6))

    def forward(self, c):
        return (c - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class AffineEdit(nn.Module):
    """x' = x + W x + U a + b (zero-initialised: identity at step 0)."""
    def __init__(self, dim, n_actions):
        super().__init__()
        self.lin = nn.Linear(dim + n_actions, dim)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, a):
        return x + self.lin(torch.cat([x, a], 1))


class MLPEdit(nn.Module):
    """x' = x + f([x, a]), two hidden layers; last layer zero-initialised (identity at step 0)."""
    def __init__(self, dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + n_actions, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, a):
        return x + self.net(torch.cat([x, a], 1))


OPERATORS = {'affine': AffineEdit, 'mlp': MLPEdit}


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def fit_operator(kind, x_src, act, x_tgt, val, *, seed, steps, batch=2048, lr=1e-3, eval_every=200, device='cuda'):
    """Paired latent reconstruction (MSE in standardized space). Selection: lowest validation MSE.

    x_*: standardized float tensors on `device`; val = (x_src, act, x_tgt). Only the operator is optimized.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device='cpu').manual_seed(seed)
    op = OPERATORS[kind](x_src.shape[1], act.shape[1]).to(device)
    opt = torch.optim.Adam(op.parameters(), lr=lr)
    best = (float('inf'), 0, copy.deepcopy(op.state_dict()))
    history = []
    for step in range(1, steps + 1):
        idx = torch.randint(len(x_src), (batch,), generator=g).to(device)
        loss = (op(x_src[idx], act[idx]) - x_tgt[idx]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            v = evaluate_mse(op, *val)
            history.append({'step': step, 'train_batch_mse': float(loss.detach()), 'val_mse': v})
            if v < best[0]:
                best = (v, step, copy.deepcopy(op.state_dict()))
    op.load_state_dict(best[2])
    op.eval()
    for p in op.parameters():
        p.requires_grad_(False)
    return op, {'selected_step': best[1], 'selected_val_mse': best[0], 'history': history,
                'parameters': n_params(op), 'steps': steps, 'batch': batch, 'lr': lr,
                'training_pairs': len(x_src), 'selection': 'lowest validation-pair latent MSE'}


@torch.no_grad()
def evaluate_mse(op, x_src, act, x_tgt, batch=65536):
    total = 0.
    for i in range(0, len(x_src), batch):
        total += float((op(x_src[i:i+batch], act[i:i+batch]) - x_tgt[i:i+batch]).square().sum())
    return total / x_src.numel()


@torch.no_grad()
def apply_operator(op, x, act, batch=65536):
    return torch.cat([op(x[i:i+batch], act[i:i+batch]) for i in range(0, len(x), batch)])


class FactorReaders(nn.Module):
    """Held-out readers: one MLP trunk, one softmax head per raw factor (including unsupervised ones)."""
    def __init__(self, dim, cards, hidden=256):
        super().__init__()
        self.names = list(cards)
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(hidden, cards[f]) for f in self.names])

    def forward(self, x):
        h = self.trunk(x)
        return [head(h) for head in self.heads]

    @torch.no_grad()
    def predict(self, x, batch=65536):
        outs = [torch.cat(o) for o in zip(*[[l.argmax(1) for l in self(x[i:i+batch])] for i in range(0, len(x), batch)])]
        return dict(zip(self.names, outs))


def fit_readers(x, y, x_val, y_val, cards, *, seed=0, steps=3000, batch=4096, lr=1e-3, eval_every=250, device='cuda'):
    """Trained on original (unedited) images only; selected on validation-context images by mean accuracy."""
    torch.manual_seed(seed)
    g = torch.Generator(device='cpu').manual_seed(seed)
    model = FactorReaders(x.shape[1], cards).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    names = list(cards)
    best = (-1., 0, None)
    for step in range(1, steps + 1):
        idx = torch.randint(len(x), (batch,), generator=g).to(device)
        loss = sum(nn.functional.cross_entropy(l, y[f][idx]) for l, f in zip(model(x[idx]), names))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % eval_every == 0 or step == steps:
            model.eval()
            pred = model.predict(x_val)
            acc = sum(float((pred[f] == y_val[f]).float().mean()) for f in names) / len(names)
            model.train()
            if acc > best[0]:
                best = (acc, step, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {'selected_step': best[1], 'val_mean_accuracy': best[0], 'steps': steps,
                   'training_images': len(x), 'selection': 'highest mean validation-image accuracy'}
