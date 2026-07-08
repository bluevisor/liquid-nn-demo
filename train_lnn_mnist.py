"""
Train a Liquid Time-Constant (LTC) network to recognize hand-drawn digits.

Framing: the LTC is a sequence model, so we feed each 28x28 image as a
sequence of 28 ROWS (one row = one timestep, 28 features). The network
integrates its ODE down the image and the final hidden state is classified.
This is the classic "row-sequential MNIST" benchmark — a natural fit for a
liquid network, unlike a CNN which sees the whole image at once.

Exports lnn_weights.json for the in-browser recognizer (digit_recognizer.html).
"""
import json, time, math
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
H = 256            # hidden neurons (browser reads this from the JSON)
STEPS = 4          # ODE solver sub-steps per row (browser reads this from the JSON)
HSTEP = 1.0 / STEPS
MEAN, STD = 0.1307, 0.3081
torch.manual_seed(0)


class LTC(nn.Module):
    def __init__(self, in_dim=28, hidden=H, n_class=10):
        super().__init__()
        self.H = hidden
        self.Win = nn.Linear(in_dim, hidden)      # input -> gate
        self.Wrec = nn.Linear(hidden, hidden)     # recurrent -> gate
        # diverse base timescales: some neurons are fast reflexes, some slow memory
        self.log_tau = nn.Parameter(torch.linspace(math.log(0.5), math.log(4.0), hidden))
        self.A = nn.Parameter(torch.randn(hidden) * 0.5)
        self.out = nn.Linear(hidden, n_class)

    def solver(self, h, u):
        inv_tau = torch.exp(-self.log_tau)
        gu = self.Win(u)
        for _ in range(STEPS):
            f = torch.sigmoid(gu + self.Wrec(h))
            h = (h + HSTEP * f * self.A) / (1.0 + HSTEP * (inv_tau + f))
        return h

    def forward(self, x):                          # x: (B,28,28) rows normalized
        h = torch.zeros(x.size(0), self.H, device=x.device)
        for t in range(28):
            h = self.solver(h, x[:, t, :])
        return self.out(h)


def load_tensors():
    tf = transforms.ToTensor()
    tr = datasets.MNIST("./_mnist", train=True, download=True, transform=tf)
    te = datasets.MNIST("./_mnist", train=False, download=True, transform=tf)
    def stack(ds):
        X = torch.stack([ds[i][0][0] for i in range(len(ds))])  # (N,28,28) in [0,1]
        y = torch.tensor([ds[i][1] for i in range(len(ds))])
        return X, y
    return stack(tr), stack(te)


def stroke_jitter(x):
    """Randomly thicken (dilate) or thin (erode) strokes per sample, so the net
    tolerates the wide range of brush widths a person draws with — MNIST strokes
    are thin and uniform; mouse strokes are not. x: (B,1,28,28) in [0,1]."""
    dil = F.max_pool2d(x, 3, 1, 1)            # 3×3 dilation: thicker strokes
    ero = -F.max_pool2d(-x, 3, 1, 1)          # 3×3 erosion:  thinner strokes
    r = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    out = x.clone()
    out = torch.where(r < 0.30, dil, out)     # 30% thicken
    out = torch.where(r > 0.75, ero, out)     # 25% thin, 45% unchanged
    return out


def _gauss1d(sigma, device):
    r = max(1, int(3 * sigma))
    xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2 * sigma * sigma))
    return k / k.sum(), r


def _gblur(x, sigma):                      # separable Gaussian blur, x: (B,1,H,W)
    k, r = _gauss1d(sigma, x.device)
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x


def elastic(x01, alpha, sigma=4.0):
    """Non-rigid stroke warp (Simard et al.) — smoothed random displacement field.
    This is the classic augmentation that makes MNIST models tolerate the wobble
    and stylistic variation of real handwriting. x01 in [0,1], (B,28,28)."""
    B, dev = x01.size(0), x01.device
    dx = _gblur(torch.rand(B, 1, 28, 28, device=dev) * 2 - 1, sigma)
    dy = _gblur(torch.rand(B, 1, 28, 28, device=dev) * 2 - 1, sigma)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, 28, device=dev),
                            torch.linspace(-1, 1, 28, device=dev), indexing="ij")
    grid = torch.stack([xs, ys], -1).unsqueeze(0).expand(B, -1, -1, -1).clone()
    s = (2.0 / 28) * alpha                 # px displacement → normalized coords
    grid[..., 0] += dx.squeeze(1) * s
    grid[..., 1] += dy.squeeze(1) * s
    out = F.grid_sample(x01.unsqueeze(1), grid, align_corners=False, padding_mode="zeros")
    return out.squeeze(1)


def augment(x01):
    """Batched affine (rotate/scale/translate/shear) + elastic warp + stroke-width
    jitter on device — teaches the net to tolerate the wobble and stylistic
    variety of a mouse-drawn digit. x01 in [0,1], (B,28,28)."""
    B, dev = x01.size(0), x01.device
    ang = (torch.rand(B, device=dev) - 0.5) * (2 * 15 * math.pi / 180)  # ±15°
    scale = 0.75 + torch.rand(B, device=dev) * 0.50                     # 0.75–1.25
    tx = (torch.rand(B, device=dev) - 0.5) * 0.24                       # ±12%
    ty = (torch.rand(B, device=dev) - 0.5) * 0.24
    shx = (torch.rand(B, device=dev) - 0.5) * 0.40                      # ±0.20 shear
    shy = (torch.rand(B, device=dev) - 0.5) * 0.40
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin + shx, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin + shy, cos, ty
    grid = F.affine_grid(theta, (B, 1, 28, 28), align_corners=False)
    out = F.grid_sample(x01.unsqueeze(1), grid, align_corners=False, padding_mode="zeros").squeeze(1)
    out = elastic(out, alpha=4.0 + torch.rand(1, device=dev).item() * 5.0)  # α 4–9
    out = stroke_jitter(out.unsqueeze(1)).squeeze(1)
    return out


def main():
    print(f"device: {DEV}")
    (Xtr, ytr), (Xte, yte) = load_tensors()
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    Xte, yte = ((Xte.to(DEV) - MEAN) / STD), yte.to(DEV)
    print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}")

    model = LTC().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    n = Xtr.size(0); BS = 300
    EPOCHS = 70; best = 0.0
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, BS):
            idx = perm[i:i + BS]
            xb = (augment(Xtr[idx]) - MEAN) / STD
            logits = model(xb)
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * idx.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            correct = 0
            for i in range(0, Xte.size(0), 1000):
                pred = model(Xte[i:i + 1000]).argmax(1)
                correct += (pred == yte[i:i + 1000]).sum().item()
        acc = correct / Xte.size(0)
        print(f"epoch {ep}/{EPOCHS}  loss {tot/n:.4f}  test_acc {acc:.4f}  "
              f"({time.time()-t0:.1f}s)")
        if acc > best:
            best = acc; export(model)
    print(f"best test accuracy: {best:.4f}  -> lnn_weights.json")


def export(model):
    m = model.cpu() if next(model.parameters()).is_cuda else model
    sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    def flat(t): return [round(float(x), 6) for x in t.flatten().tolist()]
    w = {
        "H": H, "steps": STEPS, "hstep": HSTEP, "mean": MEAN, "std": STD,
        "Win": flat(sd["Win.weight"]),   "Win_b": flat(sd["Win.bias"]),   # (H,28),(H)
        "Wrec": flat(sd["Wrec.weight"]), "Wrec_b": flat(sd["Wrec.bias"]), # (H,H),(H)
        "log_tau": flat(sd["log_tau"]),  "A": flat(sd["A"]),              # (H),(H)
        "Wout": flat(sd["out.weight"]),  "Wout_b": flat(sd["out.bias"]),  # (10,H),(10)
    }
    with open("lnn_weights.json", "w") as f:
        json.dump(w, f)
    model.to(DEV)


if __name__ == "__main__":
    main()
