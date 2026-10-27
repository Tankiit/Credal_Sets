"""Controlled 2-D scenes, not CLEVR. All edits preserve the base scene split."""
from dataclasses import dataclass, replace, asdict
import numpy as np
from PIL import Image, ImageDraw

COLORS = ((220, 45, 45), (45, 185, 70), (45, 90, 220))
D = 10  # two objects x (three colors + two shapes)

@dataclass(frozen=True)
class Scene:
    scene_id: str
    colors: tuple
    shapes: tuple
    offsets: tuple
    split: str

def concepts(s):
    y = np.zeros(D)
    for o in range(2):
        y[5*o+s.colors[o]] = 1
        y[5*o+3+s.shapes[o]] = 1
    return y

def render(s):
    im = Image.new('RGB', (64, 32), (245, 245, 245))
    draw = ImageDraw.Draw(im)
    for o in range(2):
        dx, dy = s.offsets[o]
        box = (o*32+8+dx, 8+dy, o*32+24+dx, 24+dy)
        fn = draw.rectangle if s.shapes[o] == 0 else draw.ellipse
        fn(box, fill=COLORS[s.colors[o]])
    return im

def edit(s, obj, color):
    colors = list(s.colors)
    colors[obj] = color
    return replace(s, colors=tuple(colors))

def action(obj, color):
    a = np.zeros(5)
    a[obj] = 1
    a[2+color] = 1
    return a

def heldout(s):
    # Red squares are absent from BOTH endpoints of all training/validation pairs.
    return any(c == 0 and sh == 0 for c, sh in zip(s.colors, s.shapes))

def scenes(n, split, seed):
    if n < 1:
        raise ValueError('n must be positive')
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = Scene(f'{split}-{len(out):06}', tuple(map(int, rng.integers(3, size=2))),
                  tuple(map(int, rng.integers(2, size=2))),
                  tuple(tuple(map(int, v)) for v in rng.integers(-4, 5, size=(2, 2))), split)
        if split in ('train', 'val') and heldout(s):
            continue
        out.append(s)
    return out

def pairs(base):
    out = []
    for s in base:
        for o in range(2):
            for c in range(3):
                if c == s.colors[o]:
                    continue
                t = edit(s, o, c)
                if s.split in ('train', 'val') and heldout(t):
                    continue
                out.append((s, t, o, c))
    return out

def features(ss):
    # A fixed spatial pixel encoder for a dependency-light plumbing pilot.
    # Replace this boundary with cached pretrained features for research runs.
    return np.stack([np.asarray(render(s).resize((16, 8)), dtype=float).ravel()/255 for s in ss])

def manifest(ss):
    return [asdict(s) for s in ss]
