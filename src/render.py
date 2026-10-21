"""Render a text layout onto a sign / package label / document panel, place it in a
photographed-looking scene with a homography, apply framing (crop / occlusion) and
image-quality degradations, and compute per-character visibility in the final frame.

Visibility codes per character: 2 = fully visible, 1 = partially visible, 0 = hidden
(outside the frame or under the occluder).
"""
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance

from textnorm import tokens_raw

FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "assets", "fonts")
LIB = "/usr/share/fonts/truetype/liberation"
SANS = [("PT_Sans-Web-Regular", "PT_Sans-Web-Bold"), ("Ubuntu-Regular", "Ubuntu-Bold"),
        ("FiraSans-Regular", "FiraSans-Bold"), ("NotoSans", "NotoSans"), ("Rubik", "Rubik"),
        ("Montserrat", "Montserrat"), ("RobotoCondensed", "RobotoCondensed"),
        (LIB + "/LiberationSans-Regular", LIB + "/LiberationSans-Bold")]
SERIF = [("PT_Serif-Web-Regular", "PT_Serif-Web-Bold"), ("NotoSerif", "NotoSerif"),
         (LIB + "/LiberationSerif-Regular", LIB + "/LiberationSerif-Bold")]
MONO = [("PTM55FT", "PTM55FT"), (LIB + "/LiberationMono-Regular", LIB + "/LiberationMono-Bold")]
DISPLAY = ["Lobster-Regular", "MarckScript-Regular", "Caveat", "Comfortaa", "RussoOne-Regular", "Oswald"]
ROLE_SIZE = {"title": 104, "large": 76, "body": 52, "small": 40}

SIGN_COLOURS = [((20, 70, 40), (245, 245, 235)), ((20, 50, 110), (250, 250, 250)),
                ((245, 245, 240), (25, 25, 25)), ((250, 205, 40), (20, 20, 20)),
                ((150, 25, 30), (250, 245, 230)), ((90, 55, 35), (245, 225, 180)),
                ((35, 35, 38), (240, 200, 90))]
PACK_COLOURS = [((252, 252, 248), (30, 30, 90)), ((250, 240, 215), (90, 30, 20)),
                ((215, 235, 250), (10, 40, 90)), ((230, 245, 220), (20, 70, 20)),
                ((255, 230, 235), (120, 20, 50)), ((40, 40, 45), (250, 250, 250))]
DOC_COLOURS = [((250, 250, 247), (20, 20, 20)), ((245, 242, 232), (30, 30, 60)),
               ((255, 255, 255), (10, 10, 10)), ((238, 240, 245), (25, 25, 25))]
SKIN = [(224, 172, 138), (198, 134, 96), (241, 194, 160), (141, 85, 60), (234, 190, 160)]


def _font(name, size):
    path = name + ".ttf" if name.startswith("/") else os.path.join(FONT_DIR, name + ".ttf")
    return ImageFont.truetype(path, size)


def _wrap(text, font, maxw):
    """Split text into (start, end) spans that fit maxw, breaking at spaces."""
    spans, start = [], 0
    while start < len(text):
        end = len(text)
        if font.getlength(text[start:end]) > maxw:
            brk = None
            for k in range(start + 1, len(text)):
                if text[k] == " " and font.getlength(text[start:k]) <= maxw:
                    brk = k
            end = brk if brk else len(text)
        spans.append((start, end))
        start = end + 1 if end < len(text) and text[end] == " " else end
    return spans


def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(A, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def render(spec, rng: random.Random):
    genre, lines, quality, framing = spec["genre"], spec["lines"], spec["quality"], spec["framing"]
    nrng = np.random.default_rng(rng.randrange(2 ** 32))

    # ---- fonts and colours
    if genre == "document":
        fam = rng.choice(MONO) if spec["kind"] == "receipt" else rng.choice(SERIF + SANS[:3])
        bg, fg = rng.choice(DOC_COLOURS)
        W, align = 1250, "left"
    elif genre == "packaging":
        fam = rng.choice(SANS + SERIF[:1])
        bg, fg = rng.choice(PACK_COLOURS)
        W, align = 1150, rng.choice(["center", "left"])
    else:
        fam = rng.choice(SANS)
        bg, fg = rng.choice(SIGN_COLOURS)
        W, align = 1400, "center"
    title_font = rng.choice(DISPLAY) if genre != "document" and rng.random() < 0.35 else fam[1]
    spec["font"] = {"body": os.path.basename(fam[0]), "title": os.path.basename(title_font)}

    # ---- layout
    margin = 70
    y = margin
    placed = []  # (line_idx, start, end, font, x, y, ascent, descent)
    for li, ln in enumerate(lines):
        role, raw = ln["role"], ln["raw"]
        f = _font(title_font if role == "title" else (fam[1] if role == "large" else fam[0]), ROLE_SIZE[role])
        asc, desc = f.getmetrics()
        for s, e in _wrap(raw, f, W - 2 * margin):
            w = f.getlength(raw[s:e])
            x = margin if align == "left" else (W - w) / 2
            placed.append((li, s, e, f, x, y, asc, desc))
            y += int((asc + desc) * 1.12)
        y += int(ROLE_SIZE[role] * 0.25)
    Hh = y + margin - int(ROLE_SIZE["small"] * 0.25)
    panel = Image.new("RGB", (W, Hh), bg)
    d = ImageDraw.Draw(panel)
    if genre == "signage" and rng.random() < 0.6:
        d.rectangle([14, 14, W - 15, Hh - 15], outline=fg, width=6)
    if genre == "document":
        d.line([margin, placed[0][5] + placed[0][6] + placed[0][7] + 8, W - margin,
                placed[0][5] + placed[0][6] + placed[0][7] + 8], fill=fg, width=2)
    char_boxes = {li: [None] * len(ln["raw"]) for li, ln in enumerate(lines)}
    for li, s, e, f, x, yy, asc, desc in placed:
        raw = lines[li]["raw"]
        d.text((x, yy), raw[s:e], font=f, fill=fg)
        for k in range(s, e):
            x0 = x + f.getlength(raw[s:k])
            x1 = x + f.getlength(raw[s:k + 1])
            char_boxes[li][k] = (x0, yy + asc * 0.25, x1, yy + asc + desc * 0.3)

    # ---- scene + homography
    persp = {"clean": 0.015, "blur": 0.035, "severe": 0.06}[quality]
    rot = math.radians(rng.uniform(-1, 1) * {"clean": 2, "blur": 4, "severe": 7}[quality])
    SW, SH = int(W * 1.45), int(Hh * 1.45 + 250)
    cx, cy = SW / 2 + rng.uniform(-0.05, 0.05) * SW, SH / 2 + rng.uniform(-0.05, 0.05) * SH
    src = [(0, 0), (W, 0), (W, Hh), (0, Hh)]
    dst = []
    for px, py in src:
        dx, dy = px - W / 2, py - Hh / 2
        jx, jy = rng.uniform(-persp, persp) * W, rng.uniform(-persp, persp) * W
        dst.append((cx + dx * math.cos(rot) - dy * math.sin(rot) + jx,
                    cy + dx * math.sin(rot) + dy * math.cos(rot) + jy))
    H = homography(src, dst)
    Hinv = np.linalg.inv(H)
    base = np.array(rng.choice([(120, 110, 100), (60, 70, 80), (170, 160, 140), (90, 100, 70),
                                (140, 140, 150), (50, 45, 40)]), float)
    tex = np.repeat(nrng.normal(0, 14, (SH // 40 + 1, SW // 40 + 1, 1)), 3, axis=2) + nrng.normal(0, 4, (SH // 40 + 1, SW // 40 + 1, 3))
    tex = np.array(Image.fromarray(np.clip(tex + 128, 0, 255).astype(np.uint8)).resize((SW, SH), Image.BICUBIC), float) - 128
    grad = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.9, 1.2), SW)[None, :, None]
    scene = Image.fromarray(np.clip((base + tex) * grad, 0, 255).astype(np.uint8))
    warped = panel.transform((SW, SH), Image.PERSPECTIVE, tuple((Hinv / Hinv[2, 2]).flatten()[:8]), Image.BICUBIC)
    mask = Image.new("L", (W, Hh), 255).transform((SW, SH), Image.PERSPECTIVE, tuple((Hinv / Hinv[2, 2]).flatten()[:8]), Image.BILINEAR)
    scene.paste(warped, (0, 0), mask)

    # ---- frame (crop rectangle in scene coordinates)
    corners = apply_h(H, src)
    bx0, by0 = corners.min(0)
    bx1, by1 = corners.max(0)
    pw = bx1 - bx0
    m = rng.uniform(0.04, 0.10)
    fx0, fy0 = max(0, bx0 - m * pw), max(0, by0 - m * pw)
    fx1, fy1 = min(SW, bx1 + m * pw), min(SH, by1 + m * pw)
    occ = np.zeros((SH, SW), bool)
    framing_info = {}
    if framing == "crop":
        side = rng.choice(["left", "right", "right", "bottom"])
        frac = rng.uniform(0.14, 0.30)
        if side == "right":
            fx1 = bx1 - frac * pw
        elif side == "left":
            fx0 = bx0 + frac * pw
        else:
            fy1 = by1 - rng.uniform(0.18, 0.35) * (by1 - by0)
        framing_info = {"side": side, "frac": round(frac, 3)}
    elif framing == "occlusion":
        kind = rng.choice(["finger", "glare"])
        # target: a random character centre of a text line (not the first line half the time)
        li = rng.choice([k for k in char_boxes if any(char_boxes[k])])
        ks = [k for k, b in enumerate(char_boxes[li]) if b]
        bxx = char_boxes[li][rng.choice(ks)]
        tx, ty = apply_h(H, [((bxx[0] + bxx[2]) / 2, (bxx[1] + bxx[3]) / 2)])[0]
        yy, xx = np.mgrid[0:SH, 0:SW]
        if kind == "finger":
            r = rng.uniform(55, 85)
            # finger enters from the bottom edge, up to the target point, slightly tilted
            ang = math.radians(rng.uniform(-25, 25))
            ux, uy = math.sin(ang), math.cos(ang)
            t = (xx - tx) * ux + (yy - ty) * uy  # along finger axis (downwards from tip)
            dperp = np.abs(-(xx - tx) * uy + (yy - ty) * ux)
            fm = (t >= 0) & (dperp <= r) | ((xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2)
            col = np.array(rng.choice(SKIN), float)
            arr = np.array(scene, float)
            shade = 0.85 + 0.15 * np.clip(1 - dperp / r, 0, 1)
            arr[fm] = (col[None, :] * shade[fm][:, None])
            scene = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            occ = fm
        else:
            rx, ry = rng.uniform(110, 200), rng.uniform(60, 110)
            g = np.exp(-(((xx - tx) / rx) ** 2 + ((yy - ty) / ry) ** 2) * 1.2)
            alpha = np.clip(g * 1.25, 0, 1)
            arr = np.array(scene, float)
            arr = arr * (1 - alpha[..., None]) + 255 * alpha[..., None]
            scene = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            occ = alpha > 0.85  # text under weaker glare stays legible
        framing_info = {"occluder": kind}
    spec["framing_info"] = framing_info
    fx0, fy0, fx1, fy1 = int(fx0), int(fy0), int(fx1), int(fy1)
    img = scene.crop((fx0, fy0, fx1, fy1))

    # ---- per-character visibility
    for li, ln in enumerate(lines):
        vis = []
        for b in char_boxes[li]:
            if b is None:
                vis.append(2)  # only wrapped-away spaces have no box
                continue
            gx = np.linspace(b[0] + 1, b[2] - 1, 3)
            gy = np.linspace(b[1], b[3], 3)
            pts = apply_h(H, [(a, c) for a in gx for c in gy])
            ok = 0
            for px, py in pts:
                inside = fx0 <= px < fx1 and fy0 <= py < fy1
                if inside and not occ[int(min(max(py, 0), SH - 1)), int(min(max(px, 0), SW - 1))]:
                    ok += 1
            vis.append(2 if ok >= 8 else (0 if ok <= 2 else 1))
        ln["vis"] = vis

    # ---- resize + quality degradation
    s = 1024 / max(img.size)
    img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))), Image.LANCZOS)
    q = {}
    if quality == "clean":
        noise, jpg = 2, 90
    elif quality == "blur":
        sig = rng.uniform(1.6, 2.4)
        img = img.filter(ImageFilter.GaussianBlur(sig))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 0.95))
        noise, jpg = 4, 75
        q = {"sigma": round(sig, 2)}
    else:
        sig = rng.uniform(2.8, 3.8)
        img = img.filter(ImageFilter.GaussianBlur(sig))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.5, 0.7))
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.65, 0.9))
        noise, jpg = 8, 45
        q = {"sigma": round(sig, 2)}
    arr = np.array(img, float) + nrng.normal(0, noise, (img.size[1], img.size[0], 1))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    spec["quality_info"] = q | {"jpeg": jpg, "scale": round(s, 4)}
    return img, jpg


def gt_tokens(lines):
    """Token-level ground truth with category and visibility."""
    out = []
    for li, ln in enumerate(lines):
        for tok, idx in tokens_raw(ln["raw"]):
            v = [ln["vis"][i] for i in idx]
            cats = [ln["cats"][i] for i in idx]
            cat = "num" if tok.isdigit() else max(set(cats), key=cats.count)
            state = "visible" if all(x == 2 for x in v) else ("hidden" if all(x == 0 for x in v) else "partial")
            chars = "".join(ln["raw"][i].lower() for i in idx)
            out.append({"tok": tok, "chars": chars, "line": li, "cat": cat, "state": state,
                        "vis": v, "role": ln["role"]})
    return out
