"""Turn a real image with located text into a degraded, photograph-like stimulus with
per-character visibility.

Input: a PIL image and `lines`, a list of {"raw": str, "boxes": [quad | None per character]}
where a quad is four (x, y) image points (tl, tr, br, bl) and None marks a space.
Optionally the image is first photographed as a flat sheet lying on a surface (documents);
packaging and signage photographs are used as they are. Framing (none / crop by a frame edge
/ finger or glare occlusion) and quality (clean / blur / severe) are applied as in the
synthetic study, and every character gets visibility 2 (visible), 1 (partial) or 0 (hidden).
"""
import math
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from textnorm import tokens_raw

SKIN = [(224, 172, 138), (198, 134, 96), (241, 194, 160), (141, 85, 60), (234, 190, 160)]


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


def grid(quad, n=3):
    """n x n sample points inside a quad (bilinear), slightly inset."""
    q = np.asarray(quad, float)
    pts = []
    for a in np.linspace(0.15, 0.85, n):
        top = q[0] + (q[1] - q[0]) * a
        bot = q[3] + (q[2] - q[3]) * a
        for b in np.linspace(0.15, 0.85, n):
            pts.append(top + (bot - top) * b)
    return np.array(pts)


def photograph(img, lines, quality, framing, rng: random.Random, sheet=False):
    nrng = np.random.default_rng(rng.randrange(2 ** 32))
    W, Hh = img.size
    info = {}
    if sheet:  # lay the page on a textured surface with a hand-held perspective
        persp = {"clean": 0.015, "blur": 0.03, "severe": 0.05}[quality]
        rot = math.radians(rng.uniform(-1, 1) * {"clean": 2, "blur": 4, "severe": 6}[quality])
        SW, SH = int(W * 1.3), int(Hh * 1.3)
        cx, cy = SW / 2, SH / 2
        src = [(0, 0), (W, 0), (W, Hh), (0, Hh)]
        dst = []
        for px, py in src:
            dx, dy = px - W / 2, py - Hh / 2
            dst.append((cx + dx * math.cos(rot) - dy * math.sin(rot) + rng.uniform(-persp, persp) * W,
                        cy + dx * math.sin(rot) + dy * math.cos(rot) + rng.uniform(-persp, persp) * W))
        H = homography(src, dst)
        Hinv = np.linalg.inv(H)
        base = np.array(rng.choice([(120, 105, 90), (70, 75, 85), (160, 150, 130), (95, 80, 65), (140, 140, 145)]), float)
        tex = np.repeat(nrng.normal(0, 14, (SH // 40 + 1, SW // 40 + 1, 1)), 3, axis=2)
        tex = np.array(Image.fromarray(np.clip(tex + 128, 0, 255).astype(np.uint8)).resize((SW, SH), Image.BICUBIC), float) - 128
        grad = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.9, 1.15), SW)[None, :, None]
        scene = Image.fromarray(np.clip((base + tex) * grad, 0, 255).astype(np.uint8))
        coeff = tuple((Hinv / Hinv[2, 2]).flatten()[:8])
        scene.paste(img.transform((SW, SH), Image.PERSPECTIVE, coeff, Image.BICUBIC), (0, 0),
                    Image.new("L", (W, Hh), 255).transform((SW, SH), Image.PERSPECTIVE, coeff, Image.BILINEAR))
        # uneven lighting across the sheet
        shade = np.linspace(rng.uniform(0.8, 1.0), 1.0, SH)[:, None, None]
        scene = Image.fromarray(np.clip(np.array(scene, float) * shade, 0, 255).astype(np.uint8))
    else:
        H = np.eye(3)
        SW, SH = W, Hh
        scene = img.copy()

    # context lines (text visible in the crop but outside the selected block) do not steer framing
    core = [ln for ln in lines if not ln.get("context")]
    allq = np.array([p for ln in core for b in ln["boxes"] if b is not None for p in b], float)
    tq = apply_h(H, allq)
    bx0, by0 = tq.min(0)
    bx1, by1 = tq.max(0)
    if sheet:
        pc = apply_h(H, [(0, 0), (W, 0), (W, Hh), (0, Hh)])
        m = rng.uniform(0.03, 0.08) * (pc[:, 0].max() - pc[:, 0].min())
        fx0, fy0 = max(0, pc[:, 0].min() - m), max(0, pc[:, 1].min() - m)
        fx1, fy1 = min(SW, pc[:, 0].max() + m), min(SH, pc[:, 1].max() + m)
    else:
        fx0, fy0, fx1, fy1 = 0, 0, SW, SH
    occ = np.zeros((SH, SW), bool)
    if framing == "crop":
        side = rng.choice(["left", "right", "right", "bottom"])
        frac = rng.uniform(0.14, 0.30)
        if side == "right":
            fx1 = bx1 - frac * (bx1 - bx0)
        elif side == "left":
            fx0 = bx0 + frac * (bx1 - bx0)
        else:
            fy1 = by1 - rng.uniform(0.18, 0.35) * (by1 - by0)
        info = {"side": side, "frac": round(frac, 3)}
    elif framing == "occlusion":
        kind = rng.choice(["finger", "glare"])
        boxes = [b for ln in core for b in ln["boxes"] if b is not None]
        tx, ty = apply_h(H, [np.mean(np.asarray(rng.choice(boxes), float), 0)])[0]
        yy, xx = np.mgrid[0:SH, 0:SW]
        S = max(SW, SH)
        if kind == "finger":
            r = rng.uniform(0.03, 0.045) * S
            ang = math.radians(rng.uniform(-25, 25))
            ux, uy = math.sin(ang), math.cos(ang)
            t = (xx - tx) * ux + (yy - ty) * uy
            dperp = np.abs(-(xx - tx) * uy + (yy - ty) * ux)
            fm = ((t >= 0) & (dperp <= r)) | ((xx - tx) ** 2 + (yy - ty) ** 2 <= r ** 2)
            arr = np.array(scene, float)
            shade = 0.85 + 0.15 * np.clip(1 - dperp / r, 0, 1)
            arr[fm] = np.array(rng.choice(SKIN), float)[None, :] * shade[fm][:, None]
            scene = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            occ = fm
        else:
            rx, ry = rng.uniform(0.06, 0.10) * S, rng.uniform(0.03, 0.055) * S
            alpha = np.clip(np.exp(-(((xx - tx) / rx) ** 2 + ((yy - ty) / ry) ** 2) * 1.2) * 1.25, 0, 1)
            arr = np.array(scene, float)
            arr = arr * (1 - alpha[..., None]) + 255 * alpha[..., None]
            scene = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            occ = alpha > 0.85
        info = {"occluder": kind}
    fx0, fy0, fx1, fy1 = int(fx0), int(fy0), int(fx1), int(fy1)
    out = scene.crop((fx0, fy0, fx1, fy1))

    for ln in lines:
        vis = []
        for b in ln["boxes"]:
            if b is None:
                vis.append(2)
                continue
            src_pts = grid(b)
            inside_src = [0 <= x < W and 0 <= y < Hh for x, y in src_pts]  # glyph parts cut off the source image
            pts = apply_h(H, src_pts)
            ok = sum(1 for (px, py), ins in zip(pts, inside_src) if ins and fx0 <= px < fx1 and fy0 <= py < fy1
                     and not occ[int(min(max(py, 0), SH - 1)), int(min(max(px, 0), SW - 1))])
            vis.append(2 if ok >= 8 else (0 if ok <= 2 else 1))
        ln["vis"] = vis

    s = 1024 / max(out.size)
    for ln in lines:  # line bounding box in final-image pixels (for figures; no randomness used)
        pts = [p for b in ln["boxes"] if b is not None for p in b]
        if pts:
            t = (apply_h(H, pts) - np.array([fx0, fy0])) * s
            ln["bbox"] = [round(float(v), 1) for v in (*t.min(0), *t.max(0))]
    out = out.resize((max(1, int(out.size[0] * s)), max(1, int(out.size[1] * s))), Image.LANCZOS)
    q = {}
    if quality == "clean":
        noise, jpg = 2, 90
    elif quality == "blur":
        sig = rng.uniform(1.6, 2.4)
        out = ImageEnhance.Contrast(out.filter(ImageFilter.GaussianBlur(sig))).enhance(rng.uniform(0.8, 0.95))
        noise, jpg, q = 4, 75, {"sigma": round(sig, 2)}
    else:
        sig = rng.uniform(2.8, 3.8)
        out = ImageEnhance.Contrast(out.filter(ImageFilter.GaussianBlur(sig))).enhance(rng.uniform(0.5, 0.7))
        out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.65, 0.9))
        noise, jpg, q = 8, 45, {"sigma": round(sig, 2)}
    arr = np.array(out, float) + nrng.normal(0, noise, (out.size[1], out.size[0], 1))
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    info |= q | {"jpeg": jpg, "scale": round(s, 4)}
    return out, jpg, info


def gt_tokens(lines, lexicon):
    """Token ground truth. Category: digits -> num; in the Ukrainian word list -> word;
    otherwise -> name (brands, surnames, place names, coinages)."""
    out = []
    for li, ln in enumerate(lines):
        for tok, idx in tokens_raw(ln["raw"]):
            v = [ln["vis"][i] for i in idx]
            cat = "num" if tok.isdigit() else ("word" if tok in lexicon else "name")
            state = "visible" if all(x == 2 for x in v) else ("hidden" if all(x == 0 for x in v) else "partial")
            rec = {"tok": tok, "chars": "".join(ln["raw"][i].lower() for i in idx), "line": li, "cat": cat,
                   "state": state, "vis": v, "role": ln.get("role", "body")}
            if ln.get("conf"):
                rec["conf"] = round(min(ln["conf"][i] for i in idx), 3)
                rec["lowconf"] = rec["conf"] < 0.7  # ground truth itself uncertain
            out.append(rec)
    return out


# ------------------------------------------------------------------ sources

BREAK_SPACE = {"SPACE", "SURE_SPACE"}
BREAK_LINE = {"EOL_SURE_SPACE", "LINE_BREAK"}


def lines_from_gcv(fta, scale=1.0):
    """Lines (raw text + per-character quads) from a Google Cloud Vision fullTextAnnotation."""
    lines, raw, boxes, conf = [], [], [], []

    def flush():
        nonlocal raw, boxes, conf
        while raw and raw[-1] == " ":
            raw.pop(); boxes.pop(); conf.pop()
        if raw:
            lines.append({"raw": "".join(raw), "boxes": boxes, "conf": conf})
        raw, boxes, conf = [], [], []

    for page in fta.get("pages", []):
        for block in page.get("blocks", []):
            for para in block.get("paragraphs", []):
                for word in para.get("words", []):
                    for sym in word.get("symbols", []):
                        t = sym.get("text", "")
                        v = sym.get("boundingBox", {}).get("vertices", [])
                        if len(v) != 4 or len(t) != 1:
                            continue
                        raw.append(t)
                        boxes.append([(p.get("x", 0) * scale, p.get("y", 0) * scale) for p in v])
                        conf.append(sym.get("confidence", word.get("confidence", 1.0)))
                        br = sym.get("property", {}).get("detectedBreak", {}).get("type")
                        if br in BREAK_SPACE:
                            raw.append(" "); boxes.append(None); conf.append(1.0)
                        elif br in BREAK_LINE or br == "HYPHEN":
                            flush()
                flush()
    return lines


def lines_from_pdf(pdf, page, dpi=130, max_words=170, start_line=0, with_context=False):
    """Render one PDF page and read its text layer with word boxes (poppler).
    Character boxes split each word box in proportion to character count. If the page has
    more than max_words words, a block of consecutive lines starting at start_line is cut out.
    Returns (image, lines) or (None, None) if the page has no usable text layer."""
    import html, re, subprocess, tempfile, os
    xml = subprocess.run(["pdftotext", "-bbox-layout", "-f", str(page), "-l", str(page), pdf, "-"],
                         capture_output=True, text=True, timeout=120).stdout
    k = dpi / 72.0
    L = []
    for lm in re.finditer(r"<line[^>]*>(.*?)</line>", xml, re.S):
        words = [(float(a), float(b), float(c), float(d), html.unescape(t))
                 for a, b, c, d, t in re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', lm.group(1))]
        words = [w for w in words if w[4].strip()]
        if words:
            L.append(words)
    L.sort(key=lambda ws: (round(ws[0][1] / 4), ws[0][0]))
    if not L:
        return None, None
    # choose a block of consecutive lines with at most max_words words
    sel, n = [], 0
    for ws in L[start_line:]:
        if n + len(ws) > max_words and sel:
            break
        sel.append(ws)
        n += len(ws)
    x0 = min(w[0] for ws in sel for w in ws) - 25
    y0 = min(w[1] for ws in sel for w in ws) - 25
    x1 = max(w[2] for ws in sel for w in ws) + 25
    y1 = max(w[3] for ws in sel for w in ws) + 25
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", pdf, os.path.join(td, "p")],
                       check=True, timeout=120)
        f = [x for x in os.listdir(td) if x.endswith(".png")][0]
        img = Image.open(os.path.join(td, f)).convert("RGB")
    X0, Y0 = max(0, int(x0 * k)), max(0, int(y0 * k))
    img = img.crop((X0, Y0, min(img.size[0], int(x1 * k)), min(img.size[1], int(y1 * k))))
    def mk(ws):
        raw, boxes = [], []
        for j, (a, b, c, d, t) in enumerate(ws):
            if j:
                raw.append(" "); boxes.append(None)
            wpx = (c - a) / len(t)
            for i, ch in enumerate(t):
                xa, xb = (a + i * wpx) * k - X0, (a + (i + 1) * wpx) * k - X0
                raw.append(ch)
                boxes.append([(xa, b * k - Y0), (xb, b * k - Y0), (xb, d * k - Y0), (xa, d * k - Y0)])
        return {"raw": "".join(raw), "boxes": boxes}

    if with_context:
        # other lines of the page that reach into the cropped region (often cut by the crop edge)
        ctx = []
        for ws in L:
            if ws in sel:
                continue
            ws2 = [w for w in ws if w[2] * k > X0 and w[0] * k < X0 + img.size[0] and w[3] * k > Y0 and w[1] * k < Y0 + img.size[1]]
            if ws2:
                ctx.append(dict(mk(ws2), context=True))
    lines = []
    for ws in sel:
        raw, boxes = [], []
        for j, (a, b, c, d, t) in enumerate(ws):
            if j:
                raw.append(" "); boxes.append(None)
            wpx = (c - a) / len(t)
            for i, ch in enumerate(t):
                xa, xb = (a + i * wpx) * k - X0, (a + (i + 1) * wpx) * k - X0
                ya, yb = b * k - Y0, d * k - Y0
                raw.append(ch)
                boxes.append([(xa, ya), (xb, ya), (xb, yb), (xa, yb)])
        lines.append({"raw": "".join(raw), "boxes": boxes})
    if with_context:
        return img, lines, ctx
    return img, lines
