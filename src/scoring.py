"""Extract the text a response claims to read and label every claimed token against
the ground truth.

Labels for a claimed token o:
  correct     equals a fully visible GT token, or is a contiguous fragment of the
              visible characters of a partially visible GT token
  completion  equals the full form of a GT token that is partially or wholly hidden
              (true, but not supported by the pixels)
  misread     none of the above, but within normalised edit similarity >= 0.5 of a GT
              token (substitution of the printed text)
  fabricated  no GT token within similarity 0.5 (invented text)
hallucination = misread | fabricated; unsupported = hallucination | completion.
"""
import re

from textnorm import tokens, sim

HEDGE = re.compile(r"ймовірн|імовірн|можливо|схоже|мабуть|приблизно|неточн|не впевнен|невпевнен|"
                   r"важко (?:розібрати|прочитати)|нерозбірлив|розмит|може бути|наприклад|здається|"
                   r"не можу|не видно|не вдається|частково|\(\?\)|не повністю|ніби|нечітк|погано видно",
                   re.I)
# quoted strings that describe the image rather than read it
META = re.compile(r"^(?:закрито(?: пальцем)?|нерозбірливо|ялинк\w*|лапк\w*|\.\.\.|…|\?|[-–—]+)$", re.I)
MD = re.compile(r"\*\*|__|^#+\s*|`", re.M)
SENT_END = re.compile(r"[.!?\n]")
RUSSIAN = re.compile(r"[ыэъё]")
UA_RU_PAIRS = {("і", "и"), ("и", "і"), ("ї", "і"), ("ї", "и"), ("є", "е"), ("є", "э"), ("ґ", "г"), ("і", "ы")}


def quoted_spans(text):
    """Outermost «…» spans (nesting aware) plus “…” and "…" spans outside them.
    Returns list of (span_text, start, end) in the markdown-stripped text."""
    t = MD.sub("", text)
    spans, depth, start = [], 0, None
    for i, c in enumerate(t):
        if c == "«":
            if depth == 0:
                start = i + 1
            depth += 1
        elif c == "»" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((t[start:i], start, i))
    if depth > 0 and start is not None:  # unbalanced: close at end of line
        e = t.find("\n", start)
        e = len(t) if e < 0 else e
        spans.append((t[start:e], start, e))
    covered = [(s, e) for _, s, e in spans]
    for m in re.finditer(r"“([^”\n]+)”|\"([^\"\n]+)\"", t):
        if not any(s <= m.start() < e for s, e in covered):
            spans.append((m.group(1) or m.group(2), m.start() + 1, m.end() - 1))
    return t, sorted(spans, key=lambda x: x[1])


def sentence(t, s, e):
    a = max([m.end() for m in SENT_END.finditer(t, 0, s)] or [0])
    m = SENT_END.search(t, e)
    b = m.start() if m else len(t)
    # the closing punctuation of a quoted sentence often sits inside the span; extend one more
    return t[a:max(b, e)]


def claims_desc(text):
    """Claimed tokens from a description: one record per token of each distinct span."""
    t, spans = quoted_spans(text)
    seen, out = set(), []
    for k, (sp, s, e) in enumerate(spans):
        key = sp.strip().lower()
        if not key or META.match(key) or key in seen:
            continue
        seen.add(key)
        ctx = t[max(0, s - 1):e + 1]
        sent = sentence(t, s, e).replace(sp, " ")
        hedged = bool(HEDGE.search(sent))
        for tok, _, _ in tokens(sp):
            out.append({"tok": tok, "span": k, "hedged": hedged, "uncertain": False})
    return out


def claims_trans(text):
    """Claimed tokens from a transcription; '(?)' marks the preceding word as uncertain."""
    out = []
    t = MD.sub("", text)
    t = re.sub(r"\[[^\]]*\]", " ", t)  # [нерозбірливо] placeholders
    for li, line in enumerate(t.split("\n")):
        parts = re.split(r"(\(\?\))", line)
        for pi, part in enumerate(parts):
            if part == "(?)":
                continue
            toks = tokens(part)
            unc_next = pi + 1 < len(parts) and parts[pi + 1] == "(?)"
            for j, (tok, _, _) in enumerate(toks):
                out.append({"tok": tok, "span": li, "hedged": False,
                            "uncertain": unc_next and j == len(toks) - 1})
    return out


def fragment_ok(o, g):
    """o is a contiguous run of non-hidden characters of partially visible GT token g."""
    chars, vis = g["chars"], g["vis"]
    if g["tok"].isdigit():
        chars = chars  # leading zeros of the raw chars are kept; compare loosely below
    for i in range(len(chars)):
        if vis[i] == 0:
            continue
        for j in range(i + 1, len(chars) + 1):
            if vis[j - 1] == 0:
                break
            frag = chars[i:j]
            if frag == o or (frag.isdigit() and (frag.lstrip("0") or "0") == o):
                return True
    return False


def russified(o, g):
    if RUSSIAN.search(o):
        return True
    if len(o) != len(g) or o == g:
        return False
    diffs = [(a, b) for a, b in zip(g, o) if a != b]
    return all(d in UA_RU_PAIRS for d in diffs)


def label(o, gt, lexicon=None):
    """Label claimed token o against the GT token list of one image."""
    best, bs = None, -1.0
    for g in gt:
        if g["state"] == "visible" and o == g["tok"]:
            return {"label": "correct", "gt": g["tok"], "gt_cat": g["cat"], "gt_state": g["state"], "sim": 1.0}
    for g in gt:
        if g["state"] == "partial" and fragment_ok(o, g):
            return {"label": "correct", "gt": g["tok"], "gt_cat": g["cat"], "gt_state": g["state"], "sim": 1.0}
    for g in gt:
        if g["state"] != "visible" and o == g["tok"]:
            return {"label": "completion", "gt": g["tok"], "gt_cat": g["cat"], "gt_state": g["state"], "sim": 1.0}
    for g in gt:
        s = sim(o, g["tok"])
        if g["state"] == "partial":  # compare with the visible fragment too
            vis = "".join(c for c, v in zip(g["chars"], g["vis"]) if v)
            s = max(s, sim(o, vis))
        if s > bs:
            best, bs = g, s
    if best is not None and bs >= 0.5:
        sub = "digit" if best["tok"].isdigit() else ("russified" if russified(o, best["tok"]) else (
            "lexicalised" if lexicon is not None and o in lexicon and best["tok"] not in lexicon else "other"))
        return {"label": "misread", "gt": best["tok"], "gt_cat": best["cat"], "gt_state": best["state"],
                "sim": round(bs, 3), "sub": sub}
    return {"label": "fabricated", "gt": best["tok"] if best else None, "gt_cat": best["cat"] if best else None,
            "gt_state": best["state"] if best else None, "sim": round(max(bs, 0), 3),
            "sub": "russified" if RUSSIAN.search(o) else "none"}
