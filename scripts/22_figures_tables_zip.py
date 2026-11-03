"""Figures and tables as separate files (paper/universal-access-in-2026-figures-tables.zip).

Figures, numbered as in the manuscript: FigN.pdf (vector original as included) and
FigN.tif (600 dpi, LZW-compressed RGB raster of the same page).
Tables: TableN.tex (editable source: the manuscript preamble and number macros plus the
table environment, compiles with sn-jnl.cls, also included) and TableN.pdf (typeset,
cropped to the table). Numbers and cross-references are taken from the built
manuscript's .aux, so they match the PDF. Run after scripts/08_build_paper.py.
"""
import os, re, shutil, subprocess, tempfile, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "paper")
MAIN = "universal-access-in-2026"
OUT = os.path.join(P, MAIN + "-figures-tables.zip")
TECTONIC = os.path.expanduser("~/.local/bin/tectonic")


def bbox(pdf):
    """Bounding box of the non-white content of a one-page PDF, in points from the top left."""
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftocairo", "-png", "-r", "72", "-singlefile", pdf, os.path.join(d, "p")], check=True)
        return Image.open(os.path.join(d, "p.png")).convert("L").point(lambda v: 0 if v > 250 else 255).getbbox()


def compile_tex(work, name):
    r = subprocess.run([TECTONIC, name + ".tex"], cwd=work, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{name}: tectonic failed\n{r.stderr[-2000:]}")


def main():
    tex = open(os.path.join(P, MAIN + ".tex"), encoding="utf-8").read()
    aux = open(os.path.join(P, MAIN + ".aux"), encoding="utf-8").read()
    num = dict(re.findall(r"\\newlabel\{((?:tab|fig|sec)[^}]*)\}\{\{([^}]*)\}", aux))
    pre, body = tex.split("\\begin{document}", 1)
    files = {}
    # figures
    for env in re.findall(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", body, re.S):
        n = num[re.search(r"\\label\{(fig:[^}]+)\}", env).group(1)]
        src = os.path.join(P, re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", env).group(1))
        files[f"Fig{n}.pdf"] = src
        tif = os.path.join(tempfile.gettempdir(), f"Fig{n}")
        subprocess.run(["pdftocairo", "-tiff", "-tiffcompression", "lzw", "-r", "600", "-singlefile", src, tif], check=True)
        files[f"Fig{n}.tif"] = tif + ".tif"
    # tables
    work = tempfile.mkdtemp()
    shutil.copy(os.path.join(P, "sn-jnl.cls"), work)
    for env in re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", body, re.S):
        n = num[re.search(r"\\label\{(tab:[^}]+)\}", env).group(1)]
        env = re.sub(r"(\\begin\{table\*?\})\[[^\]]*\]", r"\1[h!]", env)
        env = re.sub(r"\\ref\{([^}]+)\}", lambda m: num.get(m.group(1), "??"), env)
        doc = (pre + "\\renewcommand{\\thetable}{" + n + "}\n\\begin{document}\n\\pagestyle{empty}\\thispagestyle{empty}\n"
               + env + "\n\\end{document}\n")
        name = f"Table{n}"
        open(os.path.join(work, name + ".tex"), "w", encoding="utf-8").write(doc)
        shutil.copy(os.path.join(work, name + ".tex"), os.path.join(work, name + "_full.tex"))
        compile_tex(work, name + "_full")  # A4 page, then the page is shrunk to the table (content stays vector)
        x0, y0, x1, y1 = bbox(os.path.join(work, name + "_full.pdf"))
        pad = 6
        crop = (f"\\AtBeginDocument{{\\pdfpagewidth={x1 - x0 + 2 * pad}bp \\pdfpageheight={y1 - y0 + 2 * pad}bp "
                f"\\hoffset=\\dimexpr\\hoffset-{x0 - pad}bp\\relax \\voffset=\\dimexpr\\voffset-{y0 - pad}bp\\relax}}\n")
        open(os.path.join(work, name + "_crop.tex"), "w", encoding="utf-8").write(doc.replace("\\begin{document}", crop + "\\begin{document}", 1))
        compile_tex(work, name + "_crop")
        files[name + ".tex"] = os.path.join(work, name + ".tex")
        files[name + ".pdf"] = os.path.join(work, name + "_crop.pdf")
    files["sn-jnl.cls"] = os.path.join(P, "sn-jnl.cls")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for k in sorted(files, key=lambda s: (s[0] != "F", [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])):
            z.write(files[k], k)
            print(f"{k:16s} {os.path.getsize(files[k]) / 1e6:6.2f} MB")
    print(f"{OUT}: {len(files)} files, {os.path.getsize(OUT) / 1e6:.1f} MB")
    print("work dir:", work)


if __name__ == "__main__":
    main()
