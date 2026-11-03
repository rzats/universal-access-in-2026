"""LaTeX submission package (paper/universal-access-in-2026-latex.zip).

Bundles the built manuscript source (one .tex file), the Springer Nature class and
bibliography style, refs.bib, the generated .bbl and every figure the manuscript
includes. The source compiles with pdfLaTeX (T2A Cyrillic) or XeLaTeX (fontspec); the
preamble picks the engine. If a pdflatex binary is found (PATH or ~/.TinyTeX), the
package is test-compiled from a clean copy: pdflatex, bibtex, pdflatex x2.
Run after scripts/08_build_paper.py.
"""
import os, re, shutil, subprocess, tempfile, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "paper")
MAIN = "universal-access-in-2026"
OUT = os.path.join(P, MAIN + "-latex.zip")


def files():
    tex = open(os.path.join(P, MAIN + ".tex"), encoding="utf-8").read()
    figs = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    return [MAIN + ".tex", MAIN + ".bbl", "refs.bib", "sn-jnl.cls", "sn-mathphys-num.bst"] + figs


def test_compile(names):
    tt = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")
    env = dict(os.environ, PATH=tt + os.pathsep + os.environ["PATH"])
    if not shutil.which("pdflatex", path=env["PATH"]):
        print("pdflatex not found; test compile skipped")
        return
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(OUT) as z:
            z.extractall(d)
        os.remove(os.path.join(d, MAIN + ".bbl"))  # regenerate it, as the submission system may
        for cmd in (["pdflatex"], ["bibtex"], ["pdflatex"], ["pdflatex"]):
            subprocess.run(cmd + (["-interaction=nonstopmode"] if cmd[0] == "pdflatex" else []) + [MAIN], cwd=d, env=env,
                           capture_output=True)
        log = open(os.path.join(d, MAIN + ".log"), encoding="latin-1").read()
        errs = [l for l in log.splitlines() if l.startswith("! ")]
        undef = re.findall(r"(?:Citation|Reference) `[^']+' .*undefined", log)
        pages = re.search(r"Output written on .*\((\d+) pages", log)
        for l in errs[:5]:
            print("  ", l)
        print("pdflatex test:", "OK" if not errs and not undef and pages else "FAILED",
              f"{pages.group(1) if pages else 0} pages, {len(errs)} errors, {len(undef)} undefined refs")
        shutil.copy(os.path.join(d, MAIN + ".pdf"), os.path.join(tempfile.gettempdir(), MAIN + "-pdflatex.pdf"))


def main():
    names = files()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(P, n), n)
    print(f"{OUT}: {len(names)} files, {os.path.getsize(OUT) / 1e6:.1f} MB")
    test_compile(names)


if __name__ == "__main__":
    main()
