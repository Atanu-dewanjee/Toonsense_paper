"""Flatten the paper into self-contained .tex files and build an Overleaf-ready zip.

Writes
  paper/ToonSense_IEEE.tex                  main.tex (8-page conference version) with every \\input inlined
  paper/ToonSense_supplementary.tex         supplementary.tex, flattened
  paper/ToonSense_IEEE_overleaf.zip         both files + references.bib + every referenced figure (+ PDFs if built)

PDFs compiled on Kaggle (scripts/14_kaggle_paper_build.py) are copied in from kaggle_output/paper-build/ when present.

    python scripts/15_flatten_paper.py
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = ROOT / "kaggle_output" / "paper-build"
BS = chr(92)
INPUT = re.compile(re.escape(BS) + r"(input|inputifexists)\{([^}#]+)\}")
DOCS = [("main.tex", "ToonSense_IEEE"), ("supplementary.tex", "ToonSense_supplementary"),
        ("main_extended.tex", "ToonSense_extended")]


def expand(rel: str) -> str:
    p = PAPER / rel
    if not p.suffix:
        p = p.with_suffix(".tex")
    if not p.exists():
        return ""  # \inputifexists semantics
    txt = p.read_text(encoding="utf-8")
    return INPUT.sub(lambda m: expand(m.group(2)).rstrip() + "\n", txt)


def flatten(src: str) -> str:
    flat = expand(src)
    flat = re.sub(re.escape(BS) + r"newcommand\{" + re.escape(BS) + r"inputifexists\}.*\n", "", flat)
    return re.sub(r"\n{3,}", "\n\n", flat)


def main() -> None:
    with zipfile.ZipFile(PAPER / "ToonSense_IEEE_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(PAPER / "references.bib", "references.bib")
        figures = set()
        for src, name in DOCS:
            flat = flatten(src)
            (PAPER / (name + ".tex")).write_text(flat, encoding="utf-8")
            z.write(PAPER / (name + ".tex"), name + ".tex")
            figures |= set(re.findall(re.escape(BS) + r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", flat))
            pdf = BUILD / (Path(src).stem + ".pdf")
            if pdf.exists():
                shutil.copy(pdf, PAPER / (name + ".pdf"))
                z.write(PAPER / (name + ".pdf"), name + ".pdf")
            print("wrote paper/%s.tex (%d lines)%s" % (name, len(flat.splitlines()),
                                                      " + PDF" if pdf.exists() else ""))
        for g in sorted(figures):
            z.write(PAPER / g, g)
    print("zip: paper/ToonSense_IEEE_overleaf.zip with", len(figures), "figures")


if __name__ == "__main__":
    main()
