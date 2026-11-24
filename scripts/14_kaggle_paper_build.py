"""Package paper/ into a private Kaggle CPU kernel that compiles it with pdflatex + bibtex.

LaTeX is not installed locally, so the PDF is built on Kaggle. The kernel installs TeX Live from apt, runs
pdflatex/bibtex/pdflatex/pdflatex and writes main.pdf and main.log to /kaggle/working.

    python scripts/14_kaggle_paper_build.py --user <kaggle-username>
    kaggle kernels push -p kaggle/paper-build
    kaggle kernels output <user>/toonsense-paper-build -p kaggle_output/paper-build
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import tarfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SKIP = {"results_section_full.tex", "ToonSense_IEEE.tex", "ToonSense_IEEE_overleaf.zip", "ToonSense_IEEE.pdf"}
MAX_PNG_SIDE = 1100  # large raster figures are downscaled in the build copy only (keeps the kernel small)

RUNNER = r'''
import base64, io, os, subprocess, tarfile, shutil
W = "/kaggle/working"
SRC = "/kaggle/tmp_paper"
os.makedirs(SRC, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(SRC)

def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r.returncode

if shutil.which("pdflatex") is None:
    sh("apt-get update -qq")
    sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends texlive-latex-base "
       "texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-publishers "
       "texlive-science lmodern")
sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq poppler-utils")
import re
for doc in ["main", "supplementary", "main_extended"]:
    if not os.path.exists(os.path.join(SRC, doc + ".tex")):
        continue
    steps = ["pdflatex -interaction=nonstopmode " + doc + ".tex"]
    if "bibliography{" in open(os.path.join(SRC, doc + ".tex"), encoding="utf-8").read():
        steps += ["bibtex " + doc]
    steps += ["pdflatex -interaction=nonstopmode " + doc + ".tex"] * 2
    for step in steps:
        sh(step, cwd=SRC)
    for ext in [".pdf", ".log", ".blg"]:
        f = os.path.join(SRC, doc + ext)
        if os.path.exists(f):
            shutil.copy(f, W)
    log = open(os.path.join(SRC, doc + ".log"), encoding="latin-1").read()
    print("==", doc, "PAGES:", re.findall(r"Output written on \S+ \((\d+) pages", log))
    for line in log.splitlines():
        if any(k in line for k in ["Warning", "Overfull", "undefined", "Error"]) or line.startswith("!"):
            print("  ", line[:200])
os.makedirs(os.path.join(W, "pages"), exist_ok=True)
sh("pdftoppm -r 70 -png main.pdf " + os.path.join(W, "pages", "p"), cwd=SRC)
'''


def payload() -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(PAPER.rglob("*")):
            if not p.is_file() or p.name in SKIP or "vertical" in p.name:
                continue
            rel = p.relative_to(PAPER).as_posix()
            if p.suffix == ".png" and not rel.startswith("figures/cmban/gradcam"):
                continue  # only the Grad-CAM figure is included as a PNG; the others are used as PDF
            data = p.read_bytes()
            if p.suffix == ".png":
                im = Image.open(p)
                im.thumbnail((MAX_PNG_SIDE, MAX_PNG_SIDE))
                b = io.BytesIO()
                im.convert("RGB").quantize(colors=128).save(b, format="PNG", optimize=True)
                data = b.getvalue()
            info = tarfile.TarInfo(rel)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"), required="KAGGLE_USERNAME" not in os.environ)
    args = ap.parse_args()
    out = ROOT / "kaggle" / "paper-build"
    out.mkdir(parents=True, exist_ok=True)
    code = "PAYLOAD = '" + payload() + "'\n" + RUNNER
    (out / "run.py").write_text(code, encoding="utf-8")
    meta = {"id": args.user + "/toonsense-paper-build", "title": "toonsense-paper-build", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out / "run.py", "(%.0f KB)" % ((out / "run.py").stat().st_size / 1024))


if __name__ == "__main__":
    main()
