# Agency, Communion, Proxemics — proposal LaTeX

IEEE-style LaTeX transcription of [`../social_HRI.pdf`](../social_HRI.pdf)
(3-page proposal by Chengxu Zhou, UCL Humanoid Robotics Lab).

This is **a faithful one-to-one transcription** of the PDF — content
unchanged. Use it to iterate on the proposal in source form.

## Files

```
docs/proposal/social_HRI/
├── main.tex          # IEEEtran journal-mode source (transcribed verbatim)
├── references.bib    # 4 references from PDF page 3
├── IEEEtran.cls      # bundled for offline build / Overleaf portability
├── IEEEtran.bst      # bundled for offline build / Overleaf portability
├── figures/          # placeholder; original PDF has no figures
└── README.md         # this file
```

## Build

```bash
cd docs/proposal/social_HRI
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Upload to Overleaf

`zip -r social_HRI_overleaf.zip main.tex references.bib IEEEtran.cls IEEEtran.bst README.md figures/` and upload to a new Overleaf project (Compiler → pdfLaTeX).
