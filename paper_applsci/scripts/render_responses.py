"""Render the plain-text point-by-point response as a standalone LaTeX file."""
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]


def escape(text):
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def main():
    lines = (PAPER / "reviewer_response.txt").read_text().splitlines()
    content = [r"""% Generated from reviewer_response.txt by scripts/render_responses.py.
\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=25mm]{geometry}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{needspace}
\definecolor{reviewerred}{RGB}{128,0,0}
\newenvironment{reviewercomment}{%
  \begin{list}{}{\setlength{\leftmargin}{1.5em}%
    \setlength{\rightmargin}{0pt}\setlength{\topsep}{0.5em}}%
  \item\relax\color{reviewerred}\ignorespaces
}{\end{list}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\setlength{\emergencystretch}{2em}
\begin{document}
"""]
    for i, line in enumerate(lines):
        if not line:
            continue
        if line.startswith(">> "):
            assert i + 1 < len(lines) and lines[i + 1].startswith("R: ")
            content.append(r"\begin{reviewercomment}" + escape(line[3:])
                           + r"\end{reviewercomment}")
        elif line.startswith("R: "):
            content.append(r"{\color{black}\textbf{R:} " + escape(line[3:]) + r"\par}")
        elif line in {"EDITOR", "REVIEWER #1", "REVIEWER #2"} or i == 0:
            content.append(r"\section*{" + escape(line) + "}")
        elif line == "General assessment" or line.startswith("Comment "):
            content.append(r"\Needspace{14\baselineskip}\subsection*{" + escape(line) + "}")
        else:
            content.append(escape(line) + r"\par")
    content.append(r"\end{document}")
    (PAPER / "reviewer_response.tex").write_text("\n\n".join(content) + "\n")


if __name__ == "__main__":
    main()
