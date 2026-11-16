import json
import re
from pathlib import Path


def wrap_math(line):
    """Wrap a line in math mode. If it has prose before a colon, only wrap the part after it."""
    line = line.strip()
    if not line:
        return ""
    # Remove LaTeX commands (like \frac) before checking for prose words
    without_commands = re.sub(r"\\[A-Za-z]+", "", line)
    has_prose = re.search(r"[A-Za-z]{4,}", without_commands)

    if has_prose:
        if ":" in line:
            prose, math = line.split(":", 1)
            return f"{prose}: ${math.strip()}$"
        return line  # plain sentence, leave as is
    return f"${line}$"


def format_text_for_latex(text, is_math=True):
    """
    Cleans up text and formats multiple choice / line breaks for LaTeX.
    Set is_math=False for plain-English fields like problem_context.
    """
    if not text:
        return ""

    text = str(text).strip()

    # Escape characters that break LaTeX
    text = text.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")

    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        option = re.match(r"^([a-d])\.\s*(.*)$", line)
        if option:
            label, body = option.groups()
            body = wrap_math(body) if is_math else body
            cleaned_lines.append(rf"\quad {label}. {body}")
        else:
            cleaned_lines.append(wrap_math(line) if is_math else line)

    return "\n\n".join(cleaned_lines)

def json_to_latex(json_filepath, output_tex_filepath=None):
    json_path = Path(json_filepath)
    if not json_path.exists():
        print(f"❌ Error: File not found at {json_filepath}")
        return

    # Default output filename if not specified
    if output_tex_filepath is None:
        output_tex_filepath = json_path.with_suffix(".tex")

    with open(json_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    # Document Header
    book_title = problems[0].get("book", "Algebra 1") if problems else "Algebra 1"
    section_code = problems[0].get("block_code", "") if problems else ""
    section_title = problems[0].get("section_title", "Practice") if problems else "Practice"

    latex_doc = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{amsmath, amssymb}",
        r"\usepackage{enumitem}",
        r"\usepackage{xcolor}",
        r"\usepackage{fancyhdr}",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        rf"\lhead{{{book_title}}}",
        rf"\rhead{{Section {section_code} {section_title}}}",
        r"\cfoot{\thepage}",
        r"",
        r"\begin{document}",
        rf"\section*{{{book_title} -- Section {section_code} {section_title}}}",
        r"\noindent\rule{\textwidth}{0.5pt}",
        r"\vspace{1em}",
        r""
    ]

    for item in problems:
        num = item.get("problem_number", "")
        context = format_text_for_latex(item.get("problem_context", ""))
        text = format_text_for_latex(item.get("problem_text", ""))
        solution = format_text_for_latex(item.get("solution_text", ""))
        standard = item.get("mapped_standard", "")

        latex_doc.append(r"\begin{minipage}{\textwidth}")
        latex_doc.append(rf"\noindent\textbf{{Problem {num}}}")
        if standard:
            latex_doc.append(rf"\hfill\textit{{\color{{gray}}Standard: {standard}}}")
        latex_doc.append(r"\vspace{0.3em}")

        # Add context/instructions if present
        if context and context != "--":
            latex_doc.append(rf"\noindent\textit{{{context}}}")
            latex_doc.append(r"\vspace{0.3em}")

        # Add problem statement
        latex_doc.append(rf"\noindent {text}")
        latex_doc.append(r"\vspace{0.5em}")

        # Add solution block
        latex_doc.append(rf"\noindent\textbf{{Solution:}} \fbox{{{solution}}}")
        latex_doc.append(r"\end{minipage}")
        latex_doc.append(r"\vspace{1.5em}")
        latex_doc.append(r"\hrulefill")
        latex_doc.append(r"\vspace{1.5em}")
        latex_doc.append(r"")

    latex_doc.append(r"\end{document}")

    # Write to .tex file
    with open(output_tex_filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_doc))

    print(f"✅ Successfully converted {len(problems)} problems.")
    print(f"💾 Saved LaTeX file to: {output_tex_filepath}")


if __name__ == "__main__":
    # Change this to the path of your JSON file
    INPUT_JSON = "generated_section_6_4_6_problems.json"

    # Run conversion
    json_to_latex(INPUT_JSON)