"""
generate_pdf.py - Compile all 20 CNN course lessons into a single PDF.

Usage:
    python generate_pdf.py

Output:
    CNN_Image_Classification_Course.pdf  (written to the repository root)
"""

from fpdf import FPDF
import os
import re
from datetime import date

# ---------------------------------------------------------------------------
# Lesson metadata
# ---------------------------------------------------------------------------

LESSON_TITLES = [
    "What Is Image Classification?",
    "Understanding Digital Images & Pixels",
    "Introduction to Neural Networks",
    "From Dense Networks to CNNs",
    "The Convolution Operation Deep Dive",
    "Activation Functions",
    "Pooling Layers",
    "Building Your First Complete CNN",
    "Dataset Preparation & Loading",
    "Training a CNN",
    "Loss Functions & Optimizers",
    "Evaluating Your Model — Metrics & Confusion Matrix",
    "Overfitting & Regularization Techniques",
    "Data Augmentation",
    "Batch Normalization & Dropout",
    "Transfer Learning",
    "Fine-Tuning Pretrained Models",
    "Model Visualization & Interpretability (Grad-CAM)",
    "Deploying Your CNN Model",
    "Capstone Project — End-to-End Image Classifier",
]

# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class CoursePDF(FPDF):
    """Custom FPDF subclass with a minimal footer showing page numbers."""

    def header(self):
        # No running header — the cover/TOC pages look cleaner without one.
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0, 10,
            f"Image Classification Using CNNs  \u2014  Page {self.page_no()}",
            align="C",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Replace characters that are outside latin-1 with ASCII equivalents."""
    replacements = {
        "\u2014": "--",   # em dash
        "\u2013": "-",    # en dash
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2192": "->",
        "\u2190": "<-",
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03bb": "lambda",
        "\u03bc": "mu",
        "\u03c3": "sigma",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
        "\u221a": "sqrt",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Final fallback: encode to latin-1, replacing unmappable chars with '?'
    return text.encode("latin-1", errors="replace").decode("latin-1")


def strip_markdown(text: str) -> str:
    """Convert Markdown to plain readable text."""
    # Remove ATX headings (keep the heading text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Bold / italic
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    # Inline code
    text = re.sub(r"`(.*?)`", r"\1", text)
    # Fenced code blocks — keep content, drop fences
    text = re.sub(r"^```[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```", "", text, flags=re.MULTILINE)
    # Tables — drop separator rows, strip pipes from data rows
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|(.*)\|$", lambda m: m.group(1).replace("|", "  "), text, flags=re.MULTILINE)
    # Unordered list bullets
    text = re.sub(r"^[-*+]\s+", "  * ", text, flags=re.MULTILINE)
    # Ordered list numbers — strip numbers, keep text
    text = re.sub(r"^\d+\.\s+", "  ", text, flags=re.MULTILINE)
    # Hyperlinks — keep display text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def add_text_block(pdf: CoursePDF, text: str, font: str = "Helvetica",
                   style: str = "", size: int = 10, line_height: int = 6) -> None:
    """Write a block of plain text, wrapping long lines automatically."""
    pdf.set_font(font, style, size)
    for line in text.split("\n"):
        safe_line = _safe(line)
        if safe_line.strip() == "":
            pdf.ln(line_height // 2)
        else:
            pdf.multi_cell(0, line_height, safe_line)


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def add_cover_page(pdf: CoursePDF) -> None:
    pdf.add_page()
    pdf.ln(40)

    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 14, _safe("Image Classification Using CNNs"), align="C", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "CS Elective 102 -- Complete Course", align="C", ln=True)
    pdf.ln(20)

    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.5)
    x_margin = pdf.l_margin
    pdf.line(x_margin, pdf.get_y(), pdf.w - x_margin, pdf.get_y())
    pdf.ln(20)

    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 8, "Author: rbpelegrinojr", align="C", ln=True)
    pdf.ln(4)
    pdf.cell(0, 8, f"Generated: {date.today().strftime('%B %d, %Y')}", align="C", ln=True)
    pdf.ln(20)

    pdf.set_font("Helvetica", "I", 11)
    description = (
        "A 20-lesson university-level course covering Convolutional Neural Networks "
        "for image classification -- from pixel fundamentals to model deployment -- "
        "with full Python/TensorFlow code for every lesson."
    )
    pdf.multi_cell(0, 7, _safe(description), align="C")


def add_toc_page(pdf: CoursePDF) -> None:
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Table of Contents", ln=True)
    pdf.ln(4)

    pdf.set_draw_color(180, 180, 180)
    pdf.set_line_width(0.3)
    x_margin = pdf.l_margin
    pdf.line(x_margin, pdf.get_y(), pdf.w - x_margin, pdf.get_y())
    pdf.ln(8)

    for i, title in enumerate(LESSON_TITLES, 1):
        pdf.set_font("Helvetica", "B", 10)
        label = f"Lesson {i:02d}"
        pdf.cell(28, 7, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, _safe(title), ln=True)
        pdf.ln(1)


def add_lesson_page(pdf: CoursePDF, lesson_num: int, title: str,
                    md_path: str, py_path: str) -> None:
    pdf.add_page()

    # Lesson header
    pdf.set_font("Helvetica", "B", 16)
    header = _safe(f"Lesson {lesson_num:02d}: {title}")
    pdf.cell(0, 10, header, ln=True)
    pdf.ln(2)

    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)
    x_margin = pdf.l_margin
    pdf.line(x_margin, pdf.get_y(), pdf.w - x_margin, pdf.get_y())
    pdf.ln(6)

    # Lesson markdown content
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8", errors="replace") as f:
            raw_md = f.read()
        plain_text = strip_markdown(raw_md)
        add_text_block(pdf, plain_text, font="Helvetica", size=10, line_height=6)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 7, f"[Lesson file not found: {os.path.basename(md_path)}]", ln=True)
        print(f"  WARNING: {md_path} not found — skipping markdown section.")

    # Separator
    pdf.ln(6)
    pdf.set_draw_color(120, 120, 120)
    pdf.set_line_width(0.3)
    pdf.line(x_margin, pdf.get_y(), pdf.w - x_margin, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "--- Python Code ---", ln=True)
    pdf.ln(2)

    # Python source code
    if os.path.exists(py_path):
        with open(py_path, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()
        add_text_block(pdf, code, font="Courier", size=8, line_height=5)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 7, f"[Code file not found: {os.path.basename(py_path)}]", ln=True)
        print(f"  WARNING: {py_path} not found — skipping code section.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lessons_dir = os.path.join(base_dir, "lessons")
    code_dir = os.path.join(base_dir, "code")

    pdf = CoursePDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover
    add_cover_page(pdf)

    # Table of Contents
    add_toc_page(pdf)

    # One page (or more) per lesson
    for i, title in enumerate(LESSON_TITLES, 1):
        lesson_num_str = f"{i:02d}"
        md_path = os.path.join(lessons_dir, f"lesson_{lesson_num_str}.md")
        py_path = os.path.join(code_dir, f"lesson_{lesson_num_str}.py")
        print(f"  Adding Lesson {lesson_num_str}: {title}")
        add_lesson_page(pdf, i, title, md_path, py_path)

    # Save
    output_path = os.path.join(base_dir, "CNN_Image_Classification_Course.pdf")
    pdf.output(output_path)
    print(f"\nPDF saved to: {output_path}")


if __name__ == "__main__":
    main()
