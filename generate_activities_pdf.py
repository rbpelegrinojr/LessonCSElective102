"""
generate_activities_pdf.py - Compile the coding activities for all 20 CNN
course lessons into a standalone Activities PDF.

Usage:
    python generate_activities_pdf.py

Output:
    CNN_Image_Classification_Course_Activities.pdf  (written to the repo root)
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
    "Evaluating Your Model -- Metrics & Confusion Matrix",
    "Overfitting & Regularization Techniques",
    "Data Augmentation",
    "Batch Normalization & Dropout",
    "Transfer Learning",
    "Fine-Tuning Pretrained Models",
    "Model Visualization & Interpretability (Grad-CAM)",
    "Deploying Your CNN Model",
    "Capstone Project -- End-to-End Image Classifier",
]

# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------


class ActivitiesPDF(FPDF):
    """Custom FPDF subclass with a minimal footer showing page numbers."""

    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(
            0, 10,
            f"Image Classification Using CNNs -- Coding Activities  --  Page {self.page_no()}",
            align="C",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(text: str) -> str:
    """Replace characters outside latin-1 with ASCII equivalents."""
    replacements = {
        "\u2014": "--",
        "\u2013": "-",
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
    return text.encode("latin-1", errors="replace").decode("latin-1")


def strip_markdown(text: str) -> str:
    """Convert Markdown to plain readable text."""
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"^```[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\|(.*)\|$", lambda m: m.group(1).replace("|", "  "), text, flags=re.MULTILINE)
    text = re.sub(r"^[-*+]\s+", "  * ", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "  ", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def add_text_block(pdf: ActivitiesPDF, text: str, font: str = "Helvetica",
                   style: str = "", size: int = 10, line_height: int = 6) -> None:
    """Write a block of plain text, wrapping long lines automatically."""
    pdf.set_font(font, style, size)
    for line in text.split("\n"):
        safe_line = _safe(line)
        if safe_line.strip() == "":
            pdf.ln(line_height // 2)
        else:
            pdf.multi_cell(0, line_height, safe_line)


def extract_activities(md_path: str) -> str:
    """Return only the ## Activities section from a lesson Markdown file."""
    if not os.path.exists(md_path):
        return ""
    with open(md_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    m = re.search(r"## Activities\n(.*?)(?=\n## |\Z)", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------


def add_cover_page(pdf: ActivitiesPDF) -> None:
    pdf.add_page()
    pdf.ln(40)

    pdf.set_font("Helvetica", "B", 28)
    pdf.cell(0, 14, _safe("Image Classification Using CNNs"), align="C", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "CS Elective 102 -- Coding Activities", align="C", ln=True)
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
        "2 coding activities per lesson across all 20 lessons of the CNN Image "
        "Classification course. Each activity requires writing or running Python "
        "code using TensorFlow/Keras, NumPy, and related libraries."
    )
    pdf.multi_cell(0, 7, _safe(description), align="C")


def add_toc_page(pdf: ActivitiesPDF) -> None:
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


def add_activities_page(pdf: ActivitiesPDF, lesson_num: int, title: str,
                        md_path: str) -> None:
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
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Coding Activities", ln=True)
    pdf.ln(2)

    activities_text = extract_activities(md_path)
    if activities_text:
        plain = strip_markdown(activities_text)
        add_text_block(pdf, plain, font="Helvetica", size=10, line_height=6)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 7, f"[Activities not found in: {os.path.basename(md_path)}]", ln=True)
        print(f"  WARNING: activities not found in {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    lessons_dir = os.path.join(base_dir, "lessons")

    pdf = ActivitiesPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover
    add_cover_page(pdf)

    # Table of Contents
    add_toc_page(pdf)

    # One page per lesson (activities only)
    for i, title in enumerate(LESSON_TITLES, 1):
        lesson_num_str = f"{i:02d}"
        md_path = os.path.join(lessons_dir, f"lesson_{lesson_num_str}.md")
        print(f"  Adding activities for Lesson {lesson_num_str}: {title}")
        add_activities_page(pdf, i, title, md_path)

    # Save
    output_path = os.path.join(base_dir, "CNN_Image_Classification_Course_Activities.pdf")
    pdf.output(output_path)
    print(f"\nActivities PDF saved to: {output_path}")


if __name__ == "__main__":
    main()
