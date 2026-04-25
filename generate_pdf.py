"""
generate_pdf.py
---------------
Compiles all 20 lessons of the "Image Classification Using CNNs" course
into a single, well-formatted PDF document.

Usage:
    pip install fpdf2
    python generate_pdf.py

Output:
    CNN_Image_Classification_Course.pdf
"""

import os
import re
from datetime import date
from fpdf import FPDF

# ── Configuration ────────────────────────────────────────────────────────────

LESSONS_DIR = "lessons"
CODE_DIR = "code"
OUTPUT_FILE = "CNN_Image_Classification_Course.pdf"
COURSE_TITLE = "Image Classification Using CNNs"
COURSE_SUBTITLE = "A Complete 20-Lesson Course with TensorFlow & Keras"
TOTAL_LESSONS = 20

LESSON_TITLES = [
    "What Is Image Classification?",
    "Understanding Digital Images and Pixels",
    "Introduction to Neural Networks",
    "From Dense Networks to CNNs",
    "The Convolution Operation Deep Dive",
    "Activation Functions",
    "Pooling Layers",
    "Building Your First Complete CNN",
    "Dataset Preparation and Loading",
    "Training a CNN",
    "Loss Functions and Optimizers",
    "Evaluating Your Model — Metrics and Confusion Matrix",
    "Overfitting and Regularization Techniques",
    "Data Augmentation",
    "Batch Normalization and Dropout",
    "Transfer Learning",
    "Fine-Tuning Pretrained Models",
    "Model Visualization and Interpretability with Grad-CAM",
    "Deploying Your CNN Model",
    "Capstone Project — End-to-End Image Classifier",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """
    Convert markdown text to plain readable text suitable for PDF rendering.
    Removes markdown symbols while preserving structure.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Convert headers (keep the text, remove #)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Convert bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    # Convert inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove code fences (``` blocks) markers only — keep content
    text = re.sub(r"^```[a-z]*\s*$", "", text, flags=re.MULTILINE)
    # Convert links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Convert reference links [text][ref]
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Convert unordered list markers
    text = re.sub(r"^\s*[-*+]\s+", "  • ", text, flags=re.MULTILINE)
    # Convert ordered list markers
    text = re.sub(r"^\s*\d+\.\s+", lambda m: "  " + m.group(0).strip(), text, flags=re.MULTILINE)
    # Remove table alignment rows (|---|---|)
    text = re.sub(r"^\|[-| :]+\|$", "", text, flags=re.MULTILINE)
    # Clean up table separators (keep content)
    text = re.sub(r"\|", "  ", text)
    # Remove blockquote markers
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines to a maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_file(filepath: str) -> str:
    """Read a file and return its content, or return an error message if missing."""
    if not os.path.exists(filepath):
        return f"[File not found: {filepath}]"
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


# ── PDF Class ─────────────────────────────────────────────────────────────────

class CoursePDF(FPDF):
    """Custom FPDF subclass with header and footer defined."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Render a thin top bar on every page (except cover)."""
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8, COURSE_TITLE, align="L")
            self.ln(0)
            self.set_draw_color(200, 200, 200)
            self.line(10, 14, 200, 14)
            self.ln(4)
            self.set_text_color(0, 0, 0)

    def footer(self):
        """Render page number at the bottom of every page."""
        if self.page_no() > 1:
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")
            self.set_text_color(0, 0, 0)

    # ── Cover Page ────────────────────────────────────────────────────────────

    def add_cover_page(self):
        """Render a styled cover page with title, subtitle, and date."""
        self.add_page()

        # Background header block
        self.set_fill_color(30, 60, 120)
        self.rect(0, 0, 210, 80, "F")

        # Title
        self.set_y(20)
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(255, 255, 255)
        self.multi_cell(0, 12, COURSE_TITLE, align="C")

        # Subtitle
        self.set_y(55)
        self.set_font("Helvetica", "", 14)
        self.set_text_color(200, 220, 255)
        self.multi_cell(0, 8, COURSE_SUBTITLE, align="C")

        # Reset colors
        self.set_text_color(0, 0, 0)
        self.set_y(100)

        # Course description
        self.set_font("Helvetica", "", 12)
        description = (
            "This course guides you from the very basics of image classification "
            "through to deploying production-ready Convolutional Neural Network (CNN) "
            "models. Each of the 20 lessons includes detailed explanations, hands-on "
            "Python code, activities, and review questions."
        )
        self.multi_cell(0, 7, description, align="C")
        self.ln(10)

        # Stats table
        self.set_fill_color(245, 245, 250)
        self.set_draw_color(200, 200, 200)
        stats = [
            ("📚 Lessons", "20"),
            ("🐍 Code Files", "20"),
            ("🎯 Activities", "100+"),
            ("❓ Review Questions", "100+"),
        ]
        self.set_font("Helvetica", "B", 11)
        col_w = 85
        x_start = (210 - col_w * 2) / 2
        self.set_x(x_start)
        for label, value in stats:
            self.set_x(x_start)
            self.set_fill_color(230, 240, 255)
            self.cell(col_w, 10, label, border=1, fill=True)
            self.set_fill_color(255, 255, 255)
            self.cell(col_w, 10, value, border=1, fill=True, align="C")
            self.ln()

        self.ln(15)

        # Framework badges
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(255, 140, 0)
        self.set_text_color(255, 255, 255)
        badge_w = 40
        badge_x = (210 - badge_w * 4 - 6 * 3) / 2
        badges = ["TensorFlow", "Keras", "NumPy", "scikit-learn"]
        self.set_x(badge_x)
        for badge in badges:
            self.set_x(self.get_x())
            self.cell(badge_w, 8, badge, border=0, fill=True, align="C")
            self.set_x(self.get_x() + 6)
        self.ln(15)

        # Date
        self.set_text_color(100, 100, 100)
        self.set_font("Helvetica", "I", 10)
        self.set_fill_color(255, 255, 255)
        self.cell(0, 8, f"Generated on {date.today().strftime('%B %d, %Y')}", align="C")

    # ── Table of Contents ─────────────────────────────────────────────────────

    def add_toc_page(self):
        """Render a table of contents listing all 20 lessons."""
        self.add_page()
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, "Table of Contents", align="C")
        self.ln(6)
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

        self.set_font("Helvetica", "", 11)
        self.set_text_color(0, 0, 0)

        for i, title in enumerate(LESSON_TITLES, start=1):
            lesson_num = f"Lesson {i:02d}"
            # Alternating row shading
            if i % 2 == 0:
                self.set_fill_color(245, 248, 255)
                fill = True
            else:
                self.set_fill_color(255, 255, 255)
                fill = True

            # Bold the lesson number, normal for the title
            self.set_font("Helvetica", "B", 11)
            self.cell(28, 9, lesson_num, fill=fill)
            self.set_font("Helvetica", "", 11)
            self.cell(0, 9, title, fill=fill)
            self.ln()

        self.ln(8)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Each lesson includes: Explanation · Key Concepts · Activities · Review Questions · Code", align="C")

    # ── Lesson Section ────────────────────────────────────────────────────────

    def add_lesson(self, lesson_num: int, md_content: str, py_content: str):
        """
        Render one lesson: starts a new page with a styled header,
        renders the markdown as plain text, then shows the Python code.
        """
        title = LESSON_TITLES[lesson_num - 1]

        # ── Lesson Header Page ──
        self.add_page()

        # Coloured header band
        self.set_fill_color(30, 60, 120)
        self.rect(0, 0, 210, 30, "F")
        self.set_y(8)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"Lesson {lesson_num:02d}", align="L")
        self.ln(7)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(200, 220, 255)
        # Truncate very long titles for the header band
        display_title = title if len(title) <= 65 else title[:62] + "..."
        self.cell(0, 7, display_title, align="L")

        self.set_text_color(0, 0, 0)
        self.set_y(38)

        # ── Lesson Markdown Content ──
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, "Lesson Notes", align="L")
        self.ln(2)
        self.set_draw_color(30, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

        plain_text = strip_markdown(md_content)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        # Split into lines and handle encoding
        for line in plain_text.split("\n"):
            # Replace any remaining unsupported characters
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            if safe_line.strip() == "":
                self.ln(3)
            else:
                self.multi_cell(0, 5, safe_line)

        # ── Python Code Section ──
        self.add_page()
        self.set_fill_color(30, 60, 120)
        self.rect(0, 0, 210, 20, "F")
        self.set_y(6)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"Lesson {lesson_num:02d} — Python Code: code/lesson_{lesson_num:02d}.py", align="L")

        self.set_text_color(0, 0, 0)
        self.set_y(26)

        # Code rendered in Courier monospace
        self.set_font("Courier", "", 7)
        self.set_fill_color(248, 248, 252)
        self.rect(8, self.get_y(), 194, 0, "F")  # background box starts here

        for line in py_content.split("\n"):
            # Preserve indentation; replace unsupported chars
            safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
            self.set_fill_color(248, 248, 252)
            self.multi_cell(0, 4, safe_line, fill=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" CNN Image Classification Course — PDF Generator")
    print("=" * 60)
    print()

    pdf = CoursePDF()
    pdf.set_title(COURSE_TITLE)
    pdf.set_author("CS Elective 102")
    pdf.set_creator("generate_pdf.py — fpdf2")

    # Cover page
    print("Adding cover page...")
    pdf.add_cover_page()

    # Table of contents
    print("Adding table of contents...")
    pdf.add_toc_page()

    # All 20 lessons
    missing_md = []
    missing_py = []

    for i in range(1, TOTAL_LESSONS + 1):
        md_path = os.path.join(LESSONS_DIR, f"lesson_{i:02d}.md")
        py_path = os.path.join(CODE_DIR, f"lesson_{i:02d}.py")

        print(f"Adding Lesson {i:02d}: {LESSON_TITLES[i-1][:50]}...")

        md_content = read_file(md_path)
        py_content = read_file(py_path)

        if "[File not found" in md_content:
            missing_md.append(md_path)
        if "[File not found" in py_content:
            missing_py.append(py_path)

        pdf.add_lesson(i, md_content, py_content)

    # Save PDF
    print()
    print(f"Saving PDF to: {OUTPUT_FILE}")
    pdf.output(OUTPUT_FILE)

    print()
    print("=" * 60)
    print(f"✅  PDF generated successfully: {OUTPUT_FILE}")
    print(f"    Total pages: {pdf.page}")
    print("=" * 60)

    if missing_md or missing_py:
        print()
        print("⚠️  WARNING: The following files were not found:")
        for f in missing_md + missing_py:
            print(f"    - {f}")


if __name__ == "__main__":
    main()
