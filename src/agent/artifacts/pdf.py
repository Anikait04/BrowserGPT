# pdf.py — detailed-report PDF rendering (extracted from extract_information.py).
from __future__ import annotations

from src.workflow.schemas import ExtractionContent


def sanitize(text: str) -> str:
    """Strip characters the PDF core fonts cannot render."""
    return (text or "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(content: ExtractionContent, dest_path: str) -> None:
    """Render report content to a PDF file."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    class ReportPDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = ReportPDF()
    pdf.alias_nb_pages("{nb}")
    pdf.set_auto_page_break(True, margin=20)
    pdf.add_page()

    def _line(style: str, size: int, text: str, height: int) -> None:
        pdf.set_font("helvetica", style, size)
        pdf.multi_cell(0, height, sanitize(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    _line("B", 20, content.title or "Report", 10)
    pdf.ln(4)

    if content.summary:
        _line("I", 11, content.summary, 7)
        pdf.ln(4)

    for section in content.sections:
        if section.heading:
            _line("B", 14, section.heading, 8)
        if section.body:
            _line("", 11, section.body, 7)
        pdf.ln(3)

    if content.sources:
        _line("B", 12, "Sources", 8)
        for source in content.sources:
            _line("", 10, f"- {source}", 6)

    pdf.output(dest_path)
