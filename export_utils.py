from docx import Document
from fpdf import FPDF

def export_as_docx(content, filename="Tailored_Resume.docx"):
    doc = Document()
    for line in content.split("\n"):
        doc.add_paragraph(line)
    doc.save(filename)
    return filename

def export_as_pdf(content, filename="Tailored_Resume.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(filename)
    return filename
    
