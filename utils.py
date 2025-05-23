import os

def available_pdfs() -> list:
    """
    Returns a list of all available PDF files in the current directory.
    """
    pdf_files = [f for f in os.listdir('R:/gyaan_doc/pdfs') if f.endswith('.pdf')]
    return pdf_files