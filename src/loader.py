import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    This function opens a PDF file and extracts all the text from it.
    """
    print(f"Loading PDF from: {pdf_path}")
    
    # 1. Open the PDF file in 'read binary' mode
    reader = PdfReader(pdf_path)
    
    extracted_text = ""
    
    # 2. Loop through every page in the PDF
    for page_num, page in enumerate(reader.pages):
        # Extract the text and add it to our giant string
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"
            
    print(f"Successfully extracted {len(extracted_text)} characters of text!")
    return extracted_text

# --- Testing our function ---
if __name__ == "__main__":
    # Point it to the PDF in our data folder
    pdf_file = "../data/sample_meeting.pdf"
    
    # Get the absolute path so Python doesn't get confused
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_pdf_path = os.path.join(current_dir, pdf_file)
    
    # Run the function!
    raw_text = extract_text_from_pdf(absolute_pdf_path)
    
    # Print the first 500 characters just to prove it worked
    print("\n--- Preview of extracted text ---")
    print(raw_text[:500])
    print("---------------------------------")