import fitz
import os
from pathlib import Path
import re

base_dir = Path(r"c:\Users\Seam\Desktop\Research\NSU\Papers_Downloaded")
output_file = Path(r"c:\Users\Seam\Desktop\Research\NSU\extracted_abstracts.md")

out_lines = ["# Extracted Abstracts and Key Info\n"]

for pdf_path in base_dir.rglob("*.pdf"):
    try:
        doc = fitz.open(pdf_path)
        # We only need the first 2 pages to extract abstract
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text() + "\n"
        
        # very basic abstract extraction
        abstract = ""
        match = re.search(r'(?i)(?:Abstract|A B S T R A C T)(.*?)(?:Introduction|1\.\s+Introduction)', text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
        else:
            abstract = text[:2000].strip() # fallback to first 2000 chars

        out_lines.append(f"## {pdf_path.name}\n**Abstract/Intro Start:**\n{abstract}\n\n---\n")
        doc.close()
    except Exception as e:
        out_lines.append(f"## {pdf_path.name}\nError: {e}\n\n---\n")

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(out_lines)
print(f"Extracted to {output_file}")
