import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import fitz
import pdfplumber
import docx
import os

_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(script_dir), "models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize easyocr reader lazily
        _easyocr_reader = easyocr.Reader(['hi', 'en'], model_storage_directory=models_dir)
    return _easyocr_reader

def table_to_markdown(table_2d_list):
    if not table_2d_list:
        return ""
    
    # Clean up None values and newlines in cells
    cleaned_table = []
    for row in table_2d_list:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                cleaned_row.append(str(cell).replace("\n", " ").strip())
        cleaned_table.append(cleaned_row)
        
    if not cleaned_table or not cleaned_table[0]:
        return ""

    num_cols = len(cleaned_table[0])
    
    md_lines = []
    
    # Add header
    header = " | ".join(cleaned_table[0])
    md_lines.append(f"| {header} |")
    
    # Add separator
    separator = " | ".join(["---"] * num_cols)
    md_lines.append(f"| {separator} |")
    
    # Add body
    for row in cleaned_table[1:]:
        # Ensure row length matches num_cols (pad or truncate)
        row_cells = row[:num_cols] + [""] * max(0, num_cols - len(row))
        md_row = " | ".join(row_cells)
        md_lines.append(f"| {md_row} |")
        
    return "\n".join(md_lines) + "\n"

def is_complex_page(fitz_page):
    # Checks for high-density graphic regions or object counts
    images = fitz_page.get_images()
    drawings = fitz_page.get_drawings()
    return len(images) > 3 or len(drawings) > 15

def extract_complex_table(plumb_page, fitz_page):
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 15,
        "snap_tolerance": 5
    }
    
    tables = plumb_page.find_tables(table_settings)
    table_bboxes = [table.bbox for table in tables]
    
    def not_in_table(obj):
        for bbox in table_bboxes:
            if (obj.get('x0', 0) >= bbox[0] and obj.get('x1', 0) <= bbox[2] and
                obj.get('top', 0) >= bbox[1] and obj.get('bottom', 0) <= bbox[3]):
                return False
            if not (obj.get('x1', 0) < bbox[0] or obj.get('x0', 0) > bbox[2] or 
                    obj.get('bottom', 0) < bbox[1] or obj.get('top', 0) > bbox[3]):
                return False
        return True
    
    filtered_page = plumb_page.filter(not_in_table)
    text_outside_tables = filtered_page.extract_text()
    
    page_content_parts = []
    if text_outside_tables:
        page_content_parts.append(text_outside_tables)
        page_content_parts.append("\n")
        
    for table in tables:
        raw_table = table.extract()
        if not raw_table:
            continue
            
        # Identify QA/Status columns based on headers (case-insensitive)
        qa_status_cols = set()
        if raw_table and len(raw_table) > 0:
            header_row = raw_table[0]
            for col_idx, cell_text in enumerate(header_row):
                if cell_text:
                    text_lower = str(cell_text).lower()
                    if "status" in text_lower or "signature" in text_lower or "qa" in text_lower:
                        qa_status_cols.add(col_idx)
                        
        refined_table_2d = []
        for row_idx, row in enumerate(table.rows):
            refined_row = []
            for col_idx, cell_bbox in enumerate(row.cells):
                cell_val = raw_table[row_idx][col_idx]
                if cell_bbox is None:
                    refined_row.append(cell_val)
                    continue
                    
                is_qa_or_status = col_idx in qa_status_cols
                if not is_qa_or_status:
                    # Check if the row contains label in another column (e.g. key-value layout)
                    for val in raw_table[row_idx]:
                        if val:
                            val_lower = str(val).lower()
                            if "status" in val_lower or "signature" in val_lower or "qa" in val_lower:
                                is_qa_or_status = True
                                break
                                
                is_empty_or_junk = False
                if not cell_val:
                    is_empty_or_junk = True
                else:
                    stripped = str(cell_val).strip()
                    if stripped in ["", "-", "_", ".", "☐", "☒", "✔", "✗", "o", "x"]:
                        is_empty_or_junk = True
                        
                if is_qa_or_status and is_empty_or_junk:
                    try:
                        rect = fitz.Rect(cell_bbox)
                        pix = fitz_page.get_pixmap(clip=rect)
                        img_bytes = pix.tobytes("png")
                        
                        reader = get_easyocr_reader()
                        ocr_result = reader.readtext(img_bytes, detail=0)
                        if ocr_result:
                            ocr_text = " ".join(ocr_result).strip()
                            if ocr_text:
                                cell_val = ocr_text
                    except Exception:
                        pass
                        
                refined_row.append(cell_val)
            refined_table_2d.append(refined_row)
            
        md_table = table_to_markdown(refined_table_2d)
        if md_table:
            page_content_parts.append("\n")
            page_content_parts.append(md_table)
            page_content_parts.append("\n")
            
    return "".join(page_content_parts)

def process_pdf(filepath):
    output_text = []
    
    with pdfplumber.open(filepath) as plumb_doc, fitz.open(filepath) as fitz_doc:
        # Strict 15-page limit constraint
        num_pages = min(15, len(plumb_doc.pages))
        for page_num in range(num_pages):
            output_text.append(f"\n[--- Page {page_num + 1} ---]\n")
            
            plumb_page = plumb_doc.pages[page_num]
            fitz_page = fitz_doc[page_num]
            
            if is_complex_page(fitz_page):
                page_content = extract_complex_table(plumb_page, fitz_page)
                output_text.append(page_content)
            else:
                # Keep using PyMuPDF for standard text extraction
                raw_text = fitz_page.get_text()
                images = fitz_page.get_images()
                
                # Check if it's an image-only page (text < 50 chars) and actually contains images
                if len(raw_text.strip()) < 50 and len(images) > 0:
                    # Lazy loading OCR
                    reader = get_easyocr_reader()
                    pix = fitz_page.get_pixmap()
                    # Save temporarily for easyocr
                    img_path = f"/tmp/page_{page_num}_temp.png"
                    pix.save(img_path)
                    
                    try:
                        ocr_result = reader.readtext(img_path, detail=0)
                        page_content = "\n".join(ocr_result)
                        output_text.append(page_content)
                    except Exception as e:
                        output_text.append(f"[OCR Failed: {str(e)}]")
                    finally:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                else:
                    output_text.append(raw_text)
    return "".join(output_text)

def process_docx(filepath):
    doc = docx.Document(filepath)
    output_text = ["[--- Document Start ---]\n"]
    
    # In a typical python-docx flow, we iterate over block-level elements
    for element in doc.element.body:
        if element.tag.endswith('p'):
            # paragraph
            p = docx.text.paragraph.Paragraph(element, doc)
            if p.text.strip():
                output_text.append(p.text)
                output_text.append("\n")
        elif element.tag.endswith('tbl'):
            # table
            t = docx.table.Table(element, doc)
            table_2d = []
            for row in t.rows:
                row_data = [cell.text for cell in row.cells]
                table_2d.append(row_data)
            md_table = table_to_markdown(table_2d)
            if md_table:
                output_text.append("\n")
                output_text.append(md_table)
                output_text.append("\n")
                
    return "".join(output_text)

def process_image(filepath):
    try:
        reader = get_easyocr_reader()
        results = reader.readtext(filepath, detail=0)
        return "\n".join(results)
    except Exception as e:
        return f"Error processing image: {str(e)}"

def process_document(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return process_pdf(filepath)
    elif ext == '.docx':
        return process_docx(filepath)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return process_image(filepath)
    else:
        return ""
