"""Generate a PDF from DETECTION_GUIDE.md with rendered Mermaid diagrams and images.

Uses Playwright to render HTML (with mermaid.js for diagrams) to PDF.

Usage:
    python scripts/generate_detection_pdf.py
"""

from __future__ import annotations

import base64
import re
import sys
import time
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright


DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
MD_FILE = DOCS_DIR / "DETECTION_GUIDE.md"
IMAGES_DIR = DOCS_DIR / "images"
OUTPUT_PDF = DOCS_DIR / "DETECTION_GUIDE.pdf"


def _embed_image_as_base64(match: re.Match) -> str:
    """Replace markdown image reference with embedded base64 data URI."""
    alt = match.group(1)
    src = match.group(2)

    # Resolve relative to docs/
    img_path = DOCS_DIR / src
    if not img_path.exists():
        print(f"  WARNING: Image not found: {img_path}")
        return match.group(0)

    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    suffix = img_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".svg": "image/svg+xml"}.get(suffix, "image/png")

    return f'![{alt}](data:{mime};base64,{b64})'


def _extract_mermaid_from_md(md_text: str) -> tuple[str, dict[str, str]]:
    """Extract mermaid code blocks from markdown, replace with placeholders.

    Returns (modified_md, dict of placeholder_id -> mermaid_content).
    This must happen BEFORE markdown conversion to avoid syntax highlighting
    mangling the mermaid syntax.
    """
    mermaid_blocks: dict[str, str] = {}
    counter = 0

    def replace_block(m: re.Match) -> str:
        nonlocal counter
        content = m.group(1).strip()
        placeholder = f"MERMAID_PLACEHOLDER_{counter}"
        mermaid_blocks[placeholder] = content
        counter += 1
        return f"<!-- {placeholder} -->"

    # Match ```mermaid ... ``` blocks
    pattern = r'```mermaid\s*\n(.*?)```'
    result = re.sub(pattern, replace_block, md_text, flags=re.DOTALL)
    return result, mermaid_blocks


def _restore_mermaid_in_html(html: str, mermaid_blocks: dict[str, str]) -> int:
    """Replace HTML comment placeholders with mermaid divs."""
    count = 0
    for placeholder, content in mermaid_blocks.items():
        target = f"<!-- {placeholder} -->"
        # The markdown converter wraps comments in <p> tags
        for wrapper in [f"<p>{target}</p>", target]:
            if wrapper in html:
                html = html.replace(wrapper, f'<div class="mermaid">\n{content}\n</div>')
                count += 1
                break
    return html, count


def _prepare_markdown(md_text: str) -> str:
    """Embed images as base64 and prepare markdown for conversion."""
    # Embed all image references as base64
    md_text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _embed_image_as_base64, md_text)
    return md_text


def _build_html(md_text: str) -> str:
    """Convert markdown to a full HTML document with Mermaid support."""
    # Extract mermaid blocks BEFORE markdown conversion
    md_text, mermaid_blocks = _extract_mermaid_from_md(md_text)
    print(f"  Extracted {len(mermaid_blocks)} Mermaid blocks from markdown")

    # Convert markdown to HTML (without mermaid blocks being mangled)
    extensions = ["tables", "fenced_code", "codehilite", "toc", "attr_list"]
    html_body = markdown.markdown(md_text, extensions=extensions)

    # Restore mermaid blocks as proper <div class="mermaid"> elements
    html_body, mermaid_count = _restore_mermaid_in_html(html_body, mermaid_blocks)
    print(f"  Restored {mermaid_count} Mermaid diagrams in HTML")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: A4;
        margin: 20mm 15mm 20mm 15mm;
    }}

    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #1a1a1a;
        max-width: 100%;
        margin: 0;
        padding: 0;
    }}

    h1 {{
        font-size: 22pt;
        border-bottom: 3px solid #2c3e50;
        padding-bottom: 8px;
        margin-top: 0;
        color: #1a1a2e;
        page-break-after: avoid;
    }}

    h2 {{
        font-size: 16pt;
        border-bottom: 2px solid #34495e;
        padding-bottom: 6px;
        margin-top: 30px;
        color: #1a1a2e;
        page-break-after: avoid;
    }}

    h3 {{
        font-size: 13pt;
        color: #2c3e50;
        margin-top: 20px;
        page-break-after: avoid;
    }}

    h4 {{
        font-size: 11pt;
        color: #34495e;
        page-break-after: avoid;
    }}

    p {{
        margin: 8px 0;
        orphans: 3;
        widows: 3;
    }}

    code {{
        background: #f0f0f0;
        padding: 1px 4px;
        border-radius: 3px;
        font-family: "Consolas", "Monaco", "Courier New", monospace;
        font-size: 9.5pt;
    }}

    pre {{
        background: #f5f5f5;
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 10px 12px;
        overflow-x: auto;
        font-size: 9pt;
        line-height: 1.4;
        page-break-inside: avoid;
    }}

    pre code {{
        background: none;
        padding: 0;
        font-size: 9pt;
    }}

    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 9.5pt;
        page-break-inside: avoid;
    }}

    th, td {{
        border: 1px solid #ccc;
        padding: 6px 10px;
        text-align: left;
    }}

    th {{
        background: #2c3e50;
        color: white;
        font-weight: 600;
    }}

    tr:nth-child(even) {{
        background: #f9f9f9;
    }}

    img {{
        max-width: 100%;
        height: auto;
        display: block;
        margin: 12px auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        page-break-inside: avoid;
    }}

    em {{
        color: #555;
        font-size: 9.5pt;
    }}

    hr {{
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 25px 0;
    }}

    ul, ol {{
        margin: 8px 0;
        padding-left: 25px;
    }}

    li {{
        margin: 3px 0;
    }}

    strong {{
        color: #1a1a2e;
    }}

    .mermaid {{
        text-align: center;
        margin: 15px 0;
        page-break-inside: avoid;
    }}

    .mermaid svg {{
        max-width: 100%;
        height: auto;
    }}

    a {{
        color: #2980b9;
        text-decoration: none;
    }}

    blockquote {{
        border-left: 4px solid #2980b9;
        margin: 10px 0;
        padding: 8px 15px;
        background: #f0f7ff;
    }}

    /* Page break hints */
    h2 {{
        page-break-before: auto;
    }}

    .mermaid, img, table, pre {{
        page-break-inside: avoid;
    }}
</style>
</head>
<body>
{html_body}

<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
    mermaid.initialize({{
        startOnLoad: true,
        theme: 'default',
        securityLevel: 'loose',
        flowchart: {{
            useMaxWidth: true,
            htmlLabels: true,
            curve: 'basis'
        }},
        themeVariables: {{
            fontSize: '12px'
        }}
    }});
</script>
</body>
</html>"""
    return html


def generate_pdf():
    """Main function to generate the PDF."""
    print(f"Reading {MD_FILE}...")
    md_text = MD_FILE.read_text(encoding="utf-8")

    print("Embedding images as base64...")
    md_text = _prepare_markdown(md_text)

    print("Converting Markdown to HTML...")
    html = _build_html(md_text)

    # Write intermediate HTML for debugging
    html_path = DOCS_DIR / "DETECTION_GUIDE.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"  Intermediate HTML: {html_path}")

    print("Rendering PDF with Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Load the HTML
        page.set_content(html, wait_until="networkidle")

        # Wait for Mermaid to render all diagrams
        print("  Waiting for Mermaid diagrams to render...")
        # Wait for mermaid to finish - it converts .mermaid divs to SVGs
        try:
            page.wait_for_function(
                """() => {
                    const divs = document.querySelectorAll('.mermaid');
                    if (divs.length === 0) return true;
                    return Array.from(divs).every(
                        d => d.querySelector('svg') !== null ||
                             d.getAttribute('data-processed') === 'true'
                    );
                }""",
                timeout=30000,
            )
        except Exception as e:
            print(f"  WARNING: Mermaid rendering may be incomplete: {e}")

        # Additional wait for rendering to settle
        page.wait_for_timeout(2000)

        # Check how many diagrams rendered
        rendered = page.evaluate(
            "document.querySelectorAll('.mermaid svg').length"
        )
        total = page.evaluate(
            "document.querySelectorAll('.mermaid').length"
        )
        print(f"  Rendered {rendered}/{total} Mermaid diagrams")

        # Generate PDF
        page.pdf(
            path=str(OUTPUT_PDF),
            format="A4",
            margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
            print_background=True,
            display_header_footer=True,
            header_template='<div style="font-size:8px;color:#999;width:100%;text-align:center;">Star Pattern AI - Detection Guide</div>',
            footer_template='<div style="font-size:8px;color:#999;width:100%;text-align:center;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
        )

        browser.close()

    size_mb = OUTPUT_PDF.stat().st_size / (1024 * 1024)
    print(f"\nPDF generated: {OUTPUT_PDF} ({size_mb:.1f} MB)")

    # Clean up intermediate HTML
    html_path.unlink(missing_ok=True)
    print("Cleaned up intermediate HTML.")


if __name__ == "__main__":
    generate_pdf()
