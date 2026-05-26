import sys
import asyncio
from pathlib import Path
import markdown
from pyppeteer import launch


def md_to_html(md_text: str, base_href: str = None) -> str:
    html_body = markdown.markdown(md_text, extensions=["fenced_code", "codehilite"])
    css = """
    <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 40px; }
    pre { background: #f6f8fa; padding: 10px; border-radius: 4px; }
    code { font-family: Consolas, monospace; }
    h1,h2,h3 { color: #111827 }
    </style>
    """
    base_tag = f'<base href="{base_href}">' if base_href else ''
    return f"<html><head><meta charset=\"utf-8\">{base_tag}{css}</head><body>{html_body}</body></html>"


async def html_to_pdf(html: str, out_path: Path):
    # Try launching an existing Chrome/Edge if available to avoid downloading Chromium
    candidates = [
        Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path(r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path(r"C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
        Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    ]
    browser = None
    for p in candidates:
        if p.exists():
            try:
                browser = await launch({'executablePath': str(p), 'args': ['--no-sandbox']})
                break
            except Exception:
                browser = None
    if browser is None:
        # Fall back to default launch (may download Chromium)
        browser = await launch({'args': ['--no-sandbox']})
    page = await browser.newPage()
    await page.setContent(html)
    try:
        await page.waitForFunction('document.readyState === "complete"', timeout=5000)
    except Exception:
        await asyncio.sleep(0.5)
    # Wait for all images to load (helps when using file:// base href)
    try:
        await page.evaluate('''() => {
            const imgs = Array.from(document.images || []);
            return Promise.all(imgs.map(img => img.complete ? Promise.resolve() : new Promise(res => { img.onload = res; img.onerror = res; })));
        }''')
    except Exception:
        await asyncio.sleep(0.5)
    await page.pdf({'path': str(out_path), 'format': 'A4', 'printBackground': True})
    await browser.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: md_to_pdf.py input.md output.pdf")
        sys.exit(2)
    md_path = Path(sys.argv[1])
    out_pdf = Path(sys.argv[2])
    md_text = md_path.read_text(encoding='utf8')
    base_href = md_path.parent.resolve().as_uri()
    html = md_to_html(md_text, base_href=base_href)
    asyncio.run(html_to_pdf(html, out_pdf))


if __name__ == '__main__':
    main()
