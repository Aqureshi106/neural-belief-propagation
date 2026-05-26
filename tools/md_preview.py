import sys
import asyncio
from pathlib import Path
import markdown
from pyppeteer import launch

async def render(md_path, out_png):
    md_text = Path(md_path).read_text(encoding='utf8')
    base_href = Path(md_path).parent.resolve().as_uri()
    html_body = markdown.markdown(md_text, extensions=["fenced_code", "codehilite"])
    css = """
    <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 40px; }
    pre { background: #f6f8fa; padding: 10px; border-radius: 4px; }
    code { font-family: Consolas, monospace; }
    h1,h2,h3 { color: #111827 }
    </style>
    """
    html = f"<html><head><meta charset=\"utf-8\"><base href=\"{base_href}\">{css}</head><body>{html_body}</body></html>"
    # Prefer existing Chrome/Edge to avoid downloading Chromium
    candidates = [
        r"C:/Program Files/Google/Chrome/Application/chrome.exe",
        r"C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
        r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
        r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    ]
    browser = None
    for p in candidates:
        if Path(p).exists():
            try:
                browser = await launch({'executablePath': p, 'args': ['--no-sandbox']})
                break
            except Exception:
                browser = None
    if browser is None:
        browser = await launch({'args': ['--no-sandbox']})
    page = await browser.newPage()
    await page.setViewport({'width': 1200, 'height': 1600})
    await page.setContent(html)
    try:
        await page.waitForFunction('document.readyState === "complete"', timeout=5000)
    except Exception:
        await asyncio.sleep(0.5)
    await page.screenshot({'path': out_png, 'fullPage': True})
    await browser.close()

if __name__ == '__main__':
    md = sys.argv[1] if len(sys.argv) > 1 else 'REPORT_SUBMISSION.md'
    out = sys.argv[2] if len(sys.argv) > 2 else 'REPORT_SUBMISSION_preview.png'
    asyncio.run(render(md, out))
