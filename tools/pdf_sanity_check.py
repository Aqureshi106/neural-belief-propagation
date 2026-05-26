import sys
import os

path = sys.argv[1] if len(sys.argv) > 1 else 'REPORT_SUBMISSION.pdf'
out = {}

out['path'] = path
out['exists'] = os.path.exists(path)
if not out['exists']:
    print('EXISTS: False')
    sys.exit(2)

out['size_bytes'] = os.path.getsize(path)

# Try to get reliable page count via PyPDF2 if available
page_count = None
try:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    page_count = len(reader.pages)
    out['page_count_py'] = page_count
except Exception:
    out['page_count_py'] = None

# Fallback: count occurrences of '/Type /Page' in raw bytes
with open(path, 'rb') as f:
    data = f.read()

out['page_count_raw'] = data.count(b'/Type /Page')

png_sig = b'\x89PNG\r\n\x1a\n'
out['png_signature_count'] = data.count(png_sig)
out['png_present'] = out['png_signature_count'] > 0

# Look for referenced figure filenames
filenames = [b'bler_vs_snr.png', b'learned_weights_hist.png']
name_hits = {fn.decode(): (fn in data) for fn in filenames}
out['filename_hits'] = name_hits

# Print a concise, human-readable report
print(f"EXISTS: {out['exists']}")
print(f"SIZE_BYTES: {out['size_bytes']}")
if out['page_count_py'] is not None:
    print(f"PAGE_COUNT (PyPDF2): {out['page_count_py']}")
print(f"PAGE_COUNT (raw '/Type /Page' count): {out['page_count_raw']}")
print(f"PNG_SIGNATURE_COUNT: {out['png_signature_count']}")
print(f"PNG_PRESENT: {out['png_present']}")
for name, hit in name_hits.items():
    print(f"FILENAME_IN_PDF: {name}: {hit}")

# Exit 0 for success
sys.exit(0)
