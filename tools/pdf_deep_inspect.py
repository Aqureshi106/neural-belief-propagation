import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else 'REPORT_SUBMISSION.pdf'

print('FILE:', path)

# Raw byte searches
with open(path, 'rb') as f:
    data = f.read()

jpeg_sig = b'\xff\xd8\xff'
png_sig = b'\x89PNG\r\n\x1a\n'
print('RAW SIGNATURES:')
print('  JPEG occurrences:', data.count(jpeg_sig))
print('  PNG occurrences:', data.count(png_sig))

# Try PyPDF2 inspection
try:
    from PyPDF2 import PdfReader
    reader = PdfReader(path)
    print('PDF pages:', len(reader.pages))
    total_images = 0
    filters = Counter()
    xobj_counts = 0
    for pnum, page in enumerate(reader.pages, start=1):
        res = page.get('/Resources')
        if not res:
            continue
        xobj = res.get('/XObject') or res.get('/XObject')
        if not xobj:
            # PyPDF2 sometimes nests resources differently
            try:
                xobj = page['/Resources']['/XObject']
            except Exception:
                xobj = None
        if not xobj:
            continue
        for name, obj in xobj.items():
            xobj_counts += 1
            try:
                subtype = obj.get('/Subtype')
            except Exception:
                subtype = obj['/Subtype'] if '/Subtype' in obj else None
            if subtype and subtype == '/Image':
                total_images += 1
                f = obj.get('/Filter')
                filters[str(f)] += 1
    print('XObject entries found (approx):', xobj_counts)
    print('Image XObjects detected:', total_images)
    print('Filters summary:', dict(filters))
except Exception as e:
    print('PyPDF2 inspection failed:', repr(e))

# Heuristic: search for common image stream filters in raw bytes
for marker in [b'/Filter /DCTDecode', b'/Filter /FlateDecode', b'/Filter /JPXDecode']:
    print(marker.decode('latin1'), 'count:', data.count(marker))

# Small sample: print first occurrence of an image object header
idx = data.find(b'/Subtype /Image')
if idx != -1:
    start = max(0, idx-80)
    print('\nSample context around /Subtype /Image:')
    print(data[start:idx+160].replace(b'\n', b'\\n')[:1000])
else:
    print('\nNo literal "/Subtype /Image" found in raw bytes.')
