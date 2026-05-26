import sys
from pathlib import Path
import base64

md_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('REPORT_SUBMISSION.md')
imgs = ['bler_vs_snr.png', 'learned_weights_hist.png']
md = md_path.read_text(encoding='utf8')
for img in imgs:
    if img in md:
        p = Path(img)
        if p.exists():
            b = p.read_bytes()
            uri = 'data:image/png;base64,' + base64.b64encode(b).decode('ascii')
            md = md.replace(f'({img})', f'({uri})')
        else:
            print('Image not found:', img)
md_path.write_text(md, encoding='utf8')
print('Done')
