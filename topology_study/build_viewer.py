"""Embed generated geometry into the literal visualization template."""
from pathlib import Path

root = Path(__file__).resolve().parent
template = (root / 'atlas.template.html').read_text(encoding='utf-8')
data = (root / 'atlas_data.json').read_text(encoding='utf-8')
fragment = template.replace('__ATLAS_DATA__', data)
assert len(fragment.encode('utf-8')) < 1_000_000
assert '__ATLAS_DATA__' not in fragment
(root / 'contour-atlas.html').write_text(fragment, encoding='utf-8')
print(root / 'contour-atlas.html')
