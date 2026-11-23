from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
rows=json.loads((root/'reproduction/manifest.json').read_text())
for d in rows:
 p=root/d['path'];h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 assert h.hexdigest()==d['sha256'],p
print('Verified',len(rows),'files')
