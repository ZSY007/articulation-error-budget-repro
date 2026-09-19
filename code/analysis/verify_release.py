"""Verify the release inventory, gzip streams, and optional local PartNet URDFs."""
from pathlib import Path
import argparse, csv, gzip, hashlib

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets',type=Path)
    args=ap.parse_args(); failures=[]; checked=0
    with (ROOT/'manifests/release_inventory.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            p=ROOT/r['path']; checked+=1
            if not p.is_file() or p.stat().st_size!=int(r['bytes']) or sha(p)!=r['sha256']:
                failures.append(r['path'])
    gzip_files=list(ROOT.rglob('*.gz'))
    for p in gzip_files:
        with gzip.open(p,'rb') as f:
            for _ in iter(lambda:f.read(1024*1024),b''):pass
    assets_checked=0
    if args.assets:
        with (ROOT/'manifests/object_manifest.csv').open(encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if not r['urdf_sha256']:continue
                p=args.assets/r['urdf_relative_path'];assets_checked+=1
                if not p.is_file() or sha(p)!=r['urdf_sha256']:failures.append('asset:'+r['object_id'])
    if failures: raise SystemExit('FAILED: '+', '.join(failures))
    print(f'PASS: {checked} inventoried files, {len(gzip_files)} gzip streams, '
          f'{assets_checked} optional URDFs')


if __name__=='__main__':main()
