#!/usr/bin/env python3
import argparse, os, shutil
from pathlib import Path

def link_or_copy(src,dst,copy=False):
    src=Path(src).resolve(); dst=Path(dst)
    if not src.exists(): raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink(): shutil.rmtree(dst)
        else: dst.unlink()
    if copy:
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copy2(src,dst)
    else:
        os.symlink(src,dst,target_is_directory=src.is_dir())
    print(('COPIED' if copy else 'LINKED'),src,'->',dst)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-project',default='/root/autodl-tmp/RIFT_FEWREC_FINAL_FAST_FULL'); ap.add_argument('--copy',action='store_true'); a=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    src=Path(a.source_project)
    link_or_copy(src/'third_party'/'beats',root/'vendor'/'beats_src',a.copy)
    link_or_copy(src/'weights'/'BEATs_iter3_plus_AS2M.pt',root/'vendor'/'BEATs_iter3_plus_AS2M.pt',a.copy)
if __name__=='__main__': main()
