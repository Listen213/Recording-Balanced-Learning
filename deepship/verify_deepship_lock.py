from __future__ import annotations
import argparse
import torch
from deepship_uatr.common import load_json, sha256_file


def verify(lock_path):
    lock=load_json(lock_path)
    paths=lock['paths']
    for key,path in paths.items():
        want=lock[key+'_sha256']
        got=sha256_file(path)
        if got != want:
            raise RuntimeError(f'LOCK MISMATCH {key}: {path}\nwant={want}\ngot ={got}')

    sup=torch.load(paths['support_cache'],map_location='cpu',weights_only=False)
    tst=torch.load(paths['test_cache'],map_location='cpu',weights_only=False)
    if sup.get('manifest_sha256') != lock['support_manifest_sha256']:
        raise RuntimeError('support cache internal manifest hash mismatch')
    if tst.get('manifest_sha256') != lock['test_manifest_sha256']:
        raise RuntimeError('test cache internal manifest hash mismatch')
    return lock


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',default='deepship_lock/DEEPSHIP_LOCK.json')
    a=ap.parse_args()
    verify(a.lock)
    print('DEEPSHIP V3 LOCK VERIFIED')


if __name__=='__main__':
    main()
