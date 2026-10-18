from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from deepship_uatr.common import sha256_file, save_json, load_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='DEEPSHIP_CONFIG.json')
    ap.add_argument('--manifest-dir',default='manifests_deepship_50')
    ap.add_argument('--cache-dir',default='cache/beats_deepship_50')
    ap.add_argument('--out',default='deepship_lock/DEEPSHIP_LOCK.json')
    a=ap.parse_args()

    cfg=load_json(a.config)
    md=Path(a.manifest_dir); cd=Path(a.cache_dir)
    sup_m=md/'support_pool.json'; test_m=md/'test.json'
    sup_c=cd/'support_pool.pt'; test_c=cd/'test.pt'
    required=[
        Path(a.config), sup_m, test_m, sup_c, test_c,
        md/'selected_exact_components.json', md/'leakage_audit.json',
        md/'split_summary.json', md/'date_split_diagnostics.json',
        md/'acoustic_similarity_diagnostics.json', md/'exact_duplicate_edges.json',
    ]
    for p in required:
        if not p.exists(): raise FileNotFoundError(p)

    sup_obj=torch.load(sup_c,map_location='cpu',weights_only=False)
    test_obj=torch.load(test_c,map_location='cpu',weights_only=False)
    sup_mh=sha256_file(sup_m); test_mh=sha256_file(test_m)
    if sup_obj.get('manifest_sha256') != sup_mh:
        raise RuntimeError('support cache does not match support manifest')
    if test_obj.get('manifest_sha256') != test_mh:
        raise RuntimeError('test cache does not match test manifest')

    audit=load_json(md/'leakage_audit.json')
    if not audit.get('hard_leakage_clean',False):
        raise RuntimeError(f'DeepShip hard leakage audit is not clean: {audit}')
    if any([
        audit.get('recording_id_overlap',0),
        audit.get('wav_sha256_overlap',0),
        audit.get('exact_component_overlap',0),
    ]):
        raise RuntimeError(f'DeepShip hard leakage audit failed: {audit}')

    paths={
        'config':str(Path(a.config)),
        'support_manifest':str(sup_m),
        'test_manifest':str(test_m),
        'support_cache':str(sup_c),
        'test_cache':str(test_c),
        'components':str(md/'selected_exact_components.json'),
        'leakage_audit':str(md/'leakage_audit.json'),
        'split_summary':str(md/'split_summary.json'),
        'date_split_diagnostics':str(md/'date_split_diagnostics.json'),
        'acoustic_similarity_diagnostics':str(md/'acoustic_similarity_diagnostics.json'),
        'exact_duplicate_edges':str(md/'exact_duplicate_edges.json'),
    }
    lock={
        'lock_version':3,
        'dataset':'DeepShip',
        'experiment_role':'external_validation',
        'method_locked_on':'ShipsEar',
        'protocol_note':'Exact SHA256 duplicates are hard components; acoustic similarity is diagnostic only.',
        'config':cfg,
        'paths':paths,
    }
    for key,path in paths.items():
        lock[key+'_sha256']=sha256_file(path)

    save_json(lock,a.out)
    print(json.dumps(lock,ensure_ascii=False,indent=2))
    print('DEEPSHIP V3 EXTERNAL-VALIDATION PROTOCOL FROZEN.')
    print('Do not change ShipsEar-locked method hyperparameters after this point.')


if __name__=='__main__':
    main()
