import sys, importlib

mods=["numpy","scipy","sklearn","torch","torchaudio","tqdm"]
print("Python:",sys.version.replace("\n"," "))
for m in mods:
    x=importlib.import_module(m)
    print(f"{m}: {getattr(x,'__version__','ok')}")
import torch
print("CUDA available:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA:",torch.version.cuda)
    print("GPU:",torch.cuda.get_device_name(0))
