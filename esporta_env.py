#!/usr/bin/env python
"""
esporta_env.py — genera un requirements installabile su Windows a partire
dall'ambiente Linux in cui gira.

PERCHE' NON `conda env export`. Quel comando scrive i BUILD specifici della
piattaforma (`=py310h1234_0`), che su Windows non esistono. E `pip freeze`
porta con se' le ruote `nvidia-*` e `triton`, che sono solo Linux: l'install
fallisce a meta' lasciando l'ambiente rotto. Anche `conda-pack` non serve,
perche' impacchetta binari Linux.

COSA FA. Legge le versioni REALMENTE installate dei pacchetti che questo
progetto importa davvero — la lista viene dai sorgenti, non dalla memoria — e
scrive un requirements pulito. PyTorch resta fuori: va installato dall'indice
con la variante CUDA giusta, e il comando esatto viene stampato in fondo.

USO, sul server di partenza:

    python esporta_env.py
    # poi copia requirements-windows.txt sull'altra macchina
"""

import importlib.metadata as md
import sys

# import -> nome della distribuzione su PyPI, dove differiscono
NOMI = {
    "PIL": "pillow", "cv2": "opencv-python", "sklearn": "scikit-learn",
    "yaml": "PyYAML", "ee": "earthengine-api", "skimage": "scikit-image",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
    "huggingface_hub": "huggingface-hub",
}

# Il nucleo che serve ad addestrare LENTE.
NUCLEO = ["numpy", "scipy", "sklearn", "PIL", "cv2", "matplotlib", "tqdm",
          "yaml", "h5py", "tifffile", "transformers", "timm",
          "segmentation_models_pytorch", "albumentations", "huggingface_hub",
          "ultralytics"]

# Il resto del progetto: acquisizione dati e pipeline Sen12. Separati perche'
# rasterio e pyproj su Windows sono i piu' capricciosi, e se il tuo scopo e'
# solo addestrare puoi saltarli.
EXTRA = ["rasterio", "pyproj", "xarray", "geemap", "ee", "ollama"]

# Fuori dal file per costruzione: torch va dall'indice CUDA, e questi tre non
# esistono affatto per Windows.
ESCLUSI = {"torch", "torchvision", "torchaudio", "triton", "pytorch-triton"}


def versione(imp):
    nome = NOMI.get(imp, imp)
    try:
        return nome, md.version(nome)
    except md.PackageNotFoundError:
        return nome, None


def main():
    righe, mancanti = [], []
    for gruppo, titolo in ((NUCLEO, "nucleo: addestramento LENTE"),
                           (EXTRA, "extra: acquisizione dati e Sen12")):
        righe.append(f"# ---- {titolo} ----")
        for imp in gruppo:
            nome, v = versione(imp)
            if v is None:
                mancanti.append(nome)
                righe.append(f"# {nome}   NON INSTALLATO qui")
            elif nome.lower() in ESCLUSI:
                continue
            else:
                righe.append(f"{nome}=={v}")
        righe.append("")

    with open("requirements-windows.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(righe))

    print(f"Scritto requirements-windows.txt  ({sys.version.split()[0]})")
    if mancanti:
        print(f"  non installati qui, commentati nel file: {', '.join(mancanti)}")

    try:
        import torch
        cu = torch.version.cuda
        tag = "cu" + cu.replace(".", "") if cu else "cpu"
        import torchvision
        tv = f" torchvision=={torchvision.__version__.split('+')[0]}"
    except Exception:
        print("\n  torch non importabile: il comando qui sotto va completato a mano")
        return

    print(f"""
=========================================================================
SULLA MACCHINA WINDOWS, nell'ordine — l'ordine conta

  conda create -n landslide python={sys.version_info.major}.{sys.version_info.minor} -y
  conda activate landslide

  # 1) PRIMA torch, dall'indice CUDA. Se lo installi dopo, ultralytics
  #    si tira dietro la versione CPU e ti ritrovi senza GPU senza errori.
  pip install torch=={torch.__version__.split('+')[0]}{tv} \\
      --index-url https://download.pytorch.org/whl/{tag}

  # 2) poi il resto
  pip install -r requirements-windows.txt

  # 3) verifica: deve stampare True e DUE schede
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); \\
             print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
=========================================================================""")


if __name__ == "__main__":
    main()
