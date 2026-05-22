# Food_Freshness_Categorizer

# Quickstart

Setup our repository:
```
git clone https://github.com/Timbakimbo/Food_Freshness_Categorizer
cd Food_Freshness_Categorizer
```

Setup a virtual environment for development purposes:
```
python -m venv .venv
source venv/bin/activate    # Windows: venv\Scripts\activate
```

Install all dependencies:
```
pip install -r requirements.txt
```
# Usefull stuff
## Scripts
``move_rename_images.py``:
Move files from a ``SRC`` to ``DST`` directory, replacing each file's number with the next available one based on the highest number already present in the destination — preserving the original filename and extension.

``image_resize.py``:
Recursively processes all JPEG, PNG, and HEIC images from a source directory, resizes them to a centered 224×224 square (scale-then-pad, no distortion) with EXIF orientation correction, and saves them as compressed progressive JPEGs to a mirrored destination directory structure.

## CLI
TLDR: ``for f in data/train/edible/*.jpg; do mv -- "$f" "${f%.jpg}.jpeg"; done``

# Structure

```
root/
│
├── data/
│   ├── raw/
│   │    ├── unsorted/
│   │    ├── dataset_cat1/
│   │    │      ├── <categories>/
│   │    │      │       ├── <files>
│   │    │      ├── .../
│   │    │      │       ├── ...
│   │    ├── dataset_cat2/
│   │    │      ├── <categories>/
│   │    │      │       ├── <files>
│   ├── processed/
│   ├── train/
│   │   ├── edible/
│   │   │    ├── <categories>/
│   │   │    │      ├── <files>
│   │   │    ├── .../
│   │   │    │      ├── ...
│   │   ├── non_edible/
│   │   │    ├── <categories>/
│   │   │    │      ├── <files>
│   │   │    ├── .../
│   │   │    │      ├── ...
│   ├── val/
│   │   ├── edible/
│   │   │    ├── <categories>/
│   │   │    │      ├── <files>
│   │   │    ├── .../
│   │   │    │      ├── ...
│   │   ├── non_edible/
│   │   │    ├── <categories>/
│   │   │    │      ├── <files>
│
├── logs/
│   ├── ...
│
├── doc/
│   ├── ...
│
├── models/
│   ├── pepper_classifier.keras
│   ├── ...
│
├── src/
│   ├── data_loader.py
│   ├── train.py
│   ├── predict.py
│   ├── ...
│
├── app/
│   ├── streamlit.py
│
├── config/
│   ├── config.yaml
│   ├── ...
│
├── main.py
├── requirements.txt
└── README.md
```
