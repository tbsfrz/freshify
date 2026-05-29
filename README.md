# Freshify

## Quicktest

### Class: edible
```
EASY: python main.py predict data/raw/dataset_cat1/paprika/paprika12.jpeg
HARD: python main.py predict data/raw/dataset_cat1/banane/banane17.jpeg
```

### Class: non_edible
```
EASY: python main.py predict data/raw/dataset_cat2/banane/banane1.png
HARD: python main.py predict data/raw/dataset_cat2/paprika/paprika19.jpeg
```

## Quickstart

Setup the repository:
```
git clone https://github.com/tbsfrz/freshify
cd freshify
```

Create an environment:
```
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirments.txt
```

Run a local prediction:
```
python main.py predict path/to/image.jpg
```

Train the classifier:
```
python main.py train --data-dir data
```

Run it with Streamlit:
```
streamlit run app/streamlit.py
```

## Useful scripts

`move_rename_images.py`: rename and move files from a source directory to a destination directory, preserving the original extension.

`image_resize.py`: resize images to a centered 224×224 square while preserving aspect ratio and EXIF orientation.