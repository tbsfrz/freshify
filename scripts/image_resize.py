from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
# optional: enable HEIC support if nötig
from pillow_heif import register_heif_opener
register_heif_opener()

SRC = Path("/mnt/g/FAU/ML4B/input_images_ganz_große_dateien/")
DST = Path("/mnt/g/FAU/ML4B/output_224/")
DST.mkdir(parents=True, exist_ok=True)
SIZE = 224
QUALITY = 85  # JPEG quality 0-100

def image_resize_save(src_path: Path, dst_path: Path, size=SIZE, quality=QUALITY):
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")  # fix orientation + ensure 8-bit RGB
        w, h = im.size
        # scale so smaller side == size
        scale = size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        im = im.resize((new_w, new_h), Image.LANCZOS)
        # pad to square (center)
        pad_w = max(size - new_w, 0)
        pad_h = max(size - new_h, 0)
        left = pad_w // 2
        top  = pad_h // 2
        if pad_w > 0 or pad_h > 0:
            new_img = Image.new("RGB", (max(new_w, size), max(new_h, size)), (0,0,0))
            new_img.paste(im, (left, top))
            im = new_img
        # final crop just in case (should be exact)
        im = ImageOps.fit(im, (size, size), method=Image.LANCZOS, centering=(0.5,0.5))
        # save as compressed JPEG
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, format="JPEG", quality=quality, optimize=True, progressive=True)

# process all common image files
EXTS = (".jpg", ".jpeg", ".png", ".heic", ".HEIC")
files = [p for p in SRC.rglob("*") if p.suffix.lower() in EXTS]
for p in tqdm(files, desc="Processing"):
    out = DST / p.relative_to(SRC).with_suffix(".jpg")
    try:
        image_resize_save(p, out)
    except Exception as e:
        print("ERROR", p, e)
