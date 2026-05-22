import os, re, shutil

SRC = "/mnt/g/FAU/ML4B/sorted/"
DST = "data/train/edible/"

nums = [int(m.group()) for f in os.listdir(DST) if (m := re.search(r"\d+", f))]
counter = max(nums, default=0) + 1
 
for f in sorted(os.listdir(SRC)):
    if os.path.isfile(os.path.join(SRC, f)):
        name = re.sub(r"\d+", str(counter), os.path.splitext(f)[0])
        ext = os.path.splitext(f)[1]
        shutil.move(f"{SRC}/{f}", f"{DST}/{name}{ext}")
        counter += 1
 
