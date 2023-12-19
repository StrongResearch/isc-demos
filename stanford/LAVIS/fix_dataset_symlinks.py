import os

os.makedirs(".cache/lavis/vg", exist_ok=True)
os.makedirs(".cache/lavis/sbu_captions", exist_ok=True)
os.makedirs(".cache/lavis/conceptual_caption", exist_ok=True)

try:
    os.symlink("/mnt/.node1/Open-Datasets/vg/VG_100K/", ".cache/lavis/vg/images")
except FileExistsError:
    pass

with open(".cache/lavis/vg/vg_caption.json", "rt") as fin:
    with open(".cache/lavis/vg/vg_caption.json.out", "wt") as fout:
        for line in fin:
            fout.write(line.replace('/export/share/datasets/vision/visual-genome/image/', ''))
        
os.replace(".cache/lavis/vg/vg_caption.json.out", ".cache/lavis/vg/vg_caption.json")

try:
    os.symlink("/mnt/.node1/Open-Datasets/coco/", ".cache/lavis/coco_gt/images")
except FileExistsError:
    pass

try:
    os.symlink("/mnt/.node1/Open-Datasets/sbu/dataset/", ".cache/lavis/sbu_captions/images")
except FileExistsError:
    pass

try:
    os.symlink("/mnt/.node1/Open-Datasets/cc/cc3m/", ".cache/lavis/conceptual_caption/images_3m")
except FileExistsError:
    pass

try:
    os.symlink("/mnt/.node1/Open-Datasets/cc/cc12m/", ".cache/lavis/conceptual_caption/images_12m")
except FileExistsError:
    pass