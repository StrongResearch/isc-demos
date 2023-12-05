import os

os.symlink("/mnt/.node1/Open-Datasets/vg/VG_100K/", ".cache/vg/images")

with open(".cache/lavis/vg/annotations/vg_caption.json", "rt") as fin:
    with open(".cache/lavis/vg/annotations/vg_caption.json.out", "wt") as fout:
        for line in fin:
            fout.write(line.replace('/export/share/datasets/vision/visual-genome/image/', ''))
        
os.replace(".cache/lavis/vg/annotations/vg_caption.json.out", ".cache/lavis/vg/annotations/vg_caption.json")

os.symlink("/mnt/.node1/Open-Datasets/sbu/dataset/", ".cache/sbu_captions/images")