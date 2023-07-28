from pathlib import Path
import shutil
import pandas as pd
import plac
from tqdm import tqdm

def convert_split(kaggle_imagenet_root: Path, split: str):
    split_dir = kaggle_imagenet_root / "ILSVRC" / "Data" / "CLS-LOC" / split
    split_labels_path = kaggle_imagenet_root / f"LOC_{split}_solution.csv"
    labels = pd.read_csv(split_labels_path).set_index("ImageId", drop=True)
    labels: dict[str, str] = labels.PredictionString.apply(lambda s: s.split()[0]).to_dict()
    new_dir = split_dir.parent / f"{split}_converted"
    for example in tqdm(list(split_dir.iterdir())):
        example_label = labels[example.stem]
        example_new_path = new_dir / example_label / example.name
        example_new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example, example_new_path)



@plac.pos('kaggle_imagenet_root', "path to imagenet", type=Path)
def main(kaggle_imagenet_root: Path):
    assert kaggle_imagenet_root.exists()
    for split in ['val']:
        convert_split(kaggle_imagenet_root, split)

if __name__ == "__main__":
    plac.call(main)
