"""Convert an existing YOLO split to COCO detection annotations."""

import argparse
import json
from pathlib import Path

from PIL import Image


def convert(split, output):
    classes = (split / "classes.txt").read_text(encoding="utf-8-sig").splitlines()
    classes = [name.strip() for name in classes if name.strip()]
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("classes.txt must list unique class names in label-ID order.")
    if not (split / "labels").is_dir():
        raise FileNotFoundError(split / "labels")

    categories = [{"id": index, "name": name} for index, name in enumerate(classes)]
    images, annotations = [], []
    for path in sorted((split / "images").iterdir()):
        if not path.is_file() or path.suffix.lower() not in Image.registered_extensions():
            continue
        with Image.open(path) as image:
            width, height = image.size
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": path.name, "width": width, "height": height})

        label_path = split / "labels" / path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                category, *box = line.split()
                category_id = int(category)
                cx, cy, bw, bh = map(float, box)
                if not (0 <= category_id < len(categories)
                        and 0 <= cx <= 1 and 0 <= cy <= 1
                        and 0 < bw <= 1 and 0 < bh <= 1):
                    raise ValueError("Class ID or normalized bounding box is out of range.")
            except ValueError as error:
                raise ValueError(f"{label_path}:{line_number}: invalid YOLO label: {line}") from error

            left = max(0.0, (cx - bw / 2) * width)
            top = max(0.0, (cy - bh / 2) * height)
            right = min(width, (cx + bw / 2) * width)
            bottom = min(height, (cy + bh / 2) * height)
            box_width, box_height = right - left, bottom - top
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [left, top, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })

    if not images:
        raise ValueError(f"No supported images found in {split / 'images'}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "images": images, "annotations": annotations, "categories": categories,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output} ({len(images)} images, {len(annotations)} boxes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, help="Directory containing images/, labels/, and classes.txt")
    parser.add_argument("output", type=Path, help="Output COCO JSON path")
    args = parser.parse_args()
    convert(args.split, args.output)
