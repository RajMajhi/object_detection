import json
from collections import Counter
from pathlib import Path

COCO_TO_TRAIN = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    6: 4,
    8: 5,
    10: 6,
    11: 7,
    13: 8,
    15: 9,
    62: 10,
    63: 11,
    64: 12,
    85: 13,
    72: 14,
}

CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "chair",
    "couch",
    "potted plant",
    "clock",
    "tv",
]


def analyze(json_path):
    print(f"\nLoading: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    selected_categories = set(COCO_TO_TRAIN.keys())

    image_ids = set()
    annotation_counts = Counter()

    for ann in data["annotations"]:
        category_id = ann["category_id"]

        if category_id in selected_categories:
            image_ids.add(ann["image_id"])
            annotation_counts[category_id] += 1

    print(f"Total COCO images in JSON: {len(data['images']):,}")
    print(f"Images containing selected classes: {len(image_ids):,}")
    print(f"Selected annotations: {sum(annotation_counts.values()):,}")

    print("\nAnnotations by class:")
    for coco_id, train_id in COCO_TO_TRAIN.items():
        print(
            f"{train_id:2d}  {CLASS_NAMES[train_id]:15s} "
            f"{annotation_counts[coco_id]:,}"
        )

    return image_ids


base = Path("datasets/coco15")

train_ids = analyze(base / "instances_train2017.json")
val_ids = analyze(base / "instances_val2017.json")

print("\n" + "=" * 50)
print(f"TRAIN images needed: {len(train_ids):,}")
print(f"VAL images needed:   {len(val_ids):,}")
print("=" * 50)
