import os
from collections import Counter

DATASET_ROOT = "datasets/coco15"

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

EXPECTED_IMAGES = {
    "train": 82803,
    "val": 3527,
}


def validate_split(split):
    image_dir = os.path.join(DATASET_ROOT, "images", split)
    label_dir = os.path.join(DATASET_ROOT, "labels", split)

    images = {
        os.path.splitext(f)[0]
        for f in os.listdir(image_dir)
        if f.lower().endswith(".jpg")
    }

    labels = {
        os.path.splitext(f)[0]
        for f in os.listdir(label_dir)
        if f.endswith(".txt")
    }

    print("\n" + "=" * 60)
    print(split.upper())
    print("=" * 60)

    print(f"Images          : {len(images)}")
    print(f"Labels          : {len(labels)}")
    print(f"Expected images : {EXPECTED_IMAGES[split]}")

    missing_labels = images - labels
    orphan_labels = labels - images

    print(f"Missing labels  : {len(missing_labels)}")
    print(f"Orphan labels   : {len(orphan_labels)}")

    invalid = []
    class_counts = Counter()
    annotation_count = 0

    for filename in labels:
        path = os.path.join(label_dir, filename + ".txt")

        with open(path, "r") as f:
            for line_number, line in enumerate(f, 1):

                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:
                    invalid.append(
                        (filename, line_number, "expected 5 fields")
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    invalid.append(
                        (filename, line_number, "non-numeric value")
                    )
                    continue

                if not 0 <= class_id < len(CLASS_NAMES):
                    invalid.append(
                        (filename, line_number, "invalid class ID")
                    )
                    continue

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    invalid.append(
                        (filename, line_number, "invalid bounding box")
                    )
                    continue

                class_counts[class_id] += 1
                annotation_count += 1

    print(f"Annotations     : {annotation_count}")
    print(f"Invalid labels  : {len(invalid)}")

    print("\nClass distribution:")

    for class_id, name in enumerate(CLASS_NAMES):
        print(
            f"{class_id:2d}  "
            f"{name:15s} "
            f"{class_counts[class_id]:8d}"
        )

    if len(images) != EXPECTED_IMAGES[split]:
        print("\nWARNING: image count does not match expected count.")

    if missing_labels:
        print("\nWARNING: some images do not have labels.")

    if orphan_labels:
        print("\nWARNING: some labels do not have corresponding images.")

    if invalid:
        print("\nWARNING: invalid labels detected.")

    if (
        len(images) == EXPECTED_IMAGES[split]
        and not missing_labels
        and not orphan_labels
        and not invalid
    ):
        print("\nSTATUS: PASS")


def main():
    print("=" * 60)
    print("COCO15 DATASET VALIDATION")
    print("=" * 60)

    validate_split("train")
    validate_split("val")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
