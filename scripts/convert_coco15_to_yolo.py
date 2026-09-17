import json
import os
from collections import Counter

DATASET_ROOT = "datasets/coco15"

SELECTED_CLASSES = {
    1: 0,    # person
    2: 1,    # bicycle
    3: 2,    # car
    4: 3,    # motorcycle
    6: 4,    # bus
    8: 5,    # truck
    10: 6,   # traffic light
    11: 7,   # fire hydrant
    13: 8,   # stop sign
    15: 9,   # bench
    62: 10,  # chair
    63: 11,  # couch
    64: 12,  # potted plant
    85: 13,  # clock
    72: 14,  # tv
}

SPLITS = {
    "train": {
        "json": os.path.join(DATASET_ROOT, "instances_train2017.json"),
        "image_dir": os.path.join(DATASET_ROOT, "images/train"),
        "label_dir": os.path.join(DATASET_ROOT, "labels/train"),
    },
    "val": {
        "json": os.path.join(DATASET_ROOT, "instances_val2017.json"),
        "image_dir": os.path.join(DATASET_ROOT, "images/val"),
        "label_dir": os.path.join(DATASET_ROOT, "labels/val"),
    },
}


def convert_split(split, config):
    print("\n" + "=" * 60)
    print(f"CONVERTING {split.upper()}")
    print("=" * 60)

    with open(config["json"], "r") as f:
        data = json.load(f)

    os.makedirs(config["label_dir"], exist_ok=True)

    # Image ID -> image information
    images = {
        image["id"]: image
        for image in data["images"]
    }

    # Image ID -> annotations
    annotations_by_image = {}

    for ann in data["annotations"]:
        category_id = ann["category_id"]

        if category_id not in SELECTED_CLASSES:
            continue

        # Ignore crowd annotations
        if ann.get("iscrowd", 0) == 1:
            continue

        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    class_counts = Counter()
    total_annotations = 0
    images_with_labels = 0
    missing_images = 0
    invalid_boxes = 0

    for image_id, annotations in annotations_by_image.items():

        if image_id not in images:
            continue

        image_info = images[image_id]

        width = image_info["width"]
        height = image_info["height"]
        filename = image_info["file_name"]

        image_path = os.path.join(
            config["image_dir"],
            filename
        )

        # Make sure the extracted image exists
        if not os.path.exists(image_path):
            missing_images += 1
            continue

        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(
            config["label_dir"],
            label_filename
        )

        lines = []

        for ann in annotations:

            category_id = ann["category_id"]
            class_id = SELECTED_CLASSES[category_id]

            x, y, box_width, box_height = ann["bbox"]

            # Validate COCO bounding box
            if box_width <= 0 or box_height <= 0:
                invalid_boxes += 1
                continue

            # Clip bounding box to image boundaries
            x1 = max(0.0, x)
            y1 = max(0.0, y)

            x2 = min(float(width), x + box_width)
            y2 = min(float(height), y + box_height)

            clipped_width = x2 - x1
            clipped_height = y2 - y1

            if clipped_width <= 0 or clipped_height <= 0:
                invalid_boxes += 1
                continue

            # Convert to YOLO format
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0

            x_center /= width
            y_center /= height
            clipped_width /= width
            clipped_height /= height

            # Final sanity check
            if not (
                0.0 <= x_center <= 1.0
                and 0.0 <= y_center <= 1.0
                and 0.0 < clipped_width <= 1.0
                and 0.0 < clipped_height <= 1.0
            ):
                invalid_boxes += 1
                continue

            lines.append(
                f"{class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{clipped_width:.6f} "
                f"{clipped_height:.6f}"
            )

            class_counts[class_id] += 1
            total_annotations += 1

        if lines:
            with open(label_path, "w") as f:
                f.write("\n".join(lines) + "\n")

            images_with_labels += 1

    print(f"Images with labels : {images_with_labels}")
    print(f"Annotations        : {total_annotations}")
    print(f"Missing images     : {missing_images}")
    print(f"Invalid boxes      : {invalid_boxes}")

    print("\nAnnotations by training class:")

    class_names = [
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

    for class_id, name in enumerate(class_names):
        print(
            f"{class_id:2d}  "
            f"{name:15s} "
            f"{class_counts[class_id]:7d}"
        )


def main():
    print("=" * 60)
    print("COCO 15-CLASS → YOLO CONVERTER")
    print("=" * 60)

    for split, config in SPLITS.items():
        convert_split(split, config)

    print("\n" + "=" * 60)
    print("CONVERSION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()
