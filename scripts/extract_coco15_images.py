import json
import os
import sys
import zipfile

SELECTED_CLASSES = {
    1, 2, 3, 4, 6, 8, 10, 11, 13, 15, 62, 63, 64, 85, 72
}

CONFIGS = {
    "train": {
        "annotation": "datasets/coco15/instances_train2017.json",
        "zip": "datasets/coco15/train2017.zip",
        "output": "datasets/coco15/images/train",
        "folder": "train2017",
    },
    "val": {
        "annotation": "datasets/coco15/instances_val2017.json",
        "zip": "datasets/coco15/val2017.zip",
        "output": "datasets/coco15/images/val",
        "folder": "val2017",
    },
}


def extract_images(split):
    config = CONFIGS[split]

    annotation_file = config["annotation"]
    zip_file = config["zip"]
    output_dir = config["output"]
    archive_folder = config["folder"]

    print("=" * 60)
    print(f"COCO15 {split.upper()} IMAGE EXTRACTION")
    print("=" * 60)

    print(f"Loading annotations: {annotation_file}")

    with open(annotation_file, "r") as f:
        data = json.load(f)

    selected_image_ids = set()

    for ann in data["annotations"]:
        if ann["category_id"] in SELECTED_CLASSES:
            selected_image_ids.add(ann["image_id"])

    image_id_to_filename = {
        image["id"]: image["file_name"]
        for image in data["images"]
        if image["id"] in selected_image_ids
    }

    print(f"Eligible images: {len(selected_image_ids)}")
    print(f"Images with filenames: {len(image_id_to_filename)}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nOpening archive: {zip_file}")

    with zipfile.ZipFile(zip_file, "r") as z:
        members = set(z.namelist())

        extracted = 0
        skipped = 0
        missing = 0

        for image_id, filename in image_id_to_filename.items():

            archive_path = f"{archive_folder}/{filename}"
            output_path = os.path.join(output_dir, filename)

            if os.path.exists(output_path):
                skipped += 1
                continue

            if archive_path not in members:
                print(f"Missing from archive: {archive_path}")
                missing += 1
                continue

            z.extract(archive_path, output_dir)

            extracted_path = os.path.join(
                output_dir,
                archive_folder,
                filename
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            os.replace(extracted_path, output_path)

            extracted += 1

            if extracted % 1000 == 0:
                print(
                    f"Extracted: {extracted}/{len(image_id_to_filename)} | "
                    f"Skipped: {skipped} | "
                    f"Missing: {missing}"
                )

    print("\n" + "=" * 60)
    print(f"{split.upper()} EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Extracted : {extracted}")
    print(f"Skipped   : {skipped}")
    print(f"Missing   : {missing}")
    print(f"Output    : {output_dir}")
    print("=" * 60)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CONFIGS:
        print("Usage:")
        print("  python scripts/extract_coco15_images.py train")
        print("  python scripts/extract_coco15_images.py val")
        sys.exit(1)

    extract_images(sys.argv[1])


if __name__ == "__main__":
    main()

