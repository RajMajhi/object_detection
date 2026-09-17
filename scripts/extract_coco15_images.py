import json
import os
import zipfile

ANNOTATION_FILE = "datasets/coco15/instances_train2017.json"
ZIP_FILE = "datasets/coco15/train2017.zip"
OUTPUT_DIR = "datasets/coco15/images/train"

SELECTED_CLASSES = {
    1, 2, 3, 4, 6, 8, 10, 11, 13, 15, 62, 63, 64, 85, 72
}


def main():
    print("=" * 60)
    print("COCO15 SELECTIVE IMAGE EXTRACTION")
    print("=" * 60)

    print(f"Loading annotations: {ANNOTATION_FILE}")

    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    # Find images containing at least one selected class
    selected_image_ids = set()

    for ann in data["annotations"]:
        if ann["category_id"] in SELECTED_CLASSES:
            selected_image_ids.add(ann["image_id"])

    # Map image ID -> filename
    image_id_to_filename = {
        image["id"]: image["file_name"]
        for image in data["images"]
        if image["id"] in selected_image_ids
    }

    print(f"Eligible images: {len(selected_image_ids)}")
    print(f"Images with filenames: {len(image_id_to_filename)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nOpening archive: {ZIP_FILE}")

    with zipfile.ZipFile(ZIP_FILE, "r") as z:
        members = set(z.namelist())

        extracted = 0
        skipped = 0
        missing = 0

        for image_id, filename in image_id_to_filename.items():

            archive_path = f"train2017/{filename}"
            output_path = os.path.join(OUTPUT_DIR, filename)

            if os.path.exists(output_path):
                skipped += 1
                continue

            if archive_path not in members:
                print(f"Missing from archive: {archive_path}")
                missing += 1
                continue

            z.extract(archive_path, OUTPUT_DIR)

            # z.extract creates OUTPUT_DIR/train2017/filename
            extracted_path = os.path.join(
                OUTPUT_DIR,
                "train2017",
                filename
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            os.replace(extracted_path, output_path)

            # Remove empty train2017 directory if possible
            train_dir = os.path.join(OUTPUT_DIR, "train2017")
            if os.path.isdir(train_dir) and not os.listdir(train_dir):
                os.rmdir(train_dir)

            extracted += 1

            if extracted % 1000 == 0:
                print(
                    f"Extracted: {extracted}/{len(image_id_to_filename)} | "
                    f"Skipped: {skipped} | "
                    f"Missing: {missing}"
                )

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Extracted : {extracted}")
    print(f"Skipped   : {skipped}")
    print(f"Missing   : {missing}")
    print(f"Output    : {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
