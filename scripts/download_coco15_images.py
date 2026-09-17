import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

IMAGE_IDS_FILE = "datasets/coco15/train_image_ids.txt"
OUTPUT_DIR = "datasets/coco15/images/train"
FAILED_FILE = "datasets/coco15/failed_downloads.txt"

BASE_URL = "http://images.cocodataset.org/train2017"
NUM_WORKERS = 8

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_image(image_id):
    filename = f"{image_id:012d}.jpg"
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Already downloaded
    if os.path.exists(output_path):
        return image_id, "exists"

    url = f"{BASE_URL}/{filename}"

    try:
        urllib.request.urlretrieve(url, output_path)

        # Make sure we didn't create an empty/corrupt file
        if os.path.getsize(output_path) == 0:
            os.remove(output_path)
            return image_id, "failed"

        return image_id, "downloaded"

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)

        return image_id, f"failed: {e}"


def main():
    with open(IMAGE_IDS_FILE, "r") as f:
        image_ids = [int(line.strip()) for line in f if line.strip()]

    print("=" * 60)
    print("COCO15 IMAGE DOWNLOADER")
    print("=" * 60)
    print(f"Images requested : {len(image_ids)}")
    print(f"Output directory : {OUTPUT_DIR}")
    print(f"Workers          : {NUM_WORKERS}")
    print("=" * 60)

    downloaded = 0
    exists = 0
    failed = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(download_image, image_id): image_id
            for image_id in image_ids
        }

        for i, future in enumerate(as_completed(futures), 1):
            image_id, status = future.result()

            if status == "downloaded":
                downloaded += 1
            elif status == "exists":
                exists += 1
            else:
                failed.append(image_id)

            if i % 100 == 0 or i == len(image_ids):
                elapsed = time.time() - start_time
                print(
                    f"Progress: {i}/{len(image_ids)} | "
                    f"Downloaded: {downloaded} | "
                    f"Existing: {exists} | "
                    f"Failed: {len(failed)} | "
                    f"Time: {elapsed/60:.1f} min"
                )

    if failed:
        with open(FAILED_FILE, "w") as f:
            for image_id in sorted(failed):
                f.write(f"{image_id}\n")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Downloaded : {downloaded}")
    print(f"Already had: {exists}")
    print(f"Failed     : {len(failed)}")

    if failed:
        print(f"Failed IDs saved to: {FAILED_FILE}")

    print("=" * 60)


if __name__ == "__main__":
    main()
