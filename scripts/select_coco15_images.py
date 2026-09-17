import json
import random
from collections import defaultdict

TRAIN_JSON = "datasets/coco15/instances_train2017.json"
OUTPUT_FILE = "datasets/coco15/train_image_ids.txt"

TARGET_IMAGES = 20000
SEED = 42

SELECTED_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    15: "bench",
    62: "chair",
    63: "couch",
    64: "potted plant",
    85: "clock",
    72: "tv",
}

random.seed(SEED)

print(f"Loading: {TRAIN_JSON}")

with open(TRAIN_JSON, "r") as f:
    data = json.load(f)

# Image ID -> set of selected COCO category IDs
image_classes = defaultdict(set)

for ann in data["annotations"]:
    category_id = ann["category_id"]

    if category_id in SELECTED_CLASSES:
        image_classes[ann["image_id"]].add(category_id)

eligible_images = list(image_classes.keys())

print(f"Eligible images: {len(eligible_images)}")

# Shuffle deterministically
random.shuffle(eligible_images)

# Start with one pass of random selection
selected = eligible_images[:TARGET_IMAGES]

# Make sure every selected class is represented
for category_id in SELECTED_CLASSES:
    class_images = [
        image_id
        for image_id in eligible_images
        if category_id in image_classes[image_id]
    ]

    if not any(
        category_id in image_classes[image_id]
        for image_id in selected
    ):
        replacement = random.choice(class_images)

        # Replace a random selected image
        index = random.randrange(len(selected))
        selected[index] = replacement

# Remove duplicates while preserving order
selected = list(dict.fromkeys(selected))

# If replacements caused the count to drop, fill again
remaining = [
    image_id for image_id in eligible_images
    if image_id not in selected
]

random.shuffle(remaining)

while len(selected) < TARGET_IMAGES and remaining:
    selected.append(remaining.pop())

# Save IDs
with open(OUTPUT_FILE, "w") as f:
    for image_id in sorted(selected):
        f.write(f"{image_id}\n")

print("=" * 50)
print(f"Selected images: {len(selected)}")
print(f"Output: {OUTPUT_FILE}")
print("=" * 50)

# Report class coverage
class_counts = defaultdict(int)

for image_id in selected:
    for category_id in image_classes[image_id]:
        class_counts[category_id] += 1

print("\nImage coverage by class:")

for category_id, name in SELECTED_CLASSES.items():
    print(
        f"{category_id:2d}  {name:15s} "
        f"{class_counts[category_id]:6d} images"
    )
