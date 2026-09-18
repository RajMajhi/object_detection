This project uses a custom 15-class subset of the
[COCO 2017 dataset](https://cocodataset.org/).

The original COCO category IDs are preserved in the source
annotations, but are remapped to contiguous training IDs
from 0–14 for model training.

### 4.1 Dataset Selection

The selected classes are:

| Training ID | COCO ID | Class |
|---:|---:|---|
| 0 | 1 | person |
| 1 | 2 | bicycle |
| 2 | 3 | car |
| 3 | 4 | motorcycle |
| 4 | 6 | bus |
| 5 | 8 | truck |
| 6 | 10 | traffic light |
| 7 | 11 | fire hydrant |
| 8 | 13 | stop sign |
| 9 | 15 | bench |
| 10 | 62 | chair |
| 11 | 63 | couch |
| 12 | 64 | potted plant |
| 13 | 85 | clock |
| 14 | 72 | tv |

---
### Dataset Statistics

The original COCO 2017 annotation files contain:

- Train: 118,287 images
- Validation: 5,000 images

After filtering for the selected 15 classes:

- Eligible training images: 82,803
- Selected training images: 20,000
- Validation images containing selected classes: 3,527

The training subset was selected using a deterministic
class-aware selection procedure with random seed `42`.


> 82,803 COCO train2017 images contain at least one of the selected 15 classes. A 20,000-image subset was initially generated for pipeline testing using deterministic class-aware sampling (seed 42). The final experimental dataset will use the full eligible 82,803-image training set.

## Evaluation chart

> Results will be added after training.

Metrics:
- Precision
- Recall
- mAP@50
- mAP@50:95
- Per-class AP
- Confusion matrix
- Parameter count
- GFLOPs
- Model size
- Inference latency
- FPS

### Scripts and their uses
| Script | Purpose |
|---|---|
| `coco15_mapping.py` | Defines COCO-to-training-ID mapping |
| `analyze_coco15.py` | Analyzes selected COCO classes |
| `select_coco15_images.py` | Selects reproducible 20,000-image training subset |
| `download_coco15_images.py` | Downloads selected COCO images |


PyTorch       : 2.14.0+cu130
Torchvision   : 0.29.0+cu130
CPU cores     : 224
Torch threads : 112
CUDA available: False
CUDA version  : 13.0
cuDNN version : 92400
