# Drowsiness & Yawn Detection

**Project**: Detects eye state and yawn state from images using an EfficientNetB0-based classifier. Includes training code (Colab-ready) and an inference script (`inference.py`) that crops the best eye region (`eye_crop_best.detect_best_eye`) and runs predictions with a saved model (`best_model.h5`).

---

## Contents

```
README.md                        # this file
train.ipynb                       # Colab training notebook (example)
inference.py                      # CLI inference script
best_model.h5                     # trained model (use Git LFS or external storage)
eye_crop_best.py                  # helper module: detect_best_eye(image_path) -> cropped eye image
dataset_new/                      # expected dataset structure (train/test folders)
confusion_matrix.png              # saved confusion matrix plot
accuracy_loss.png                 # saved accuracy/loss curves
requirements.txt                  # dependencies
.gitignore                        # ignore large/binary/cache files
```

---

## Quick description

The model classifies into four categories:

* `Closed` (eye closed)
* `Open` (eye open)
* `no_yawn`
* `yawn`

At inference, predictions are made on both the full-face image and the best-eye crop. Their outputs are combined via rule-based logic to decide the final state:

* **Drowsy**
* **Non-Drowsy**

---

## Installation

1. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate    # windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training (Colab)

* Open `train.ipynb` in Google Colab.
* It expects a zipped dataset at `drive/MyDrive/dataset_new.zip`.
* The notebook extracts, prepares data generators, builds EfficientNetB0, trains with callbacks, and saves `best_model.h5` plus plots.

Dataset structure:

```
dataset_new/
  train/
    Closed/
    Open/
    no_yawn/
    yawn/
  test/
    Closed/
    Open/
    no_yawn/
    yawn/
```

---

## Inference (CLI)

Run on a single image:

```bash
python inference.py --image "testing images/sample.jpg" --model best_model.h5 --save output.jpg --show
```

Options:

* `--image`: path to input image
* `--model`: path to saved Keras model
* `--save`: optional path to save annotated output
* `--show`: display result with matplotlib
* `--classes`: optional override of class names (default: Closed Open no\_yawn yawn)

Example output:

```
Face prediction: yawn (0.91)
Eye prediction:  Closed (0.87)
Combined state: Drowsy (avg_conf=0.89)
```

---

## Requirements

Dependencies are in `requirements.txt`. Main ones:

* numpy
* opencv-python
* matplotlib
* tensorflow (includes keras)
* scikit-learn
* seaborn
* google-colab (optional, for Colab use)

---

## Suggested .gitignore

```
__pycache__/
*.pyc
*.h5
*.png
*.jpg
.DS_Store
venv/
.env
*.zip
.ipynb_checkpoints/
```

---

## Notes & Improvements

* Use **Git LFS** for `best_model.h5`.
* Consider training **separate eye and mouth classifiers** for better results.
* Add real-time webcam support (`cv2.VideoCapture`) for deployment.
* Convert to **TensorFlow Lite** or **ONNX** for edge devices.

---

## License

Choose and add a license (e.g., MIT) for reuse terms.
