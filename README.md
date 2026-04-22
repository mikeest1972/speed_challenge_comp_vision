
# Speed Prediction from Dashcam Video

Comma.ai speed challenge: predict vehicle speed (mph) from dashcam video, evaluated on MSE.

- `data/train.mp4` — 20,400 frames @ 20fps with ground truth speeds in `data/train.txt`
- `data/test.mp4` — 10,798 frames @ 20fps, no labels (submit `test.txt`)

**Target:** MSE < 10 (good), < 5 (better), < 3 (excellent)

---

## Approach: Optical Flow → CNN + LSTM

### Why optical flow?

Raw RGB frames contain too much static information (road texture, sky, buildings). Optical flow isolates what we actually care about: **pixel motion between frames**, which directly encodes apparent speed.

We use **Farneback dense optical flow** (`cv2.calcOpticalFlowFarneback`) to produce a 2-channel (dx, dy) flow field between each pair of consecutive frames.
[ref](https://en.wikipedia.org/wiki/Optical_flow)

### Architecture

```
Frame t, Frame t+1
        │
        ▼
  Optical Flow (Farneback)
  Output: (H, W, 2) — dx, dy channels
        │
        ▼ (for each frame in window)
  CNN Backbone
  - Conv2D + BatchNorm + ReLU blocks
  - GlobalAveragePooling
  Output: feature vector per timestep
        │
        ▼ (sequence of feature vectors)
  LSTM
  - Captures temporal continuity of speed
  Output: hidden state per timestep
        │
        ▼
  Dense → scalar speed prediction
```

**Key design decisions:**
- CNN encodes spatial motion features from each flow frame independently
- LSTM models temporal dynamics — speed is smooth and doesn't change instantaneously
- Window of N consecutive flow frames fed as a sequence to the LSTM
- Loss: MSE, with optional L1 smoothness penalty on consecutive predictions

### Training

- Input: sliding window of N consecutive optical flow frames
- Labels: corresponding speeds from `train.txt`
- Train/val split on `train.mp4`
- Optimizer: Adam
- Augmentation: horizontal flip (with negated dx channel), brightness jitter on source frames

---

## Results

| Experiment | MSE (val) | Notes |
|---|---|---|
| baseline | — | — |

---

## Setup

```bash
pip install -r requirements.txt
python train.py
python predict.py  # outputs test.txt
```

