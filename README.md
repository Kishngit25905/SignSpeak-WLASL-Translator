# SignSpeak 🤟🔊

**Real-Time Sign Language to Voice Translation Using Machine Learning and Computer Vision**

SignSpeak is a system that recognizes American Sign Language (ASL) word-level signs from video and converts them into spoken audio, aiming to bridge communication between sign language users and non-signers.

---

## 📌 Overview

SignSpeak processes sign language video clips, detects hand landmarks using **MediaPipe**, classifies the signed word using a trained machine learning model, and speaks the recognized word aloud using **Text-to-Speech (TTS)**.

```
Video Input → Hand Landmark Detection (MediaPipe) → Gesture/Word Classification (ML Model) → Text Builder → Text-to-Speech Output
```

---

## ✨ Features

- Word-level ASL sign recognition from video clips
- Hand landmark extraction using MediaPipe (21 keypoints per hand, per frame)
- ML classifier trained on landmark sequences extracted from WLASL video data
- Confidence-based prediction filtering to reduce flicker
- Word-building logic
- Offline text-to-speech voice output

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Model Training | TensorFlow / Scikit-learn |
| Text-to-Speech | pyttsx3 |
| Dataset | [WLASL — Word-Level American Sign Language](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) |

---

## 📂 Dataset

This project uses **WLASL (Word-Level American Sign Language)**, a large-scale video dataset for dynamic, word-level sign recognition — as opposed to static fingerspelling/alphabet datasets.

Each entry in the dataset's metadata JSON identifies a video clip, its train/val/test split, and the action label with start/end frame range, e.g.:

```json
"05237": {"subset": "train", "action": [77, 1, 55]}
```

- `"subset"` — which split the clip belongs to (train / val / test)
- `"action"` — `[gloss/class ID, start frame, end frame]`, identifying which word is signed and where the sign occurs within the clip

Because WLASL is video-based and word-level (rather than static images), it requires extracting frames from each clip and building landmark **sequences** per video, rather than a single feature vector per image — a meaningfully more complex pipeline than static alphabet classification.

> The dataset is **not included** in this repository due to size. Download it from the Kaggle link above.

---

## 📂 Project Structure

```
SignSpeak/
│
├── dataset_preprocessing/     # Extracts frames + hand landmark sequences from WLASL video clips
├── model_training/            # Training scripts and saved model
├── real_time_detection/       # Live/video-based prediction pipeline
├── tts_output/                # Text-to-speech integration
├── models/                    # Trained model files (.h5 / .pkl)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
pip install opencv-python mediapipe tensorflow numpy matplotlib scikit-learn pyttsx3
```

---

## 🚀 How It Works

1. **Dataset Preprocessing** — Video clips from WLASL are read using the metadata JSON (video ID, split, action/gloss label, frame range). Frames within the labeled action range are extracted, and MediaPipe is run on each frame to get a sequence of hand landmark vectors per clip.
2. **Model Training** — A classifier is trained on these landmark sequences to predict the signed word (gloss class).
3. **Detection Pipeline** — New video/frame sequences are processed the same way and passed to the trained model to predict the signed word.
4. **Word Building** — Recognized words are accumulated into output text.
5. **Voice Output** — The resulting text is passed to a TTS engine and spoken aloud.

---

## 📊 Results & Status

This project was built as a functional prototype for a term paper. Every stage of the pipeline was implemented and worked to some extent:

- ✅ Dataset preprocessing and hand landmark sequence extraction — working
- ✅ Model training — model trained and produced predictions
- ✅ Sign/word detection pipeline — working, though not fully reliable
- ✅ Text-to-speech voice output — working

**Detection accuracy was inconsistent across signs** — some words were recognized reliably, while others were frequently confused, most likely due to visually/motion-similar signs, limited video samples for some classes in the WLASL dataset, and the inherent difficulty of word-level (motion-based) recognition compared to static gestures.

*(Add your actual accuracy % and a confusion matrix screenshot here once you have them handy)*

### ⚠️ Limitations

- Not all planned features were completed within the project timeline.
- Detection accuracy is inconsistent — the model performs well on some signed words and poorly on others.
- WLASL is a challenging dataset: many classes have limited video samples, and motion-based signs are harder to classify than static poses.
- Real-world performance can vary with lighting, camera angle, and signer variation.

This repository represents a working end-to-end pipeline demonstrated to a lecturer as a proof of concept, rather than a fully polished, production-ready system — a natural stopping point for further development.

---

## 🔮 Future Work

- Improve accuracy using sequence models better suited to motion (LSTM/GRU/Transformer) if not already fully utilized
- Expand the number of word classes covered
- Support full sentence-level translation with grammar structuring
- Improve robustness to lighting/background/signer variation
- Deploy as a mobile application
- Extend to Indian Sign Language (ISL) and other sign languages

---

## 📄 License

This project was developed for academic purposes as part of a term paper submission.
