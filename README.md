# LOSAT Based Violence Detector

AI-based violence detection project using a pretrained 3D CNN (`R3D-18`) and Adaptive LOSAT thresholding. The project supports uploaded video prediction through a Streamlit dashboard and includes Colab-ready training and evaluation scripts for the `RWF-2000` dataset.

## Features

- Violence / Non-Violence classification using `R3D-18`
- 16-frame clip-based video analysis
- Adaptive LOSAT thresholding
- Uploaded video prediction dashboard in Streamlit
- CSV export for per-clip uploaded-video predictions
- Google Colab training script
- Google Colab evaluation script with metrics

## Project Structure

```text
violence_detector/
├── app.py
├── model.py
├── losat.py
├── dataset.py
├── train_colab.py
├── evaluate_colab.py
├── utils.py
├── requirements.txt
└── best_model.pth
```

## Tech Stack

- `torch` - Core deep learning framework used to load and run the violence detection model.
- `torchvision` - Provides the pretrained `R3D-18` video model and related vision utilities.
- `opencv-python` - Handles video reading, frame extraction, resizing, and preprocessing.
- `numpy` - Supports numerical operations and frame array manipulation.
- `pandas` - Used for event logging, tabular results, and CSV file handling.
- `streamlit` - Builds the web-based dashboard for the project interface.
- `streamlit-webrtc` - Enables live webcam video streaming inside the Streamlit app.

## Model Details

- Backbone: `torchvision.models.video.r3d_18`
- Classes: `Violence`, `Non-Violence`
- Clip length: `16` frames
- Input size: `112 x 112`
- Inference: CPU
- Training: GPU recommended in Google Colab

## LOSAT Logic

The project uses Adaptive LOSAT thresholding:

- Motion metric:
  `Mt = average L1 difference between consecutive frames`
- Threshold update:
  `Tt = alpha * T_prev + (1 - alpha) * score + beta * Mt`
- Parameters:
  `alpha = 0.8`
  `beta = 0.2`

Uploaded-video mode uses majority-based final decision logic over analyzed clips.

## Installation

Create and activate a Python environment if needed, then install dependencies:

```bash
cd c:\Users\Balaji\ViolenceDetector\violence_detector
pip install -r requirements.txt
```

## Run the Streamlit App

```bash
cd c:\Users\Balaji\ViolenceDetector\violence_detector
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal.

## Uploaded Video Demo

1. Open the dashboard.
2. Select `Upload Video`.
3. Upload a video file.
4. Click `Predict Uploaded Video`.
5. View:
   - Score
   - Threshold
   - Motion
   - Final Alert
   - Per-Clip Predictions

## Training in Google Colab

Use the `RWF-2000` dataset with this structure:

```text
/content/RWF-2000/
├── train/
│   ├── Fight/
│   └── NonFight/
└── val/
    ├── Fight/
    └── NonFight/
```

Run training:

```python
!pip install torch torchvision opencv-python numpy pandas
!python train_colab.py --data_root /content/RWF-2000 --output_dir /content --epochs 10 --batch_size 4 --lr 1e-4
```

This saves:

- `/content/best_model.pth`
- `/content/final_model.pth`
- `/content/validation_metrics.csv`

Place `best_model.pth` inside the project folder for local inference:

```text
violence_detector/best_model.pth
```

## Evaluation in Google Colab

To evaluate an already trained model without retraining:

```python
!pip install torch torchvision opencv-python numpy pandas
!python evaluate_colab.py --data_root /content/RWF-2000 --model_path /content/best_model.pth --split val --batch_size 4 --output_csv /content/evaluation_predictions.csv
```

This prints:

- Accuracy
- Precision
- Recall
- Specificity
- False Alarm Rate
- Miss Rate
- F1 Score
- TP, TN, FP, FN

It also saves:

- `/content/evaluation_predictions.csv`

## Notes

- Uploaded-video prediction is more stable than live mode for presentation/demo use.
- High-motion non-violent actions may still create false positives.
- Performance depends on the trained `best_model.pth`.
- For best presentation output, use uploaded-video mode.

## Author

Major Project: `AI-Based Video Analytics for Violence Detection`
By: Deevi Balaji

