# Pneumonia Detection from Chest X-Rays

An end-to-end deep learning project that classifies chest X-ray images into three categories — **Covid**, **Normal**, and **Viral Pneumonia** — built entirely from scratch (no pretrained models used in the deployed system), and served through a live web application.

**Live demo:** https://pneumonia-detection-project-kappa.vercel.app/
**Backend API:** https://pneumonia-detector-api-195849682317.asia-south1.run.app

---

## Project Overview

Pneumonia is a serious respiratory infection responsible for a large share of childhood deaths worldwide, and diagnosis typically requires expert review of chest X-rays (CXRs) alongside clinical history. This project explores whether a convolutional neural network can learn to distinguish pneumonia-related lung opacities from normal lungs, using a public chest X-ray image dataset, and packages the result as a usable web tool.

The project covers the full pipeline: data exploration, preprocessing, model building (starting from a basic CNN and comparing against transfer learning), training with systematic hyperparameter experimentation, evaluation, and deployment as an interactive web tool with a live backend API and frontend UI.

## Dataset

- **Source:** Covid19-dataset (public chest X-ray image dataset)
- **Classes:** `Covid`, `Normal`, `Viral Pneumonia`
- **Format:** JPEG images, organized in `train/` and `test/` folders, each containing one subfolder per class
- **Size:** 251 training images (Covid: 111, Normal: 70, Viral Pneumonia: 70), 66 test images (Covid: 26, Normal: 20, Viral Pneumonia: 20)

Note: the original project brief referenced the RSNA Pneumonia Detection dataset (DICOM format, bounding-box detection). The dataset actually used here is a pre-labeled, pre-split JPEG classification dataset, so this project is framed as **3-class image classification** rather than bounding-box detection. Covid was retained as a class (rather than dropped) since COVID-19 pneumonia is radiologically similar to viral pneumonia and is a relevant part of the same underlying detection problem.

## Exploratory Data Analysis

- **Class balance:** moderately imbalanced (Covid has ~1.6x more images than the other two classes)
- **Corrupt/unreadable files:** none found (checked by attempting to open and verify every image file — the image-data equivalent of a missing-values check)
- **Image size consistency:** highly inconsistent, ranging from ~340px to over 4000px on a side, confirming resizing is necessary during preprocessing
- **Visual inspection:** sample images per class reviewed to confirm correct labeling and image quality

## Preprocessing

- All images resized to 224x224
- Normalized using ImageNet mean/std statistics (kept consistent across experiments for a fair comparison)
- Training images augmented with random horizontal flips, rotation, translation/scale jitter, and brightness/contrast jitter to compensate for the small dataset size
- Test images left unaugmented (resize + normalize only) for a clean, honest evaluation

## Modeling

Two architectures were built and compared:

1. **Baseline / deployed model — SimpleCNN (trained fully from scratch):** a convolutional network with 4 conv blocks and a dropout-regularized classifier head, trained with no pretrained weights at any stage. This is the model used in the final deployed application.
2. **Comparison model — ResNet18 (transfer learning):** an ImageNet-pretrained ResNet18 with its final layer replaced and retrained for the 3 target classes, backbone frozen. Built to quantify how much transfer learning helps versus training from scratch on a dataset this small.

## Hyperparameter Experiments

Five training configurations were run and evaluated to satisfy the project's requirement to compare optimizers, epoch counts, and report their effect on final performance:

| Model | Optimizer | Epochs | Accuracy | Covid F1 | Normal F1 | Viral Pneumonia F1 | Notes |
|---|---|---|---|---|---|---|---|
| ResNet18 (transfer learning) | Adam | 15 | 0.89 | 0.94 | 0.88 | 0.85 | Stable convergence, most balanced errors |
| SimpleCNN (from scratch) | Adam | 40 | 0.85 | 0.96 | 0.75 | 0.81 | From-scratch baseline |
| SimpleCNN (from scratch) | SGD | 40 | 0.82 | 0.94 | 0.79 | 0.67 | Slowest convergence, weakest Viral Pneumonia recall (0.55) |
| SimpleCNN (from scratch) | RMSprop | 40 | **0.91** | 0.98 | 0.86 | 0.86 | Highest raw accuracy, but unstable training curve (see observations) |
| SimpleCNN (from scratch) | Adam | 20 | 0.80 | 0.96 | 0.75 | 0.65 | Halving epochs clearly hurt performance |

### Observations on hyperparameter effects

- **Optimizer choice had a larger effect than model architecture.** RMSprop with the from-scratch CNN (91%) actually outperformed the ResNet transfer-learning model (89%) on raw accuracy — a genuinely interesting result given ResNet had the advantage of pretrained features.
- **However, the RMSprop run's training curve was highly unstable** — validation accuracy swung between roughly 36% and 91% across nearby epochs before settling on its best checkpoint. This means the 91% figure reflects genuine peak capability but not necessarily a reliably reproducible result on a re-run; it should be reported alongside this caveat rather than as an unqualified "best model."
- **SGD was consistently the weakest optimizer** on this dataset — slower to converge, and specifically weak at recalling Viral Pneumonia (recall 0.55, missing 8 of 20 true cases in the test set). This is the most clinically concerning single result in the whole comparison, since a missed pneumonia diagnosis is a more serious error than a false alarm.
- **Epoch count has a clear, direct, isolatable effect.** Comparing Adam at 20 vs. 40 epochs (identical in every other respect): accuracy dropped from 85% to 80%, and Viral Pneumonia F1 dropped from 0.81 to 0.65 — confirming the model needs the fuller training run to properly learn the harder class.
- **Across every from-scratch variant, Viral Pneumonia is consistently the hardest class to classify**, and the dominant failure mode is Viral Pneumonia being misclassified as Normal. Since this pattern persists across all optimizers and epoch counts, it appears to be a genuine feature-learning limitation of training from scratch on this small dataset, rather than an artifact of any single hyperparameter choice.
- **Transfer learning (ResNet) remains the most balanced and stable option overall** — slightly lower peak accuracy than the best from-scratch run, but with a smoother training curve and no single class performing dramatically worse than the others.

## Final Model Used for Deployment

**SimpleCNN, trained from scratch, Adam optimizer, 40 epochs, class-weighted loss** was selected as the deployed model, in line with the project's goal of building and deploying a fully from-scratch solution (no pretrained models in the final system).

**Test set performance:**

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Covid            | 1.00      | 0.92   | 0.96     | 26      |
| Normal           | 0.75      | 0.75   | 0.75     | 20      |
| Viral Pneumonia  | 0.77      | 0.85   | 0.81     | 20      |
| **Overall accuracy** |       |        | **0.85** | 66      |

**Confusion matrix:**

|                  | Predicted Covid | Predicted Normal | Predicted Viral Pneumonia |
|------------------|:---:|:---:|:---:|
| **True Covid**            | 24 | 2  | 0  |
| **True Normal**           | 0  | 15 | 5  |
| **True Viral Pneumonia**  | 0  | 3  | 17 |

## Deployment Architecture

- **Backend:** FastAPI app serving the trained model, containerized with Docker and deployed on **Google Cloud Run** (serverless, scales to zero, generous free tier)
- **Frontend:** Next.js + React + TypeScript + Tailwind CSS, deployed on **Vercel**
- The frontend sends uploaded images to the backend's `/predict` endpoint and displays the returned classification, confidence score, and per-class probability breakdown
- A **60% confidence threshold** guardrail returns an "Uncertain" result instead of forcing a classification when the model isn't confident — a safeguard against confidently-wrong predictions on unclear or non-X-ray images

## Project Structure

```
pneumonia-detection-project/
├── backend/
│   ├── app/
│   │   └── main.py                    # FastAPI app serving predictions
│   ├── src/
│   │   ├── dataset.py                 # Data loading + preprocessing
│   │   ├── model.py                   # SimpleCNN + ResNet18 model definitions
│   │   ├── train.py                   # Main training script (SimpleCNN, Adam, 40 epochs)
│   │   ├── train_sgd.py               # Hyperparameter experiment: SGD optimizer
│   │   ├── train_rmsprop.py           # Hyperparameter experiment: RMSprop optimizer
│   │   ├── train_adam_20ep.py         # Hyperparameter experiment: Adam, 20 epochs
│   │   └── evaluate.py                # Evaluation metrics + confusion matrix
│   ├── notebooks/
│   │   └── 01_eda.ipynb               # Exploratory data analysis
│   ├── models/                        # Saved trained model weights (all experiments)
│   ├── Dockerfile
│   └── requirements-deploy.txt
├── frontend/
│   ├── app/
│   │   └── page.tsx                   # Upload UI + result display
│   └── package.json
└── README.md
```

## Tech Stack

- **Modeling:** Python, PyTorch, torchvision
- **Backend API:** FastAPI, Docker, Google Cloud Run
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS, Vercel
- **Data handling / evaluation:** Pandas, scikit-learn, Matplotlib

## Running Locally

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Visit `http://localhost:3000`

## Limitations & Disclaimer

This project is an educational demonstration, not a medical diagnostic tool. The model was trained on a small dataset (317 images total) from a single source, has not been clinically validated, and should not be used to inform real medical decisions. It is not designed to recognize non-X-ray images; the confidence-threshold guardrail reduces but does not eliminate the risk of a confidently-wrong prediction on unrelated image types.

## Future Improvements

- Expand the dataset for better generalization and more reliable minority-class recall
- Investigate learning-rate scheduling to stabilize the RMSprop training curve and potentially reproduce its peak accuracy more reliably
- Fine-tune more layers of the ResNet backbone rather than only the final layer
- Add Grad-CAM visualization to show which regions of the X-ray influenced each prediction
- Run formal k-fold cross-validation given the small test set size