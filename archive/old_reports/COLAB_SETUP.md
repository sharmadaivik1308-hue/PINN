# Google Colab Setup Guide
## Earthquake PINN — Physics-Informed Neural Network

---

## Quick Start

1. Upload `Earthquake_PINN.ipynb` to [Google Colab](https://colab.research.google.com)
2. Upload `earthquake_29_features.csv` to Google Drive under `MyDrive/Earthquake_PINN_Project/data/`
3. Click **Runtime → Change runtime type → T4 GPU** (recommended)
4. Click **Runtime → Run all**
5. Everything runs automatically — all outputs saved to Google Drive

---

## Required Files

| File | Description | Required |
|---|---|---|
| `Earthquake_PINN.ipynb` | Main Colab notebook | ✅ Yes |
| `earthquake_29_features.csv` | Dataset (2,497 samples × 31 columns) | ✅ Yes |

All Python source files are **embedded inside the notebook** and automatically written to Google Drive on first run. You do NOT need to upload any `.py` files.

---

## Folder Structure (Created Automatically)

```
MyDrive/
└── Earthquake_PINN_Project/
    ├── config/
    │   └── config.yaml
    ├── data/
    │   └── earthquake_29_features.csv   ← Place dataset here
    ├── physics/
    │   ├── __init__.py
    │   ├── energy_constraint.py
    │   ├── frequency_constraint.py
    │   └── physics_loss.py
    ├── src/
    │   ├── __init__.py
    │   ├── data_loader.py
    │   ├── preprocess.py
    │   ├── model.py
    │   ├── pinn_model.py
    │   ├── train.py
    │   ├── pinn_train.py
    │   ├── evaluate.py
    │   └── utils.py
    ├── saved_models/
    │   ├── baseline_mlp.pth
    │   ├── basic_pinn.pth
    │   ├── attention_pinn.pth
    │   ├── full_pinn.pth
    │   └── scaler.pkl
    ├── outputs/
    │   ├── plots/           ← All visualizations (27+ PNG files)
    │   ├── predictions/     ← Prediction CSV files
    │   └── metrics/         ← JSON metrics and loss histories
    ├── reports/
    │   ├── PROJECT_ANALYSIS_REPORT.md
    │   ├── BASELINE_MODEL_REPORT.md
    │   ├── PINN_IMPLEMENTATION_REPORT.md
    │   └── FINAL_MODEL_REPORT.md
    └── compare.py
```

---

## Dataset Location

### Method 1: Google Drive (Preferred)

1. Open [Google Drive](https://drive.google.com)
2. Create a folder: `Earthquake_PINN_Project/data/`
3. Upload `earthquake_29_features.csv` into that folder
4. The notebook will auto-detect it on mount

The notebook also searches these fallback locations:
- `MyDrive/Earthquake_PINN_Project/earthquake_29_features.csv`
- `MyDrive/earthquake_29_features.csv`
- `/content/earthquake_29_features.csv`

### Method 2: Manual Upload

If the dataset is not found in Google Drive, the notebook will show an upload dialog. Simply drag and drop the CSV file.

---

## How to Run

### Step 1: Open in Colab
- Go to [https://colab.research.google.com](https://colab.research.google.com)
- Click **File → Upload notebook**
- Select `Earthquake_PINN.ipynb`

### Step 2: Enable GPU (Recommended)
- Click **Runtime → Change runtime type**
- Select **T4 GPU** (free tier) or **A100** (Colab Pro)
- Click **Save**

### Step 3: Run Everything
- Click **Runtime → Run all**
- The notebook will:
  1. Install missing dependencies
  2. Mount Google Drive
  3. Create project structure and `os.chdir()` into it
  4. Write all source files to Drive
  5. Load dataset (from Drive or via upload)
  6. Configure output paths via `configure_paths()`
  7. Train all 4 models (with GPU if available)
  8. Generate 30+ comparison plots
  9. Save everything to Drive
  10. Create a downloadable ZIP

### Total Runtime
- **CPU:** ~2–5 minutes
- **T4 GPU:** ~1–2 minutes

---

## Expected Outputs

### Models Trained
| Model | Architecture | Description |
|---|---|---|
| Baseline MLP | 29→128→64→32→1 | Pure data-driven (control) |
| Basic PINN | Same + physics loss | Energy + frequency constraints |
| Attention PINN | + attention gate | Learned feature importance |
| Full PINN | + uncertainty head | Aleatoric uncertainty (NLL) |

### Plots Generated (27+)
- Loss curves (per model + comparison)
- Predicted vs Actual scatter plots
- Error distribution histograms
- Attention weight visualizations
- Uncertainty plots (Full PINN)
- Physics loss component plots
- Training cost comparison

### Files Saved
- 4 trained model files (`.pth`)
- 4 prediction CSV files
- 10 metrics JSON files
- 27+ PNG plots
- 4 research reports (`.md`)
- 1 ZIP export

---

## Troubleshooting

### "Dataset not found"
- Ensure `earthquake_29_features.csv` is in `MyDrive/Earthquake_PINN_Project/data/`
- Or use the manual upload option when prompted

### "Module not found" error
- Make sure you run cells in order (Run all)
- The notebook calls `os.chdir(PROJECT_ROOT)` and adds it to `sys.path`
- The project files cell must execute before the model training cells

### Slow training
- Check Runtime → Change runtime type → GPU is selected
- Free Colab uses T4 GPU, which is sufficient for this project

### "Drive mount failed"
- Allow the Google Drive permissions popup
- If persistent, try: Runtime → Restart runtime, then Run all again

### Plots not displaying
- The notebook uses `%matplotlib inline` for Colab display
- Plots are both saved to Google Drive AND displayed inline via `plt.show()`

### Results differ from local runs
- Random seeds are fixed (`seed=42`) for reproducibility
- Minor floating-point differences may occur between CPU/GPU

### Path-related errors
- The notebook calls `os.chdir(PROJECT_ROOT)` after creating directories
- It also calls `configure_paths(PROJECT_ROOT)` to set all output directories
- Both ensure that relative paths in the source code resolve correctly

---

## GPU Recommendations

| GPU | Speed | Availability |
|---|---|---|
| **T4** (Free tier) | Good | Always available |
| **A100** (Pro) | Fastest | Colab Pro/Pro+ |
| **L4** (Pro) | Very Good | Colab Pro/Pro+ |
| **CPU** | Slowest | Always available |

This project is lightweight (~15K parameters) and runs fine on CPU. GPU mainly accelerates batch operations.

---

## Sharing with Collaborators

1. Share the `Earthquake_PINN.ipynb` notebook file
2. Share the `earthquake_29_features.csv` dataset
3. The collaborator opens the notebook in Colab and clicks "Run all"
4. Everything else is automatic

No local setup, no terminal commands, no environment configuration needed.

---

## Reproducing the Notebook Locally

If you modify the Python source files and want to regenerate the notebook:

```bash
python generate_notebook.py
```

This reads all `.py` files and embeds them into `Earthquake_PINN.ipynb`.
