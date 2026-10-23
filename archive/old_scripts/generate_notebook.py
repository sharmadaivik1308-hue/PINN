"""
generate_notebook.py
--------------------
Reads all project source files and generates a fully self-contained
Google Colab notebook (Earthquake_PINN.ipynb).

Usage:
    python generate_notebook.py

The generated notebook will:
    1. Install all dependencies
    2. Mount Google Drive
    3. Write the complete project structure to Google Drive
    4. Run all 4 experiments (Baseline, Basic PINN, Attention PINN, Full PINN)
    5. Generate comparison visualisations
    6. Save everything to Google Drive
    7. Create a downloadable ZIP file
"""

import json
import os
import sys


def read_file(path):
    """Read a file's content."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_md_cell(source_lines):
    """Create a Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


def make_code_cell(source_lines):
    """Create a Code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def build_notebook():
    """Build the complete notebook structure."""
    cells = []

    project_dir = os.path.dirname(os.path.abspath(__file__))

    # ================================================================
    # CELL 1: Title & Overview
    # ================================================================
    cells.append(make_md_cell([
        "# \U0001f30d Physics-Informed Neural Network for Earthquake Magnitude Prediction\n",
        "\n",
        "**Research Project \u2014 Complete Google Colab Implementation**\n",
        "\n",
        "This notebook implements a progressive series of neural network models for\n",
        "earthquake magnitude prediction from 29 P-wave features:\n",
        "\n",
        "| # | Model | Description |\n",
        "|---|---|---|\n",
        "| 1 | **Baseline MLP** | Standard feed-forward network (control) |\n",
        "| 2 | **Basic PINN** | + Physics constraints (energy + frequency) |\n",
        "| 3 | **Attention PINN** | + Learned feature importance via sigmoid gating |\n",
        "| 4 | **Full PINN** | + Uncertainty estimation (aleatoric, NLL loss) |\n",
        "\n",
        "### Physics Constraints\n",
        "- **Energy\u2013Magnitude**: Gutenberg\u2013Richter law \u2014 `log\u2081\u2080(E) \u221d 1.5 \u00d7 M`\n",
        "- **Frequency\u2013Magnitude**: Inverse scaling \u2014 `M \u221d \u2212log(f)`\n",
        "\n",
        "---\n",
        "**Instructions:** Click **Runtime \u2192 Run all** to execute the entire pipeline.\n",
        "All outputs are automatically saved to Google Drive.\n",
    ]))

    # ================================================================
    # CELL 2: Install Dependencies
    # ================================================================
    cells.append(make_md_cell([
        "## 1\ufe0f\u20e3 Install Dependencies\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Install all required packages (only installs if missing)\n",
        "# ============================================================\n",
        "import subprocess, sys, importlib\n",
        "\n",
        "REQUIRED = [\n",
        "    ('torch',        'torch'),\n",
        "    ('torchvision',  'torchvision'),\n",
        "    ('numpy',        'numpy'),\n",
        "    ('pandas',       'pandas'),\n",
        "    ('matplotlib',   'matplotlib'),\n",
        "    ('seaborn',      'seaborn'),\n",
        "    ('scikit-learn', 'sklearn'),\n",
        "    ('scipy',        'scipy'),\n",
        "    ('pyyaml',       'yaml'),\n",
        "]\n",
        "\n",
        "missing = []\n",
        "for pip_name, import_name in REQUIRED:\n",
        "    try:\n",
        "        importlib.import_module(import_name)\n",
        "    except ImportError:\n",
        "        missing.append(pip_name)\n",
        "\n",
        "if missing:\n",
        "    print(f'\U0001f4e6 Installing missing packages: {missing}')\n",
        "    subprocess.check_call(\n",
        "        [sys.executable, '-m', 'pip', 'install', '-q'] + missing\n",
        "    )\n",
        "    print('\u2705 Installation complete.')\n",
        "else:\n",
        "    print('\u2705 All dependencies already installed.')\n",
    ]))

    # ================================================================
    # CELL 3: Imports & GPU Detection
    # ================================================================
    cells.append(make_md_cell([
        "## 2\ufe0f\u20e3 Imports & GPU Detection\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Core imports and automatic device selection\n",
        "# ============================================================\n",
        "%matplotlib inline\n",
        "\n",
        "import os, sys, json, time, pickle, random, shutil, warnings\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "import yaml\n",
        "\n",
        "warnings.filterwarnings('ignore')\n",
        "\n",
        "# --- Automatic Device Selection ---\n",
        "print('=' * 55)\n",
        "print('  DEVICE DETECTION')\n",
        "print('=' * 55)\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    DEVICE = torch.device('cuda')\n",
        "    gpu_name = torch.cuda.get_device_name(0)\n",
        "    cuda_ver = torch.version.cuda\n",
        "    mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)\n",
        "    print(f'  \U0001f7e2 GPU Detected: {gpu_name}')\n",
        "    print(f'     CUDA Version: {cuda_ver}')\n",
        "    print(f'     GPU Memory:   {mem_gb:.1f} GB')\n",
        "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n",
        "    DEVICE = torch.device('mps')\n",
        "    print('  \U0001f7e2 Apple MPS Detected')\n",
        "else:\n",
        "    DEVICE = torch.device('cpu')\n",
        "    print('  \U0001f7e1 Using CPU (no GPU detected)')\n",
        "\n",
        "print(f'\\n  PyTorch Version: {torch.__version__}')\n",
        "print(f'  Device:          {DEVICE}')\n",
        "print('=' * 55)\n",
    ]))

    # ================================================================
    # CELL 4: Reproducibility
    # ================================================================
    cells.append(make_md_cell([
        "## 3\ufe0f\u20e3 Set Random Seeds for Reproducibility\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Deterministic seeds \u2014 ensures identical results across runs\n",
        "# ============================================================\n",
        "SEED = 42\n",
        "\n",
        "random.seed(SEED)\n",
        "np.random.seed(SEED)\n",
        "torch.manual_seed(SEED)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(SEED)\n",
        "    torch.backends.cudnn.deterministic = True\n",
        "    torch.backends.cudnn.benchmark = False\n",
        "\n",
        "os.environ['PYTHONHASHSEED'] = str(SEED)\n",
        "\n",
        "print(f'\u2705 Random seed set to {SEED} for Python, NumPy, and PyTorch.')\n",
        "print(f'   CUDNN deterministic: {torch.backends.cudnn.deterministic if torch.cuda.is_available() else \"N/A (CPU)\"}')\n",
    ]))

    # ================================================================
    # CELL 5: Mount Google Drive & Setup Project
    # ================================================================
    cells.append(make_md_cell([
        "## 4\ufe0f\u20e3 Mount Google Drive & Setup Project\n",
        "\n",
        "Creates the full project directory structure in Google Drive.\n",
        "If not running in Colab, uses the current working directory.\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Mount Google Drive and create project directory structure\n",
        "# ============================================================\n",
        "try:\n",
        "    from google.colab import drive\n",
        "    drive.mount('/content/drive')\n",
        "    IN_COLAB = True\n",
        "    DRIVE_ROOT = '/content/drive/MyDrive'\n",
        "    print('\u2705 Google Drive mounted successfully.')\n",
        "except ImportError:\n",
        "    IN_COLAB = False\n",
        "    DRIVE_ROOT = os.getcwd()\n",
        "    print('\u2139\ufe0f  Not in Colab. Using current directory as project root.')\n",
        "\n",
        "# --- Project paths ---\n",
        "PROJECT_ROOT = os.path.join(DRIVE_ROOT, 'Earthquake_PINN_Project')\n",
        "\n",
        "DIRS = [\n",
        "    os.path.join(PROJECT_ROOT, 'config'),\n",
        "    os.path.join(PROJECT_ROOT, 'data'),\n",
        "    os.path.join(PROJECT_ROOT, 'physics'),\n",
        "    os.path.join(PROJECT_ROOT, 'src'),\n",
        "    os.path.join(PROJECT_ROOT, 'saved_models'),\n",
        "    os.path.join(PROJECT_ROOT, 'outputs', 'plots'),\n",
        "    os.path.join(PROJECT_ROOT, 'outputs', 'predictions'),\n",
        "    os.path.join(PROJECT_ROOT, 'outputs', 'metrics'),\n",
        "    os.path.join(PROJECT_ROOT, 'reports'),\n",
        "]\n",
        "\n",
        "for d in DIRS:\n",
        "    os.makedirs(d, exist_ok=True)\n",
        "\n",
        "# Change working directory to project root so all relative\n",
        "# paths in config.yaml and module imports resolve correctly\n",
        "os.chdir(PROJECT_ROOT)\n",
        "\n",
        "# Add project root to sys.path for module imports\n",
        "if PROJECT_ROOT not in sys.path:\n",
        "    sys.path.insert(0, PROJECT_ROOT)\n",
        "\n",
        "print(f'\\n\U0001f4c2 Project root: {PROJECT_ROOT}')\n",
        "print(f'\U0001f4c2 Working dir:  {os.getcwd()}')\n",
        "print('\u2705 All directories created:')\n",
        "for d in DIRS:\n",
        "    print(f'   {os.path.relpath(d, DRIVE_ROOT)}')\n",
    ]))

    # ================================================================
    # CELL 6: Write Project Source Files
    # ================================================================
    cells.append(make_md_cell([
        "## 5\ufe0f\u20e3 Write Project Source Files to Google Drive\n",
        "\n",
        "This cell writes all Python source files, config, and reports to Google Drive,\n",
        "preserving the exact project structure. The source code is **embedded** in this\n",
        "notebook so collaborators only need this single `.ipynb` file + the dataset.\n",
    ]))

    # Read all source files
    file_map = {}

    # Config
    file_map["config/config.yaml"] = read_file(
        os.path.join(project_dir, "config", "config.yaml")
    )

    # Physics package
    for fname in ["__init__.py", "energy_constraint.py",
                  "frequency_constraint.py", "physics_loss.py"]:
        file_map[f"physics/{fname}"] = read_file(
            os.path.join(project_dir, "physics", fname)
        )

    # Src package
    for fname in ["__init__.py", "data_loader.py", "preprocess.py",
                  "model.py", "pinn_model.py", "train.py",
                  "pinn_train.py", "evaluate.py", "utils.py"]:
        file_map[f"src/{fname}"] = read_file(
            os.path.join(project_dir, "src", fname)
        )

    # Compare script
    file_map["compare.py"] = read_file(
        os.path.join(project_dir, "compare.py")
    )

    # Reports
    for report in ["BASELINE_MODEL_REPORT.md", "PINN_IMPLEMENTATION_REPORT.md",
                    "FINAL_MODEL_REPORT.md", "PROJECT_ANALYSIS_REPORT.md"]:
        report_path = os.path.join(project_dir, report)
        if os.path.exists(report_path):
            file_map[f"reports/{report}"] = read_file(report_path)

    # Build the code cell
    write_lines = [
        "# ============================================================\n",
        "# Write all project source files to Google Drive\n",
        "# ============================================================\n",
        "\n",
        "PROJECT_FILES = " + json.dumps(file_map, indent=2, ensure_ascii=False) + "\n",
        "\n",
        "files_written = 0\n",
        "for rel_path, content in PROJECT_FILES.items():\n",
        "    full_path = os.path.join(PROJECT_ROOT, rel_path)\n",
        "    os.makedirs(os.path.dirname(full_path), exist_ok=True)\n",
        "    with open(full_path, 'w', encoding='utf-8') as f:\n",
        "        f.write(content)\n",
        "    files_written += 1\n",
        "\n",
        "print(f'\u2705 {files_written} project files written to Google Drive.')\n",
        "print(f'   Location: {PROJECT_ROOT}')\n",
        "print()\n",
        "for rel_path in sorted(PROJECT_FILES.keys()):\n",
        "    print(f'   \U0001f4c4 {rel_path}')\n",
    ]

    cells.append(make_code_cell(write_lines))

    # ================================================================
    # CELL 7: Load Dataset
    # ================================================================
    cells.append(make_md_cell([
        "## 6\ufe0f\u20e3 Load Dataset\n",
        "\n",
        "The dataset is loaded automatically using one of two methods:\n",
        "\n",
        "**Method 1 (Preferred):** Place `earthquake_29_features.csv` in your Google Drive:\n",
        "```\n",
        "MyDrive/Earthquake_PINN_Project/data/earthquake_29_features.csv\n",
        "```\n",
        "\n",
        "**Method 2 (Fallback):** If the file is not found, you'll be prompted to upload it.\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Load dataset \u2014 auto-detect from Drive or prompt upload\n",
        "# ============================================================\n",
        "DATASET_NAME = 'earthquake_29_features.csv'\n",
        "\n",
        "# Search paths (in order of priority)\n",
        "SEARCH_PATHS = [\n",
        "    os.path.join(PROJECT_ROOT, 'data', DATASET_NAME),\n",
        "    os.path.join(PROJECT_ROOT, DATASET_NAME),\n",
        "    os.path.join(DRIVE_ROOT, DATASET_NAME),\n",
        "    os.path.join('/content', DATASET_NAME),\n",
        "    DATASET_NAME,\n",
        "]\n",
        "\n",
        "dataset_path = None\n",
        "for p in SEARCH_PATHS:\n",
        "    if os.path.exists(p):\n",
        "        dataset_path = p\n",
        "        print(f'\u2705 Dataset found: {p}')\n",
        "        break\n",
        "\n",
        "if dataset_path is None:\n",
        "    if IN_COLAB:\n",
        "        print('\u26a0\ufe0f  Dataset not found in Google Drive.')\n",
        "        print('    Please upload earthquake_29_features.csv:')\n",
        "        from google.colab import files\n",
        "        uploaded = files.upload()\n",
        "        for fname in uploaded:\n",
        "            if fname.endswith('.csv'):\n",
        "                dest = os.path.join(PROJECT_ROOT, 'data', fname)\n",
        "                with open(dest, 'wb') as f:\n",
        "                    f.write(uploaded[fname])\n",
        "                dataset_path = dest\n",
        "                print(f'\u2705 Uploaded and saved to: {dest}')\n",
        "                break\n",
        "    else:\n",
        "        raise FileNotFoundError(\n",
        "            f'{DATASET_NAME} not found. '\n",
        "            f'Place it in: {os.path.join(PROJECT_ROOT, \"data\")}'\n",
        "        )\n",
        "\n",
        "if dataset_path is None:\n",
        "    raise FileNotFoundError('No CSV file was uploaded.')\n",
        "\n",
        "# Also copy dataset to project data/ folder for consistency\n",
        "data_dest = os.path.join(PROJECT_ROOT, 'data', DATASET_NAME)\n",
        "if dataset_path != data_dest:\n",
        "    shutil.copy2(dataset_path, data_dest)\n",
        "    print(f'   Copied to: {data_dest}')\n",
        "\n",
        "# Load the CSV\n",
        "df = pd.read_csv(dataset_path)\n",
        "if 'filename' in df.columns:\n",
        "    df = df.drop(columns=['filename'])\n",
        "\n",
        "X = df.drop(columns=['magnitude'])\n",
        "y = df['magnitude']\n",
        "\n",
        "print(f'\\n\U0001f4ca Dataset Summary')\n",
        "print(f'   Samples:           {X.shape[0]:,}')\n",
        "print(f'   Features:          {X.shape[1]}')\n",
        "print(f'   Magnitude range:   {y.min():.1f} \u2013 {y.max():.1f}')\n",
        "print(f'   Unique magnitudes: {y.nunique()}')\n",
        "print(f'   Missing values:    {X.isnull().sum().sum()}')\n",
    ]))

    # ================================================================
    # CELL 8: Data Preprocessing
    # ================================================================
    cells.append(make_md_cell([
        "## 7\ufe0f\u20e3 Data Preprocessing\n",
        "\n",
        "- Handle missing values (drop rows with NaN)\n",
        "- Train/test split (80/20)\n",
        "- Extract raw physics features **before** scaling (needed for physics constraints)\n",
        "- StandardScaler normalisation\n",
        "- Convert to PyTorch tensors\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Preprocessing: clean, split, scale, tensorise\n",
        "# ============================================================\n",
        "\n",
        "# Handle missing values\n",
        "missing = X.isnull().sum().sum()\n",
        "if missing > 0:\n",
        "    print(f'\u26a0\ufe0f  Dropping {missing} rows with missing values')\n",
        "    X = X.dropna()\n",
        "    y = y.loc[X.index]\n",
        "else:\n",
        "    print('\u2705 No missing values detected.')\n",
        "\n",
        "# Read config from project\n",
        "config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')\n",
        "with open(config_path, 'r') as f:\n",
        "    config = yaml.safe_load(f)\n",
        "\n",
        "test_size = config['dataset']['test_size']\n",
        "random_state = config['dataset']['random_state']\n",
        "energy_feature = config['physics']['energy_feature']\n",
        "frequency_feature = config['physics']['frequency_feature']\n",
        "\n",
        "# Train/test split\n",
        "X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(\n",
        "    X, y, test_size=test_size, random_state=random_state\n",
        ")\n",
        "\n",
        "# Extract raw physics features BEFORE scaling\n",
        "raw_energy_train = torch.tensor(X_train_df[energy_feature].values, dtype=torch.float32).to(DEVICE)\n",
        "raw_freq_train = torch.tensor(X_train_df[frequency_feature].values, dtype=torch.float32).to(DEVICE)\n",
        "raw_energy_test = torch.tensor(X_test_df[energy_feature].values, dtype=torch.float32).to(DEVICE)\n",
        "raw_freq_test = torch.tensor(X_test_df[frequency_feature].values, dtype=torch.float32).to(DEVICE)\n",
        "\n",
        "print(f'\\n\U0001f4ca Raw physics features (before scaling):')\n",
        "print(f'   Energy range:    {raw_energy_train.min():.4f} \u2013 {raw_energy_train.max():.4f}')\n",
        "print(f'   Frequency range: {raw_freq_train.min():.4f} \u2013 {raw_freq_train.max():.4f}')\n",
        "\n",
        "# Feature scaling\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train_df)\n",
        "X_test_scaled = scaler.transform(X_test_df)\n",
        "\n",
        "# Convert to tensors and move to device\n",
        "X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)\n",
        "X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)\n",
        "y_train_t = torch.tensor(y_train_s.values, dtype=torch.float32).view(-1, 1).to(DEVICE)\n",
        "y_test_t = torch.tensor(y_test_s.values, dtype=torch.float32).view(-1, 1).to(DEVICE)\n",
        "\n",
        "feature_names = X.columns.tolist()\n",
        "input_dim = X_train_t.shape[1]\n",
        "\n",
        "print(f'\\n\u2705 Data ready (device: {DEVICE}):')\n",
        "print(f'   Train: {X_train_t.shape[0]:,} samples \u00d7 {input_dim} features')\n",
        "print(f'   Test:  {X_test_t.shape[0]:,} samples \u00d7 {input_dim} features')\n",
    ]))

    # ================================================================
    # CELL 9: Model Definitions & Helpers
    # ================================================================
    cells.append(make_md_cell([
        "## 8\ufe0f\u20e3 Model Definitions & Output Helpers\n",
        "\n",
        "Import all model classes from the project source files written to Google Drive.\n",
        "Define helper functions for saving outputs to Drive.\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Import model classes from project modules\n",
        "# ============================================================\n",
        "from src.model import BaselineMLP\n",
        "from src.pinn_model import EarthquakePINN\n",
        "from src.train import train_model\n",
        "from src.pinn_train import train_pinn, get_magnitude_prediction\n",
        "from src.evaluate import (\n",
        "    compute_metrics, plot_loss_curves,\n",
        "    plot_predicted_vs_actual, plot_error_distribution,\n",
        ")\n",
        "from src.utils import configure_paths\n",
        "from physics.physics_loss import compute_physics_loss\n",
        "\n",
        "# --- Configure output paths to point to Google Drive ---\n",
        "configure_paths(PROJECT_ROOT)\n",
        "\n",
        "# --- Output path constants (all point to Google Drive) ---\n",
        "SAVED_MODELS = os.path.join(PROJECT_ROOT, 'saved_models')\n",
        "PLOTS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'plots')\n",
        "PREDICTIONS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'predictions')\n",
        "METRICS_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'metrics')\n",
        "\n",
        "# --- Save helpers ---\n",
        "def save_model_to_drive(model, path):\n",
        "    torch.save(model.state_dict(), path)\n",
        "    print(f'  \U0001f4be Model \u2192 {os.path.basename(path)}')\n",
        "\n",
        "def save_predictions_csv(y_true, y_pred, path):\n",
        "    pd.DataFrame({\n",
        "        'actual_magnitude': y_true.flatten(),\n",
        "        'predicted_magnitude': y_pred.flatten(),\n",
        "        'error': (y_true - y_pred).flatten(),\n",
        "    }).to_csv(path, index=False)\n",
        "    print(f'  \U0001f4be Predictions \u2192 {os.path.basename(path)}')\n",
        "\n",
        "def save_metrics_json(metrics, path):\n",
        "    with open(path, 'w') as f:\n",
        "        json.dump(metrics, f, indent=4)\n",
        "    print(f'  \U0001f4be Metrics \u2192 {os.path.basename(path)}')\n",
        "\n",
        "def save_loss_history(history, path):\n",
        "    with open(path, 'w') as f:\n",
        "        json.dump(history, f, indent=4)\n",
        "    print(f'  \U0001f4be Loss history \u2192 {os.path.basename(path)}')\n",
        "\n",
        "print('\u2705 All model classes and utilities imported.')\n",
        "print(f'   Outputs will be saved to: {PROJECT_ROOT}')\n",
    ]))

    # ================================================================
    # CELL 10: Experiment 1 \u2014 Baseline MLP
    # ================================================================
    cells.append(make_md_cell([
        "## 9\ufe0f\u20e3 Experiment 1/4: Baseline MLP\n",
        "\n",
        "**Architecture:** `Input(29) \u2192 Linear(128) + ReLU \u2192 Linear(64) + ReLU \u2192 Linear(32) + ReLU \u2192 Linear(1)`\n",
        "\n",
        "**Loss:** Mean Squared Error (MSE) \u2014 no physics constraints.\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Train the Baseline MLP (no physics constraints)\n",
        "# ============================================================\n",
        "print('\\n' + '\u2588' * 65)\n",
        "print('  EXPERIMENT 1/4: BASELINE MLP')\n",
        "print('\u2588' * 65 + '\\n')\n",
        "\n",
        "model_baseline = BaselineMLP(input_dim=input_dim).to(DEVICE)\n",
        "print(f'[Model] BaselineMLP \u2014 Parameters: {sum(p.numel() for p in model_baseline.parameters()):,}')\n",
        "print(f'[Model] Device: {DEVICE}')\n",
        "\n",
        "start = time.time()\n",
        "model_baseline, train_losses_baseline, val_losses_baseline = train_model(\n",
        "    model=model_baseline,\n",
        "    X_train=X_train_t,\n",
        "    y_train=y_train_t,\n",
        "    X_val=X_test_t,\n",
        "    y_val=y_test_t,\n",
        "    epochs=config['training']['epochs'],\n",
        "    lr=config['training']['learning_rate'],\n",
        "    print_every=config['training']['print_every'],\n",
        ")\n",
        "time_baseline = time.time() - start\n",
        "\n",
        "# --- Evaluate ---\n",
        "model_baseline.eval()\n",
        "with torch.no_grad():\n",
        "    y_pred_baseline = model_baseline(X_test_t).cpu().numpy().flatten()\n",
        "y_true = y_test_t.cpu().numpy().flatten()\n",
        "\n",
        "metrics_baseline = compute_metrics(y_true, y_pred_baseline)\n",
        "metrics_baseline['training_time_seconds'] = round(time_baseline, 2)\n",
        "metrics_baseline['num_parameters'] = sum(p.numel() for p in model_baseline.parameters())\n",
        "\n",
        "# --- Save ---\n",
        "history_baseline = {'train_total': train_losses_baseline, 'val_total': val_losses_baseline}\n",
        "save_model_to_drive(model_baseline, os.path.join(SAVED_MODELS, 'baseline_mlp.pth'))\n",
        "save_predictions_csv(y_true, y_pred_baseline, os.path.join(PREDICTIONS_DIR, 'baseline_mlp_predictions.csv'))\n",
        "save_metrics_json(metrics_baseline, os.path.join(METRICS_DIR, 'baseline_mlp_metrics.json'))\n",
        "save_loss_history(history_baseline, os.path.join(METRICS_DIR, 'baseline_mlp_loss_history.json'))\n",
        "\n",
        "# --- Plots ---\n",
        "plot_loss_curves(train_losses_baseline, val_losses_baseline, os.path.join(PLOTS_DIR, 'baseline_mlp_loss_curves.png'))\n",
        "plot_predicted_vs_actual(y_true, y_pred_baseline, os.path.join(PLOTS_DIR, 'baseline_mlp_predicted_vs_actual.png'))\n",
        "plot_error_distribution(y_true, y_pred_baseline, os.path.join(PLOTS_DIR, 'baseline_mlp_error_distribution.png'))\n",
        "\n",
        "# Display plots inline\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "fig.suptitle('Baseline MLP \u2014 Results', fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "axes[0].plot(train_losses_baseline, label='Train', color='#2196F3', linewidth=1.5)\n",
        "axes[0].plot(val_losses_baseline, label='Val', color='#FF5722', linewidth=1.5, linestyle='--')\n",
        "axes[0].set_title('Loss Curves'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')\n",
        "axes[0].legend(); axes[0].grid(True, alpha=0.3)\n",
        "\n",
        "axes[1].scatter(y_true, y_pred_baseline, alpha=0.5, s=15, color='#2196F3', edgecolors='none')\n",
        "min_v, max_v = min(y_true.min(), y_pred_baseline.min()), max(y_true.max(), y_pred_baseline.max())\n",
        "axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5)\n",
        "axes[1].set_title('Predicted vs Actual'); axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')\n",
        "axes[1].set_aspect('equal', adjustable='box'); axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "errors = y_true - y_pred_baseline\n",
        "axes[2].hist(errors, bins=40, color='#4CAF50', edgecolor='white', alpha=0.85)\n",
        "axes[2].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')\n",
        "axes[2].set_title('Error Distribution'); axes[2].set_xlabel('Error'); axes[2].set_ylabel('Frequency')\n",
        "axes[2].legend(); axes[2].grid(True, alpha=0.3)\n",
        "\n",
        "fig.tight_layout()\n",
        "plt.show()\n",
        "print('\\n\u2705 Baseline MLP complete.')\n",
    ]))

    # ================================================================
    # Helper: Build PINN experiment cell
    # ================================================================
    def make_pinn_experiment_cell(model_name, display_name, exp_num,
                                  use_attention, use_uncertainty,
                                  description):
        """Build a training + evaluation code cell for a PINN variant."""
        lines = [
            "# ============================================================\n",
            f"# Train {display_name}\n",
            "# ============================================================\n",
            f"print('\\n' + '\u2588' * 65)\n",
            f"print('  EXPERIMENT {exp_num}/4: {display_name.upper()}')\n",
            "print('\u2588' * 65 + '\\n')\n",
            "\n",
            f"model_{model_name} = EarthquakePINN(\n",
            "    input_dim=input_dim,\n",
            "    hidden_layers=config['model']['hidden_layers'],\n",
            f"    use_attention={use_attention},\n",
            f"    use_uncertainty={use_uncertainty},\n",
            f").to(DEVICE)\n",
            f"print(f'[Model] {{model_{model_name}.get_config_string()}}')\n",
            f"print(f'[Model] Parameters: {{model_{model_name}.count_parameters():,}}')\n",
            f"print(f'[Model] Device: {{DEVICE}}')\n",
            "\n",
            "start = time.time()\n",
            f"model_{model_name}, history_{model_name} = train_pinn(\n",
            f"    model=model_{model_name},\n",
            "    X_train=X_train_t, y_train=y_train_t,\n",
            "    X_val=X_test_t, y_val=y_test_t,\n",
            "    raw_energy_train=raw_energy_train,\n",
            "    raw_frequency_train=raw_freq_train,\n",
            "    raw_energy_val=raw_energy_test,\n",
            "    raw_frequency_val=raw_freq_test,\n",
            "    epochs=config['training']['epochs'],\n",
            "    lr=config['training']['learning_rate'],\n",
            "    lambda_energy=config['physics']['lambda_energy'],\n",
            "    lambda_frequency=config['physics']['lambda_frequency'],\n",
            "    energy_coeff_a=config['physics']['energy_coeff_a'],\n",
            "    energy_coeff_b=config['physics']['energy_coeff_b'],\n",
            "    epsilon=config['physics']['epsilon'],\n",
            f"    use_uncertainty={use_uncertainty},\n",
            "    use_physics=True,\n",
            "    print_every=config['training']['print_every'],\n",
            f"    model_name='{model_name}',\n",
            ")\n",
            f"time_{model_name} = time.time() - start\n",
            "\n",
            "# --- Evaluate ---\n",
            f"model_{model_name}.eval()\n",
            "with torch.no_grad():\n",
            f"    preds_{model_name} = model_{model_name}(X_test_t)\n",
        ]

        if use_uncertainty:
            lines += [
                f"y_pred_{model_name} = preds_{model_name}[:, 0].cpu().numpy().flatten()\n",
                f"uncertainty_{model_name} = np.exp(\n",
                f"    torch.clamp(preds_{model_name}[:, 1], min=-2.0, max=2.0).cpu().numpy().flatten()\n",
                ")\n",
            ]
        else:
            lines += [
                f"y_pred_{model_name} = preds_{model_name}.cpu().numpy().flatten()\n",
            ]

        lines += [
            "y_true = y_test_t.cpu().numpy().flatten()\n",
            "\n",
            f"metrics_{model_name} = compute_metrics(y_true, y_pred_{model_name})\n",
            f"metrics_{model_name}['training_time_seconds'] = round(time_{model_name}, 2)\n",
            f"metrics_{model_name}['num_parameters'] = model_{model_name}.count_parameters()\n",
            "\n",
            "# --- Save ---\n",
            f"save_model_to_drive(model_{model_name}, os.path.join(SAVED_MODELS, '{model_name}.pth'))\n",
            f"save_predictions_csv(y_true, y_pred_{model_name}, os.path.join(PREDICTIONS_DIR, '{model_name}_predictions.csv'))\n",
            f"save_metrics_json(metrics_{model_name}, os.path.join(METRICS_DIR, '{model_name}_metrics.json'))\n",
            f"save_loss_history(history_{model_name}, os.path.join(METRICS_DIR, '{model_name}_loss_history.json'))\n",
            "\n",
            "# --- Plots (saved to Drive) ---\n",
            f"plot_loss_curves(history_{model_name}['train_total'], history_{model_name}['val_total'],\n",
            f"                 os.path.join(PLOTS_DIR, '{model_name}_loss_curves.png'))\n",
            f"plot_predicted_vs_actual(y_true, y_pred_{model_name},\n",
            f"                        os.path.join(PLOTS_DIR, '{model_name}_predicted_vs_actual.png'))\n",
            f"plot_error_distribution(y_true, y_pred_{model_name},\n",
            f"                        os.path.join(PLOTS_DIR, '{model_name}_error_distribution.png'))\n",
            "\n",
        ]

        # Inline display
        lines += [
            "# --- Display inline ---\n",
            f"fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            f"fig.suptitle('{display_name} \u2014 Results', fontsize=16, fontweight='bold', y=1.02)\n",
            "\n",
            f"axes[0].plot(history_{model_name}['train_total'], label='Train', color='#2196F3', linewidth=1.5)\n",
            f"axes[0].plot(history_{model_name}['val_total'], label='Val', color='#FF5722', linewidth=1.5, linestyle='--')\n",
            "axes[0].set_title('Loss Curves'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')\n",
            "axes[0].legend(); axes[0].grid(True, alpha=0.3)\n",
            "\n",
            f"axes[1].scatter(y_true, y_pred_{model_name}, alpha=0.5, s=15, color='#4CAF50', edgecolors='none')\n",
            f"min_v, max_v = min(y_true.min(), y_pred_{model_name}.min()), max(y_true.max(), y_pred_{model_name}.max())\n",
            "axes[1].plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5)\n",
            "axes[1].set_title('Predicted vs Actual'); axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')\n",
            "axes[1].set_aspect('equal', adjustable='box'); axes[1].grid(True, alpha=0.3)\n",
            "\n",
            f"errors = y_true - y_pred_{model_name}\n",
            "axes[2].hist(errors, bins=40, color='#4CAF50', edgecolor='white', alpha=0.85)\n",
            "axes[2].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')\n",
            "axes[2].set_title('Error Distribution'); axes[2].set_xlabel('Error'); axes[2].set_ylabel('Frequency')\n",
            "axes[2].legend(); axes[2].grid(True, alpha=0.3)\n",
            "\n",
            "fig.tight_layout()\n",
            "plt.show()\n",
        ]

        # Physics loss components
        lines += [
            "\n",
            "# --- Physics loss component plot ---\n",
            f"epochs_range = range(1, len(history_{model_name}['train_energy']) + 1)\n",
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            f"ax.plot(epochs_range, history_{model_name}['train_energy'], label='Energy Loss', color='#E91E63', linewidth=1.5)\n",
            f"ax.plot(epochs_range, history_{model_name}['train_frequency'], label='Frequency Loss', color='#9C27B0', linewidth=1.5)\n",
            f"ax.plot(epochs_range, history_{model_name}['train_physics'], label='Total Physics Loss', color='#FF5722', linewidth=2, linestyle='--')\n",
            "ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Loss', fontsize=12)\n",
            f"ax.set_title('Physics Loss Components \u2014 {display_name}', fontsize=14, fontweight='bold')\n",
            "ax.legend(fontsize=11); ax.grid(True, alpha=0.3)\n",
            "fig.tight_layout()\n",
            f"fig.savefig(os.path.join(PLOTS_DIR, '{model_name}_physics_losses.png'), dpi=150, bbox_inches='tight')\n",
            "plt.show()\n",
        ]

        if use_attention:
            lines += [
                "\n",
                "# --- Attention weights visualization ---\n",
                f"attn_w = model_{model_name}.get_attention_weights(X_train_t)\n",
                "if attn_w is not None:\n",
                "    mean_w = attn_w.cpu().mean(dim=0).numpy()\n",
                "    sorted_idx = np.argsort(mean_w)\n",
                "    sorted_names = [feature_names[i] for i in sorted_idx]\n",
                "    sorted_weights = mean_w[sorted_idx]\n",
                "\n",
                "    fig, ax = plt.subplots(figsize=(10, 8))\n",
                "    colors_attn = plt.cm.RdYlGn(sorted_weights / sorted_weights.max())\n",
                "    ax.barh(sorted_names, sorted_weights, color=colors_attn, edgecolor='white', linewidth=0.5)\n",
                "    ax.set_xlabel('Attention Weight', fontsize=12)\n",
                f"    ax.set_title('Feature Importance (Attention) \u2014 {display_name}', fontsize=14, fontweight='bold')\n",
                "    ax.grid(True, alpha=0.3, axis='x')\n",
                "    fig.tight_layout()\n",
                f"    fig.savefig(os.path.join(PLOTS_DIR, '{model_name}_attention_weights.png'), dpi=150, bbox_inches='tight')\n",
                "    plt.show()\n",
            ]

        if use_uncertainty:
            lines += [
                "\n",
                "# --- Uncertainty visualization ---\n",
                "fig, ax = plt.subplots(figsize=(10, 6))\n",
                f"scatter = ax.scatter(y_pred_{model_name}, uncertainty_{model_name}, alpha=0.5, s=20,\n",
                f"                     c=uncertainty_{model_name}, cmap='YlOrRd', edgecolors='none')\n",
                "plt.colorbar(scatter, ax=ax, label='Uncertainty (\u03c3)')\n",
                "ax.set_xlabel('Predicted Magnitude', fontsize=12)\n",
                "ax.set_ylabel('Prediction Uncertainty (\u03c3)', fontsize=12)\n",
                f"ax.set_title('Prediction Uncertainty \u2014 {display_name}', fontsize=14, fontweight='bold')\n",
                "ax.grid(True, alpha=0.3)\n",
                "fig.tight_layout()\n",
                f"fig.savefig(os.path.join(PLOTS_DIR, '{model_name}_uncertainty.png'), dpi=150, bbox_inches='tight')\n",
                "plt.show()\n",
            ]

        lines += [
            "\n",
            f"print('\\n\u2705 {display_name} complete.')\n",
        ]

        return lines

    # ================================================================
    # PINN Experiments
    # ================================================================
    cells.append(make_md_cell([
        "## \U0001f51f Experiment 2/4: Basic PINN\n",
        "\n",
        "**Same architecture as Baseline MLP** but with physics-informed loss:\n",
        "\n",
        "```\n",
        "L_total = L_MSE + \u03bb_energy \u00d7 L_energy + \u03bb_freq \u00d7 L_frequency\n",
        "```\n",
    ]))

    cells.append(make_code_cell(
        make_pinn_experiment_cell(
            "basic_pinn", "Basic PINN", 2,
            use_attention=False, use_uncertainty=False,
            description="Physics constraints only",
        )
    ))

    cells.append(make_md_cell([
        "## 1\ufe0f\u20e31\ufe0f\u20e3 Experiment 3/4: Attention PINN\n",
        "\n",
        "Adds a **sigmoid gating attention layer** before the backbone network:\n",
        "\n",
        "```\n",
        "Input \u2192 Attention(Linear + Sigmoid) \u2192 Element-wise multiply \u2192 Backbone \u2192 Output\n",
        "```\n",
        "\n",
        "This learns which of the 29 features are most important for prediction.\n",
    ]))

    cells.append(make_code_cell(
        make_pinn_experiment_cell(
            "attention_pinn", "Attention PINN", 3,
            use_attention=True, use_uncertainty=False,
            description="Physics + attention",
        )
    ))

    cells.append(make_md_cell([
        "## 1\ufe0f\u20e32\ufe0f\u20e3 Experiment 4/4: Full PINN (Attention + Uncertainty)\n",
        "\n",
        "The **complete model** with all components:\n",
        "- Attention-based feature weighting\n",
        "- Dual output head: `[magnitude, log_uncertainty]`\n",
        "- Negative Log-Likelihood (NLL) loss with MSE anchor\n",
        "\n",
        "```\n",
        "L_total = L_NLL + 0.1 \u00d7 L_MSE_anchor + \u03bb_energy \u00d7 L_energy + \u03bb_freq \u00d7 L_frequency\n",
        "```\n",
    ]))

    cells.append(make_code_cell(
        make_pinn_experiment_cell(
            "full_pinn", "Full PINN", 4,
            use_attention=True, use_uncertainty=True,
            description="Physics + attention + uncertainty",
        )
    ))

    # ================================================================
    # Comparison cells
    # ================================================================
    cells.append(make_md_cell(["## 1\ufe0f\u20e33\ufe0f\u20e3 Model Comparison\n"]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Compare all 4 models \u2014 formatted table\n",
        "# ============================================================\n",
        "all_metrics = {\n",
        "    'baseline_mlp': metrics_baseline,\n",
        "    'basic_pinn': metrics_basic_pinn,\n",
        "    'attention_pinn': metrics_attention_pinn,\n",
        "    'full_pinn': metrics_full_pinn,\n",
        "}\n",
        "\n",
        "with open(os.path.join(METRICS_DIR, 'all_experiments_comparison.json'), 'w') as f:\n",
        "    json.dump(all_metrics, f, indent=4)\n",
        "print(f'\U0001f4be Comparison saved \u2192 all_experiments_comparison.json')\n",
        "\n",
        "display_names = {\n",
        "    'baseline_mlp': 'Baseline MLP',\n",
        "    'basic_pinn': 'Basic PINN',\n",
        "    'attention_pinn': 'Attention PINN',\n",
        "    'full_pinn': 'Full PINN',\n",
        "}\n",
        "\n",
        "print('\\n' + '=' * 90)\n",
        "print('  FINAL COMPARISON TABLE')\n",
        "print('=' * 90)\n",
        "print(f'{\"Model\":<25} {\"MAE\":>10} {\"RMSE\":>10} {\"R\u00b2\":>12} {\"Time (s)\":>10} {\"Params\":>10}')\n",
        "print('-' * 90)\n",
        "\n",
        "for name in ['baseline_mlp', 'basic_pinn', 'attention_pinn', 'full_pinn']:\n",
        "    m = all_metrics[name]\n",
        "    print(f'{display_names[name]:<25} {m[\"MAE\"]:>10.6f} {m[\"RMSE\"]:>10.6f} '\n",
        "          f'{m[\"R2\"]:>12.6f} {m[\"training_time_seconds\"]:>10.2f} '\n",
        "          f'{m[\"num_parameters\"]:>10,}')\n",
        "\n",
        "print('=' * 90)\n",
        "\n",
        "best_r2 = max(all_metrics.items(), key=lambda x: x[1]['R2'])\n",
        "best_rmse = min(all_metrics.items(), key=lambda x: x[1]['RMSE'])\n",
        "print(f'\\n\U0001f3c6 Best R\u00b2:   {display_names[best_r2[0]]} ({best_r2[1][\"R2\"]:.6f})')\n",
        "print(f'\U0001f3c6 Best RMSE: {display_names[best_rmse[0]]} ({best_rmse[1][\"RMSE\"]:.6f})')\n",
    ]))

    # Comparison visualizations
    cells.append(make_md_cell(["## 1\ufe0f\u20e34\ufe0f\u20e3 Comparison Visualizations\n", "\n", "Six publication-quality comparison charts.\n"]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# 1. Metrics Comparison Bar Chart (MAE, RMSE, R\u00b2)\n",
        "# ============================================================\n",
        "model_names_disp = ['Baseline MLP', 'Basic PINN', 'Attention PINN', 'Full PINN']\n",
        "model_keys = ['baseline_mlp', 'basic_pinn', 'attention_pinn', 'full_pinn']\n",
        "colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "fig.suptitle('Model Comparison \u2014 Evaluation Metrics', fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "for idx, (metric, title) in enumerate([\n",
        "    ('MAE', 'Mean Absolute Error (MAE)'),\n",
        "    ('RMSE', 'Root Mean Squared Error (RMSE)'),\n",
        "    ('R2', 'R\u00b2 Score'),\n",
        "]):\n",
        "    values = [all_metrics[k][metric] for k in model_keys]\n",
        "    bars = axes[idx].bar(model_names_disp, values, color=colors, edgecolor='white', linewidth=1.5, width=0.6)\n",
        "    axes[idx].set_title(title, fontsize=13, fontweight='bold')\n",
        "    axes[idx].tick_params(axis='x', rotation=20, labelsize=10)\n",
        "    axes[idx].grid(True, alpha=0.3, axis='y')\n",
        "    for bar, val in zip(bars, values):\n",
        "        y_pos = bar.get_height()\n",
        "        va = 'bottom' if y_pos >= 0 else 'top'\n",
        "        axes[idx].text(bar.get_x() + bar.get_width()/2., y_pos,\n",
        "                       f'{val:.4f}', ha='center', va=va, fontsize=9, fontweight='bold')\n",
        "\n",
        "fig.tight_layout()\n",
        "fig.savefig(os.path.join(PLOTS_DIR, 'comparison_metrics.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# 2. Training & Validation Loss Curves \u2014 All Models\n",
        "# ============================================================\n",
        "all_histories = {\n",
        "    'Baseline MLP': history_baseline,\n",
        "    'Basic PINN': history_basic_pinn,\n",
        "    'Attention PINN': history_attention_pinn,\n",
        "    'Full PINN': history_full_pinn,\n",
        "}\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "fig.suptitle('Loss Curve Comparison \u2014 All Models', fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "for (name, hist), color in zip(all_histories.items(), colors):\n",
        "    train_l = hist.get('train_total', [])\n",
        "    val_l = hist.get('val_total', [])\n",
        "    ep = range(1, len(train_l) + 1)\n",
        "    axes[0].plot(ep, train_l, label=name, color=color, linewidth=1.5)\n",
        "    axes[1].plot(ep, val_l, label=name, color=color, linewidth=1.5)\n",
        "\n",
        "axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')\n",
        "axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)\n",
        "axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')\n",
        "axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)\n",
        "\n",
        "fig.tight_layout()\n",
        "fig.savefig(os.path.join(PLOTS_DIR, 'comparison_loss_curves.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# 3. Predicted vs Actual \u2014 Side-by-Side\n",
        "# ============================================================\n",
        "models_for_plot = [\n",
        "    ('Baseline MLP', model_baseline, False),\n",
        "    ('Basic PINN', model_basic_pinn, False),\n",
        "    ('Attention PINN', model_attention_pinn, False),\n",
        "    ('Full PINN', model_full_pinn, True),\n",
        "]\n",
        "\n",
        "fig, axes = plt.subplots(1, 4, figsize=(22, 5))\n",
        "fig.suptitle('Predicted vs Actual \u2014 All Models', fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "for idx, (name, model, is_unc) in enumerate(models_for_plot):\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        p = model(X_test_t)\n",
        "        pv = p[:, 0].cpu().numpy().flatten() if is_unc else p.cpu().numpy().flatten()\n",
        "    tv = y_test_t.cpu().numpy().flatten()\n",
        "    ax = axes[idx]\n",
        "    ax.scatter(tv, pv, alpha=0.5, s=15, color=colors[idx], edgecolors='none')\n",
        "    mn, mx = min(tv.min(), pv.min()), max(tv.max(), pv.max())\n",
        "    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)\n",
        "    ax.set_title(name, fontsize=12, fontweight='bold')\n",
        "    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')\n",
        "    ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.3)\n",
        "\n",
        "fig.tight_layout()\n",
        "fig.savefig(os.path.join(PLOTS_DIR, 'comparison_predicted_vs_actual.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# 4. Error Distribution \u2014 Overlay\n",
        "# ============================================================\n",
        "fig, ax = plt.subplots(figsize=(12, 6))\n",
        "\n",
        "for idx, (name, model, is_unc) in enumerate(models_for_plot):\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        p = model(X_test_t)\n",
        "        pv = p[:, 0].cpu().numpy().flatten() if is_unc else p.cpu().numpy().flatten()\n",
        "    errs = y_test_t.cpu().numpy().flatten() - pv\n",
        "    ax.hist(errs, bins=40, alpha=0.5, label=name, color=colors[idx], edgecolor='white', linewidth=0.5)\n",
        "\n",
        "ax.set_xlabel('Prediction Error (Actual \u2212 Predicted)', fontsize=13)\n",
        "ax.set_ylabel('Frequency', fontsize=13)\n",
        "ax.set_title('Error Distribution Comparison', fontsize=15, fontweight='bold')\n",
        "ax.legend(fontsize=11); ax.grid(True, alpha=0.3)\n",
        "fig.tight_layout()\n",
        "fig.savefig(os.path.join(PLOTS_DIR, 'comparison_error_distribution.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# 5. Residual Plots \u2014 Error vs Actual Magnitude\n",
        "# ============================================================\n",
        "fig, axes = plt.subplots(1, 4, figsize=(22, 5))\n",
        "fig.suptitle('Residual Plots \u2014 All Models', fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "for idx, (name, model, is_unc) in enumerate(models_for_plot):\n",
        "    model.eval()\n",
        "    with torch.no_grad():\n",
        "        p = model(X_test_t)\n",
        "        pv = p[:, 0].cpu().numpy().flatten() if is_unc else p.cpu().numpy().flatten()\n",
        "    tv = y_test_t.cpu().numpy().flatten()\n",
        "    residuals = tv - pv\n",
        "    ax = axes[idx]\n",
        "    ax.scatter(tv, residuals, alpha=0.5, s=15, color=colors[idx], edgecolors='none')\n",
        "    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)\n",
        "    ax.set_title(name, fontsize=12, fontweight='bold')\n",
        "    ax.set_xlabel('Actual Magnitude'); ax.set_ylabel('Residual')\n",
        "    ax.grid(True, alpha=0.3)\n",
        "\n",
        "fig.tight_layout()\n",
        "fig.savefig(os.path.join(PLOTS_DIR, 'comparison_residuals.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# 6. Training Time & Parameter Count Comparison\n",
        "# ============================================================\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "fig.suptitle('Computational Cost Comparison', fontsize=16, fontweight='bold', y=1.02)\n",
        "\n",
        "times = [all_metrics[k]['training_time_seconds'] for k in model_keys]\n",
        "bars = axes[0].bar(model_names_disp, times, color=colors, edgecolor='white', linewidth=1.5, width=0.6)\n",
        "for bar, val in zip(bars, times):\n",
        "    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),\n",
        "                 f'{val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
        "axes[0].set_title('Training Time (seconds)', fontsize=13, fontweight='bold')\n",
        "axes[0].tick_params(axis='x', rotation=20, labelsize=10); axes[0].grid(True, alpha=0.3, axis='y')\n",
        "\n",
        "params = [all_metrics[k]['num_parameters'] for k in model_keys]\n",
        "bars = axes[1].bar(model_names_disp, params, color=colors, edgecolor='white', linewidth=1.5, width=0.6)\n",
        "for bar, val in zip(bars, params):\n",
        "    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),\n",
        "                 f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')\n",
        "axes[1].set_title('Number of Parameters', fontsize=13, fontweight='bold')\n",
        "axes[1].tick_params(axis='x', rotation=20, labelsize=10); axes[1].grid(True, alpha=0.3, axis='y')\n",
        "\n",
        "fig.tight_layout()\n",
        "fig.savefig(os.path.join(PLOTS_DIR, 'comparison_training_cost.png'), dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "print('\\n\u2705 All comparison visualizations saved to Google Drive.')\n",
    ]))

    # Save & Download cells
    cells.append(make_md_cell(["## 1\ufe0f\u20e35\ufe0f\u20e3 Save Models & Scaler\n"]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Save scaler for future inference\n",
        "# ============================================================\n",
        "scaler_path = os.path.join(SAVED_MODELS, 'scaler.pkl')\n",
        "with open(scaler_path, 'wb') as f:\n",
        "    pickle.dump(scaler, f)\n",
        "print(f'\U0001f4be Scaler saved \u2192 {os.path.basename(scaler_path)}')\n",
        "\n",
        "print('\\n\U0001f4c1 Saved Models:')\n",
        "for f in sorted(os.listdir(SAVED_MODELS)):\n",
        "    size = os.path.getsize(os.path.join(SAVED_MODELS, f)) / 1024\n",
        "    print(f'   {f} ({size:.1f} KB)')\n",
        "\n",
        "print('\\n\U0001f4c1 Output Plots:')\n",
        "for f in sorted(os.listdir(PLOTS_DIR)):\n",
        "    print(f'   \U0001f4c8 {f}')\n",
        "\n",
        "print('\\n\U0001f4c1 Metrics & Histories:')\n",
        "for f in sorted(os.listdir(METRICS_DIR)):\n",
        "    print(f'   \U0001f4cb {f}')\n",
        "\n",
        "print('\\n\U0001f4c1 Predictions:')\n",
        "for f in sorted(os.listdir(PREDICTIONS_DIR)):\n",
        "    print(f'   \U0001f4ca {f}')\n",
        "\n",
        "print(f'\\n\u2705 Everything saved to: {PROJECT_ROOT}')\n",
    ]))

    cells.append(make_md_cell([
        "## 1\ufe0f\u20e36\ufe0f\u20e3 Export & Download Results\n",
        "\n",
        "Creates a ZIP file of the complete project with all source code, trained models,\n",
        "metrics, reports, predictions, and plots. Automatically provides a download link.\n",
    ]))

    cells.append(make_code_cell([
        "# ============================================================\n",
        "# Create ZIP file of complete project for download\n",
        "# ============================================================\n",
        "zip_name = 'Earthquake_PINN_Project'\n",
        "zip_path = os.path.join(DRIVE_ROOT, zip_name)\n",
        "\n",
        "shutil.make_archive(zip_path, 'zip', PROJECT_ROOT)\n",
        "zip_file = zip_path + '.zip'\n",
        "zip_size_mb = os.path.getsize(zip_file) / (1024 * 1024)\n",
        "\n",
        "print(f'\U0001f4e6 ZIP created: {os.path.basename(zip_file)}')\n",
        "print(f'   Size: {zip_size_mb:.1f} MB')\n",
        "print(f'   Location: {zip_file}')\n",
        "\n",
        "total_files = sum(len(f) for _, _, f in os.walk(PROJECT_ROOT))\n",
        "print(f'   Total files: {total_files}')\n",
        "\n",
        "if IN_COLAB:\n",
        "    print('\\n\u2b07\ufe0f  Starting download...')\n",
        "    from google.colab import files\n",
        "    files.download(zip_file)\n",
        "else:\n",
        "    print(f'\\n\U0001f4c2 ZIP saved at: {zip_file}')\n",
    ]))

    # Summary
    cells.append(make_md_cell([
        "## \u2705 Experiment Complete\n",
        "\n",
        "### All 4 Models Trained & Evaluated\n",
        "\n",
        "| Model | Description |\n",
        "|---|---|\n",
        "| **Baseline MLP** | Pure data-driven, MSE loss (control) |\n",
        "| **Basic PINN** | + Gutenberg\u2013Richter energy + frequency\u2013magnitude constraints |\n",
        "| **Attention PINN** | + Learned feature importance via sigmoid gating |\n",
        "| **Full PINN** | + Aleatoric uncertainty estimation (NLL loss) |\n",
        "\n",
        "### Saved Outputs (Google Drive)\n",
        "\n",
        "All results are in `MyDrive/Earthquake_PINN_Project/`:\n",
        "\n",
        "| Category | Contents |\n",
        "|---|---|\n",
        "| \U0001f9e0 `saved_models/` | 4 trained models (`.pth`) + scaler (`.pkl`) |\n",
        "| \U0001f4ca `outputs/predictions/` | 4 prediction CSV files |\n",
        "| \U0001f4c8 `outputs/plots/` | 30+ visualisation PNG files |\n",
        "| \U0001f4cb `outputs/metrics/` | JSON metrics, loss histories, comparison |\n",
        "| \U0001f4c4 `reports/` | 4 research report markdown files |\n",
        "| \U0001f4e6 `Earthquake_PINN_Project.zip` | Complete project archive |\n",
        "\n",
        "### Reproducibility\n",
        "\n",
        "- Random seed: `42` (Python, NumPy, PyTorch)\n",
        "- CUDNN deterministic: `True`\n",
        "- All configuration in `config/config.yaml`\n",
        "\n",
        "### Share with Collaborators\n",
        "\n",
        "1. Share this `.ipynb` notebook\n",
        "2. Share the `earthquake_29_features.csv` dataset\n",
        "3. Collaborator clicks **Runtime \u2192 Run All** \u2014 everything is automatic\n",
    ]))

    # Build final notebook JSON
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "name": "Earthquake_PINN.ipynb",
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
            },
            "language_info": {
                "name": "python",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }

    return notebook


def main():
    """Generate the Colab notebook."""
    notebook = build_notebook()

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_dir, "Earthquake_PINN.ipynb")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    num_cells = len(notebook["cells"])
    md_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "markdown")
    code_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")

    print(f"[OK] Notebook generated: {output_path}")
    print(f"     Size: {size_kb:.1f} KB")
    print(f"     Total cells: {num_cells} ({md_cells} markdown + {code_cells} code)")


if __name__ == "__main__":
    main()
