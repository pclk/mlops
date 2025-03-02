# 🚗 Used Car Predictor

## 📌 Project Overview
`used-car-predictor` is a machine learning project designed to predict the prices of used cars. The project follows a structured and reproducible workflow using **Poetry** for dependency management, **Hydra** for configuration handling, **DVC** for dataset versioning, and **MLflow** for experiment tracking.

---

## 📌 Clone the Repository

```bash
git clone https://github.com/pclk/mlops
```

---

## 📌 Used Car Predictor

`used-car-predictor` contains:

- A **Poetry** file (`pyproject.toml`) for managing dependencies.
- A **Hydra** configuration file (`configs/config.yaml`) for handling experiment settings.
- **DVC** metadata (`.dvc/` and `datasets.dvc`) to track datasets.
- **MLflow** integration to track model training and evaluation.

## 🚀 Installation & Setup

### **1️⃣ Change to Project Directory**
```bash
cd mlops/used-car-predictor
```

### **2️⃣ Set Up the Virtual Environment with Poetry**
```bash
poetry install
```

### **3️⃣ Activate the Virtual Environment (If it isn't already)**
```bash
poetry env activate
```

### **4️⃣ Run the Jupyter Notebook**
Start Jupyter and open the notebook:

```bash
jupyter notebook
```

Open **`notebooks/ML Pipeline.ipynb`**

---

### **🔹 Getting the Dataset**
The dataset is stored in **Google Cloud Storage (GCS)** under the remote name **`myremote`** (`gs://mlops-assignment-dvc-bucket`).  

1. First, authenticate using your **Google Cloud Service Account key**.  
   This key is provided by **misterworker** (GitHub username). If you do not have it, please contact him.

2. Set up the environment variable for the service account key:  
   ```powershell
   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\your-service-key.json"
   ```
   (For **Command Prompt** use `set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-service-key.json`)

3. Add the DVC remote (if not already configured):  
   ```bash
   poetry run dvc remote add myremote gs://mlops-assignment-dvc-bucket
   poetry run dvc remote modify myremote credentialpath $GOOGLE_APPLICATION_CREDENTIALS
   ```

4. Pull the dataset:  
   ```bash
   poetry run dvc pull
   ```

This will download the necessary dataset files to your local environment.

---

## ⚙️ Modifying Hydra Configuration
The **Hydra** configuration file is located at:
```
configs/config.yaml
```
This file controls the parameters for the **PyCaret** `setup()` function. You can modify it to change settings like imputation strategies, normalization, feature selection, and outlier removal.

To apply new changes, edit `configs/config.yaml` before running the notebook.

---

## 🚀 Modifying & Deploying BentoML Service

The BentoML service is located in:
```
used-car-predictor/bentoml/service.py
```

### **1️⃣ Install Dependencies**
Navigate to the `bentoml` directory and install dependencies:

```bash
cd used-car-predictor/bentoml
pip install -r requirements.txt
```

### **2️⃣ Modify `service.py`**
Edit `service.py` as needed to update the model serving logic.

### **3️⃣ Deploy the Service**
Refer to the official BentoML documentation for the latest deployment steps:
🔗 [BentoML Deployment Guide](https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/create-deployments.html)

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| **Poetry** | Dependency management |
| **Hydra** | Configuration management |
| **DVC** | Dataset versioning |
| **MLflow** | Experiment tracking |
| **PyCaret** | Automated machine learning |
| **BentoML** | Model deployment |

---

## 📌 Commands Quick Reference

| Task | Command |
|------|---------|
| Clone repository | `git clone https://github.com/pclk/mlops` |
| Install dependencies | `poetry install` |
| Activate virtual environment | `poetry shell` |
| Pull dataset from GCS | `dvc pull` |
| Run Jupyter Notebook | `jupyter notebook` |
| Modify Hydra config | Edit `configs/config.yaml` |
| Track changes with DVC | `dvc add datasets/` |
| Install BentoML dependencies | `pip install -r requirements.txt` |

---

## 📬 Contact

For any questions, please contact **misterworker** or **pclk** on GitHub.
