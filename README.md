# 🚗 Used Car Predictor

## 📌 Project Overview
`used-car-predictor` is a machine learning project designed to predict the prices of used cars. The project follows a structured and reproducible workflow using **Poetry** for dependency management, **Hydra** for configuration handling, **DVC** for dataset versioning, and **MLflow** for experiment tracking.

---

## 📌 Used Car Predictor

`used-car-predictor` contains:

- A **Poetry** file (`pyproject.toml`) for managing dependencies.
- A **Hydra** configuration file (`configs/config.yaml`) for handling experiment settings.
- **DVC** metadata (`.dvc/` and `datasets.dvc`) to track datasets.
- **MLflow** integration to track model training and evaluation.

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
   dvc remote add myremote gs://mlops-assignment-dvc-bucket
   dvc remote modify myremote credentialpath $GOOGLE_APPLICATION_CREDENTIALS
   ```

4. Pull the dataset:  
   ```bash
   dvc pull
   ```

This will download the necessary dataset files to your local environment.

---

## 🚀 Installation & Setup

### **1️⃣ Change to Project Directory**
```bash
cd used-car-predictor
```

### **2️⃣ Set Up the Virtual Environment with Poetry**
```bash
poetry install
```

### **3️⃣ Activate the Virtual Environment**
```bash
poetry shell
```

### **4️⃣ Pull the Dataset using DVC**
Ensure you have set up your **Google Cloud credentials**, then run:

```bash
dvc pull
```

### **5️⃣ Run the Jupyter Notebook**
Start Jupyter and open the notebook:

```bash
jupyter notebook
```

Open **`notebooks/ML Pipeline.ipynb`** and execute the cells.

---

## ⚙️ Modifying Hydra Configuration
The **Hydra** configuration file is located at:
```
configs/config.yaml
```
This file controls the parameters for the **PyCaret** `setup()` function. You can modify it to change settings like imputation strategies, normalization, feature selection, and outlier removal.

To apply new changes, edit `configs/config.yaml` before running the notebook.

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| **Poetry** | Dependency management |
| **Hydra** | Configuration management |
| **DVC** | Dataset versioning |
| **MLflow** | Experiment tracking |
| **PyCaret** | Automated machine learning |

---

## 📌 Commands Quick Reference

| Task | Command |
|------|---------|
| Install dependencies | `poetry install` |
| Activate virtual environment | `poetry shell` |
| Pull dataset from GCS | `dvc pull` |
| Run Jupyter Notebook | `jupyter notebook` |
| Modify Hydra config | Edit `configs/config.yaml` |
| Track changes with DVC | `dvc add datasets/` |

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 🙌 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

---

## 📬 Contact

For any questions, please contact **misterworker** on GitHub.
