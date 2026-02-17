# Quantum Support Vector Machine (QSVM) with Quantum Kernels

This repository contains a Python implementation of a **Quantum Support Vector Machine (QSVM)** using **quantum kernel methods** based on statevector simulation. The script evaluates multiple quantum feature maps, computes quantum kernel Gram matrices, trains a classical SVM, and reports classification performance across different dataset sizes.

The implementation is intended for **experimental and benchmarking studies** in quantum machine learning.

---

## Overview

The workflow implemented in this script is:

1. Load classical datasets from CSV files  
2. Apply **PCA dimensionality reduction**  
3. Embed classical data into quantum states using configurable **quantum feature maps**  
4. Compute **quantum kernels** via statevector overlaps  
5. Train a **classical SVM** using the quantum kernel  
6. Evaluate classification performance  
7. Log results to CSV and optionally to **MLflow**  
8. (Optional) Export quantum circuits as images  

---

## Quantum Kernel Method

For each pair of samples \(x_i, x_j\):

1. A parameterized quantum circuit \(U(x)\) prepares a state  
   \[
   |\psi(x)\rangle = U(x)\,|0\rangle
   \]
2. The kernel value is computed as:
   \[
   K(x_i, x_j) = |\langle \psi(x_i) | \psi(x_j) \rangle|^2
   \]
3. This kernel matrix is passed to a classical SVM (`sklearn.svm.SVC`)

All quantum computations are performed using **Qiskit statevector simulation**.

---

## Supported Quantum Feature Maps

The script includes multiple feature maps, including:

- **Amplitude embedding**
- **Pauli feature maps** with configurable Pauli strings:
  - `ZZ`, `XX`, `YY`
  - `XYZ`
  - `Z + ZZ`
  - Custom multi-layer re-uploading circuits

Feature maps are defined using `qiskit.circuit.library.PauliFeatureMap`.

---

## Dataset Structure

The script expects datasets in the following directory structure:

```
dataset/
└── efficiency_study/
    ├── train/
    │   ├── x_n_<SIZE>.csv
    │   └── y_n_<SIZE>.csv
    └── test/
        ├── x_test<SUFFIX>.csv
        └── y_test<SUFFIX>.csv
```

Where:
- `<SIZE>` corresponds to training set size (e.g. `40`, `100`, `200`)
- `<SUFFIX>` may be empty or dataset-specific

---

## Output

### Metrics
The following metrics are computed and logged:

- Training accuracy  
- Test accuracy  
- F1 score  
- Precision  
- Recall  
- Balanced accuracy  
- Class-wise accuracy:
  - SEP  
  - PPT  
  - NPPT  

### Results Storage
- CSV results written to:
  ```
  ./results/<dim>/qsvm_sizetest.csv
  ```
- Optional MLflow tracking (disabled by default)

### Circuit Visualization
If enabled, quantum feature map circuits are exported as PNG images to:
```
results/circuits/
```

---

## Requirements

### Python version
- Python **3.9+** recommended

### Dependencies

Key dependencies include:

- `numpy`
- `pandas`
- `scikit-learn`
- `qiskit`
- `pennylane`
- `tensorflow`
- `matplotlib`
- `mlflow` (optional)

Example installation:

```bash
pip install numpy pandas scikit-learn qiskit pennylane tensorflow matplotlib mlflow
```

---

## Configuration

Important parameters can be modified directly in the script:

- Dataset sizes:
  ```python
  size = [40, 100, 200]
  ```
- Feature maps:
  ```python
  FEATURE_MAP = [amplitude_embedding_circuit, pauli_zz_circuit]
  ```
- Number of qubits and PCA components  
- MLflow enable/disable flag  
- Output verbosity  

---

## How to Run

From the project root:

```bash
python qsvm_experiment.py
```

(The script name may be adjusted to match your file name.)

The script will:
1. Execute all configured experiments  
2. Save metrics to CSV  
3. Optionally generate circuit diagrams  

---

## Notes

- Kernel computation scales as \(O(N^2)\) and uses **statevector simulation**
- Intended for **small to medium datasets**
- No quantum hardware execution (simulator only)
- MLflow logging is disabled by default

---
