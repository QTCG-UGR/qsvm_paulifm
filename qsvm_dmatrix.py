import os
import pandas as pd
import numpy as np
import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt
import time 
import fcntl
from functools import partial

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, balanced_accuracy_score

import mlflow
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import PauliFeatureMap
from qiskit.visualization import circuit_drawer

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
#
# Appends a dictionary of results to a CSV file
def append_to_csv(d, file_path):
    # Create a DataFrame with the provided data
    df = pd.DataFrame(data=d)
    
    # Open the CSV file in append mode
    with open(file_path, 'a') as csvfile:
        # Acquire a file lock
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        
        # Append the DataFrame to the CSV file
        df.to_csv(csvfile, header=False, index=False)
        
        # Release the file lock
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)
# Computes and prints the accuracy of predictions for each type of data (SEP, PPT, NPPT) separately
def detailed_accuracy(y_test_pred, dsize):
  test_size=dsize
  pred_split=np.array_split(y_test_pred,3)
  y_sep_pred=pred_split[0]
  y_ppt_pred=pred_split[1]
  y_nppt_pred=pred_split[2]

  score_sep = accuracy_score(y_sep_pred, np.full(test_size, 0))
  score_ppt= accuracy_score(y_ppt_pred, np.full(test_size,1))
  score_nppt= accuracy_score(y_nppt_pred, np.full(test_size,1))

  print("SEP accuracy: ", score_sep)
  print("PPT accuracy: ", score_ppt)
  print("NPPT accuracy: ", score_nppt)
  
  return {'SEP_acc': score_sep, 'PPT_acc': score_ppt, 'NPPT_acc': score_nppt}
# Prints the overall performance metrics of the model on the training and test data
def performance(y_train_pred, y_train, y_test_pred, y_test, dsize):
  tr_acc = accuracy_score(y_train_pred, y_train)
  test_acc = accuracy_score(y_test_pred, y_test)

  tr_f1 = f1_score(y_train, y_train_pred)
  test_f1 = f1_score(y_test, y_test_pred)

  print("Train accuracy: ", tr_acc)
  print("Train F-1 score: ", tr_f1)

  print("\nTest accuracy: ", test_acc)
  print("Test F-1 score: ", test_f1)

  # Test accuracy per type
  print("\nTest accuracy broken down per type")
  detailed_accuracy(y_test_pred, dsize)

  print("\n")
# Helper function to convert values to float for MLflow logging, returns None for non-numerical values
def to_real(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

# -----------------------------
# KERNEL CIRCUITS
# -----------------------------
def amplitude_embedding_circuit(x):
    qc = QuantumCircuit(nqubits_amplitude)

    # Normalized vector
    x_norm = x / np.linalg.norm(x)

    # Add embedding
    qc.initialize(x_norm, qc.qubits)

    return qc

def pauli_zrz_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['Z', 'ZZ'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_xyz_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['X', 'Y','Z'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_xyz_circuit_reup(x):
  paulis = ['X', 'Y','Z']
  layer_slices = np.array_split(x, fm_reps)
  
  qc_total = QuantumCircuit(nqubits_pauli)
  
  for x_slice in layer_slices:
    # Create a new feature map for this layer
    qc_layer = PauliFeatureMap(nqubits_pauli, reps=1, paulis=paulis)
    
    # Get number of parameters needed by this feature map
    num_params = len(qc_layer.parameters)
    
    # Pad slice with zeros if it's shorter than required
    x_padded = np.pad(x_slice, (0, num_params - len(x_slice)), 'constant')
    
    # Assign parameters
    qc_layer = qc_layer.assign_parameters(x_padded)
    
    # Append this layer to the total circuit
    qc_total.compose(qc_layer, inplace=True)

  #print(qc_total.decompose().draw())

  return qc_total

def pauli_zxx_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['Z', 'XX'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_xx_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['XX'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_xy_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['XY'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_xz_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['XZ'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_yx_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['YX'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_yy_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['YY'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_yz_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['YZ'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_zx_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['ZX'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_zy_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['ZY'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_zz_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['ZZ'])
  qc1 = qc.assign_parameters(x)

  return qc1

def pauli_zyyzxz_circuit(x):
  qc = PauliFeatureMap(nqubits_pauli, reps=fm_reps, paulis=['Z', 'YY', 'ZXZ'])
  qc1 = qc.assign_parameters(x)

  return qc1

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
tf.keras.backend.set_floatx('float64')
np.random.seed(42)
os.environ["OMP_NUM_THREADS"] = '8'

# MLflow configuration
ML_EXPERIMENT_NAME = "QSVM_density_2gates"
verbose = False
mlflow_enabled = False
if mlflow_enabled:
  #mlflow.set_tracking_uri("http://127.0.0.1:5000/")  #Default local MLflow server
  mlflow.set_tracking_uri("http://192.168.2.152:5000/") #Remote MLflow server
  mlflow.set_experiment(ML_EXPERIMENT_NAME)

PRINT_CIRCUITS = True

# Dimension of the bipartite system
dim = '3x3'
folder = 'efficiency_study'
csv_file_path = './results/'+dim+'/qsvm_sizetest.csv'
y_train = np.empty((0,))
y_test = np.empty((0,))

# All tests
#size=      [40,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#cat_size = [13,100, 100, 100, 100, 100, 100, 100, 100, 100, 100]#EA test_39.csv=13, Default:100
#size_test= ['_39','', '', '', '', '', '', '', '', '', '']

# Tailor made test
size=      [100, 200, 300, 400, 500, 600]
cat_size = [100, 100, 100, 100, 100, 100]
size_test= ['', '', '', '', '', '']

#FEATURE_MAP = [amplitude_embedding_circuit,  pauli_xx_circuit, pauli_xy_circuit, pauli_xz_circuit, pauli_yx_circuit, pauli_yy_circuit, pauli_yz_circuit, pauli_zx_circuit, pauli_zy_circuit, pauli_zz_circuit, pauli_zrz_circuit, pauli_xyz_circuit, pauli_zxx_circuit, pauli_zyyzxz_circuit]#All
FEATURE_MAP = [amplitude_embedding_circuit, pauli_zz_circuit, pauli_xx_circuit, pauli_yy_circuit, pauli_zxx_circuit, pauli_zrz_circuit, pauli_xyz_circuit]
fm_reps = 2

nqubits_amplitude=3
n_pca_amplitude=8

nqubits_pauli=8
n_pca_pauli=8

# -----------------------------
# OPERATIONAL FUNCTIONS
# -----------------------------
#
# The kernel circuit to compute the kernel value for two samples a and b.
def kernel_circ(a,b, embedding_fn):
    qc_a = embedding_fn(a)
    qc_b = embedding_fn(b)

    psi_a = Statevector.from_instruction(qc_a)
    psi_b = Statevector.from_instruction(qc_b)

    inner = psi_a.data.conj().dot(psi_b.data)
    return np.abs(inner)**2

# The kernel computes the Gram MAtrix for every pair of samples in A and B,
# applying the feature map to each sample, and calculating the inner product of the resulting quantum states
def qkernel(A, B, featuremap):
  GramMat = np.zeros((len(A), len(B)))
  if verbose: print(f"Creating Gram Matrix for {len(A)}x{len(B)} samples...")
  
  _ind = 0
  for i, a in enumerate(A):
      for j, b in enumerate(B):
          GramMat[i, j] = kernel_circ(a, b, embedding_fn=featuremap)
          if verbose: print(f"Calculated element {_ind+1} / {len(A)*len(B)}", end='\r')
          _ind += 1
  if verbose: print("\nGram Matrix completed.")
  return GramMat

# Logging experiment attributes to MLflow
def log_mlflow_attributes(dim, fm, fm_reps, train_data, test_data):
    mlflow.log_param("dataset length", len(train_data) + len(test_data))
    mlflow.log_param("Dimensionality", dim)
    mlflow.log_param("Feature_map", (str)(fm))
    mlflow.log_param("Fm_reps", (str)(fm_reps))
    mlflow.log_param("PCA_amplitude", (str)(n_pca_amplitude))
    mlflow.log_param("PCA_pauli", (str)(n_pca_pauli))

# Function to print the circuits of the feature maps and save them as images
def print_circuits():
  out_dir = "results/circuits"
  os.makedirs(out_dir, exist_ok=True)
  
  filename1 = os.path.join(out_dir, "pauli_zz_feature_map.png")
  abs_path1 = os.path.abspath(filename1)
  qc1 = pauli_zz_circuit(np.random.rand(nqubits_pauli))
  circuit_drawer(qc1.decompose(), output='mpl', filename=filename1)
  print(f"Pauli ZZ feature map circuit saved to:\n{abs_path1}")
  
  filename2 = os.path.join(out_dir, "pauli_zrz_feature_map.png")
  abs_path2 = os.path.abspath(filename2)
  qc2 = pauli_zrz_circuit(np.random.rand(nqubits_pauli))
  circuit_drawer(qc2.decompose(), output='mpl', filename=filename2)
  print(f"Pauli ZRZ feature map circuit saved to:\n{abs_path2}")

# Logging experiment results to MLflow
def log_mlflow_results(results):
    for key, value in results.items():
      if isinstance(value, list):
        for i, v in enumerate(value):
            real_v = to_real(v)
            if real_v is not None:
                mlflow.log_metric(f"{key}_{i}", real_v)
            else:
                print(f"Skipping non-numerical metric: {key}_{i}={v}")
      else:
        real_v = to_real(value)
        if real_v is not None:
            mlflow.log_metric(key, real_v)
        else:
            print(f"Skipping non-numerical metric: {key}={value}")

# Logging experiment results to CSV report file
def log_results(y_train_pred, y_train, y_test_pred, y_test, start, end, dsize, cat_s, fm_name):
  # Metrics scores we will save in a pandas dataframe
  acc_scores=[]
  f1_scores=[]
  prec_scores=[]
  rec_scores=[]
  bacc_scores=[]
  sep_scores=[]
  ppt_scores=[]
  nppt_scores=[]

  tr_acc = accuracy_score(y_train_pred, y_train)
  acc_scores.append(accuracy_score(y_test_pred, y_test))
  f1_scores.append(f1_score(y_test, y_test_pred))
  prec_scores.append(precision_score(y_test, y_test_pred))
  rec_scores.append(recall_score(y_test, y_test_pred))
  bacc_scores.append(balanced_accuracy_score(y_test, y_test_pred))
  acc_per_type=detailed_accuracy(y_test_pred, cat_s)
  sep_scores.append(acc_per_type['SEP_acc'])
  ppt_scores.append(acc_per_type['PPT_acc'])
  nppt_scores.append(acc_per_type['NPPT_acc'])

  performance(y_train_pred, y_train, y_test_pred, y_test,cat_s)

  # Create a dictionary to sumarise metric's scores
  d={'size': dsize, 'PPT_ratio': '1', 'acc_train': tr_acc, 'accuracy': acc_scores, 'f1': f1_scores,
    'precision': prec_scores, 'recall': rec_scores, 'balanced_accuracy': bacc_scores,
    'SEP_accuracy': sep_scores, 'PPT_accuracy': ppt_scores, 'NPPT_accuracy': nppt_scores,
    'time': end-start, 'kernel': (str)(fm_name)}

  if mlflow_enabled:
    log_mlflow_results(d)

  append_to_csv(d, csv_file_path)

# Load the dataset from CSV files, apply PCA for dimensionality reduction and return the training and test data
def load_dataset(featuremap, dssize, dssize_test):
  x_train = np.genfromtxt('./dataset/'+folder+'/train/x_n_'+str(dssize)+'.csv', delimiter=",",dtype=None)
  x_test = np.genfromtxt('./dataset/'+folder+'/test/x_test'+str(dssize_test)+'.csv', delimiter=",",dtype=None)

  if featuremap == amplitude_embedding_circuit:
      n_pca = n_pca_amplitude
  elif featuremap.__name__.startswith("pauli"):
      n_pca = n_pca_pauli
  else:
    raise ValueError(f"Feature {featuremap} map not supported")
  
  pca = PCA(n_components = n_pca)
  xs_train = pca.fit_transform(x_train)
  xs_test = pca.transform(x_test)

  y_train = np.genfromtxt('./dataset/'+folder+'/train/y_n_'+str(dssize)+'.csv', delimiter=",",dtype=None)
  y_test = np.genfromtxt('./dataset/'+folder+'/test/y_test'+str(dssize_test)+'.csv', delimiter=",",dtype=None)

  return xs_train, xs_test, y_train, y_test

# Execute the experiments for all combinations of dataset sizes and feature maps, train the SVM, compute predictions and log results
def execute_experiments():
  if not (len(size) == len(cat_size) == len(size_test)):
      raise ValueError("size, cat_size, and size_test must all have the same length")

  for sz, cs, st in zip(size, cat_size, size_test):
    for fm in FEATURE_MAP:
      print(f"Training for {sz} records, with feature map: {(str)(fm.__name__)}", )
      
      xs_train, xs_test, y_train, y_test = load_dataset(fm, sz, st)
      qkernel_fixed = partial(qkernel, featuremap=fm)
      
      start= time.time()
      # Training of the SVM, compute predictions and metrics 
      if mlflow_enabled:
          with mlflow.start_run(run_name=datetime.now().strftime('%Y-%m-%d %H:%M:%S')):
              log_mlflow_attributes(dim, fm.__name__, fm_reps, xs_train, xs_test)
              svm = SVC(kernel = qkernel_fixed).fit(xs_train, y_train)
              y_train_pred=svm.predict(xs_train)
              y_test_pred=svm.predict(xs_test)
              log_results(y_train_pred, y_train, y_test_pred, y_test, start, time.time(), sz, cs, fm.__name__)
      else:
          svm = SVC(kernel = qkernel_fixed).fit(xs_train, y_train)
          y_train_pred=svm.predict(xs_train)
          y_test_pred=svm.predict(xs_test)
          log_results(y_train_pred, y_train, y_test_pred, y_test, start, time.time(), sz, cs, fm.__name__)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
  execute_experiments()
  if PRINT_CIRCUITS:
    print_circuits()
