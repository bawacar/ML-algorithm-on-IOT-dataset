# IoT Botnet Detection on the N-BaIoT Dataset

Machine learning detection of Mirai and Gafgyt UDP-flood botnet traffic using the
N-BaIoT behavioural feature set. Supporting code for CMP7239 coursework.

Eleven algorithms are implemented across three families and compared under one
consistent methodology, with robustness checks for overfitting, split-dependence
and dimensionality reduction.

---

## Summary of results

| Family | Model | Key metric |
|---|---|---|
| Supervised | Logistic Regression | 0.9987 macro-F1 |
| Supervised | Decision Tree | 0.9998 macro-F1 |
| Supervised | **Random Forest** | **1.0000 macro-F1** |
| Supervised | Support Vector Machine | 0.9987 macro-F1 |
| Unsupervised | K-Means | ARI 0.429 |
| Unsupervised | Agglomerative Clustering | ARI 0.477 |
| Unsupervised | **DBSCAN** | **ARI 0.986** |
| Deep learning | MLP | 0.9987 macro-F1 |
| Deep learning | Deep DNN | 0.9982 macro-F1 |
| Deep learning | 1D-CNN | 0.5021 macro-F1 (failed on Gafgyt) |
| Deep learning | Autoencoder | 0.2426 macro-F1 (failed to flag attacks) |

Three findings are worth highlighting:

1. A depth-4 Decision Tree reaches 0.9998 macro-F1 using only **two** of the 115
   features, so the engineered statistics carry nearly all of the signal.
2. Selecting K-Means settings by silhouette score gives ARI 0.429, while a
   configuration the same score ranked lower gives ARI 0.798. Internal validation
   is necessary but not sufficient.
3. The two most capable deep learning architectures performed the worst, each
   defeated by a property of the data rather than by any lack of capacity.

---

## Repository contents

```
.
├── IoT_Botnet_Detection_Analysis.ipynb   # full analysis notebook
├── iot_botnet_detection_analysis.py      # script export of the notebook
├── save_models.py                        # saves every trained model + scaler
├── requirements.txt
├── .gitignore
└── saved_models/                         # trained models (see below)
    ├── scaler.joblib                     # REQUIRED to use any sklearn model
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── random_forest.joblib
    ├── svm.joblib
    ├── kmeans.joblib
    ├── agglomerative.joblib
    ├── dbscan.joblib
    ├── mlp.keras
    ├── deep_dnn.keras
    ├── cnn.keras
    └── autoencoder.keras
```

---

## Where to find things

The report refers to the notebook for detail that was moved out of the write-up.
This table maps report sections to notebook sections.

| Report section | Notebook section | What is there |
|---|---|---|
| 3.2 Data integrity and redundancy | 3.x EDA | Correlation heatmap, PCA scree plot |
| 3.5 Gafgyt in projection | 3.x Gafgyt diagnostic | float32 vs float64 uniqueness test, PC5 analysis |
| 5.2 Hyperparameter tuning | 5.x Tuning | **Full grids and the winning value at every stage** |
| 5.3 Results | 5.x Final models | Per-class classification reports for all models |
| 5.5 Overfitting and CV | 5.x Robustness | Per-fold cross-validation scores |
| 5.6 Dimensionality reduction | 5.x PCA test | Full 115 vs 11-component comparison |
| 7.1 Timing | 7.x Timing | Train and predict timing measurements |

---

## Dataset

The three N-BaIoT CSV files are **not included** in this repository, because they
exceed GitHub's file size limits.

Required files:

- `benign_traffic.csv` (approx. 62,000 rows)
- `gafgyt_attacks_udp.csv` (approx. 104,000 rows)
- `mirai_attacks_udp.csv` (approx. 156,000 rows)

Download them from the UCI Machine Learning Repository (N-BaIoT dataset) and place
them in the project root before running the notebook.

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook IoT_Botnet_Detection_Analysis.ipynb
```

Run the cells top to bottom. `RANDOM_STATE = 42` is fixed throughout, so results
are reproducible. The deep learning sections require TensorFlow and are much
faster on a GPU runtime such as Google Colab.

---

## Using a saved model

Every sklearn model was trained on scaled features, so the **scaler must be loaded
with it**. Predictions made without the scaler will be wrong.

```python
import joblib

model  = joblib.load("saved_models/random_forest.joblib")
scaler = joblib.load("saved_models/scaler.joblib")

X_scaled = scaler.transform(X_new)      # X_new: raw 115-feature rows
predictions = model.predict(X_scaled)   # 0 = benign, 1 = gafgyt_udp, 2 = mirai_udp
```

For the Keras models:

```python
from tensorflow import keras
import joblib

mlp    = keras.models.load_model("saved_models/mlp.keras")
scaler = joblib.load("saved_models/scaler.joblib")

predictions = mlp.predict(scaler.transform(X_new)).argmax(axis=1)
```

---

## Methodology notes

- **Sampling.** Modelling uses a stratified sample of 6,000 rows per class
  (18,000 total), the largest that keeps Agglomerative Clustering and SVM
  workable on a single CPU. The held-out test set is 4,500 rows.
- **Scaling.** `StandardScaler` is fitted on the training split only. In
  cross-validation it is refitted inside each fold via a `Pipeline`, so no
  test-fold information leaks into training.
- **Tuning.** A staged search is used: the most influential hyperparameter is
  swept first, then further hyperparameters with earlier winners held fixed.
  This assumes weak interaction between hyperparameters, which is stated as a
  limitation in the report.
- **Duplicates.** 34.4 percent of rows are exact duplicates. These were retained
  deliberately, since removing them would alter the class-conditional density
  the models are meant to learn.

---

## Author

Barakah Bolaji Obileye — CMP7239 coursework.
