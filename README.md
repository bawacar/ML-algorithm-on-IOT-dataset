# IoT Botnet Detection on the N-BaIoT Dataset

Machine learning detection of Mirai and Gafgyt UDP-flood botnet traffic using the
N-BaIoT behavioural feature set. Supporting code for CMP7239 coursework.

Eleven algorithms across three families, tuned under one methodology, then
evaluated on a balanced split **and** on the 304,413 rows no stage of modelling
ever touched.

---

## Headline results

| Family | Model | Balanced test | Novel holdout |
|---|---|---|---|
| Supervised | Logistic Regression | 0.9987 | 0.7711 |
| Supervised | Decision Tree | 0.9998 | 0.8078 |
| Supervised | Random Forest | 1.0000 | **0.8377** |
| Supervised | Support Vector Machine | 0.9987 | 0.7847 |
| Unsupervised | K-Means | ARI 0.429 | — |
| Unsupervised | Agglomerative Clustering | ARI 0.477 | — |
| Unsupervised | **DBSCAN** | **ARI 0.986** | — |
| Deep learning | MLP | 0.9987 | — |
| Deep learning | Deep DNN | 0.9982 | — |
| Deep learning | 1D-CNN | 0.5021 | — |
| Deep learning | Autoencoder (benign-scaled) | — | **0.9832** |

**Recommended for deployment: Decision Tree.** 96.8 microseconds per packet at
the median, 237 at p99, 2.1 KB on disk, and benign/Mirai recall indistinguishable
from Random Forest.

---

## Five findings worth reading the code for

1. **Two features do the work.** A depth-4 Decision Tree reaches 0.9998 macro-F1
   using only `MI_dir_L0.01_weight` and `H_L0.1_variance`. The other 113 features
   have exactly zero importance.

2. **Gafgyt is a converged class.** At float64, 94.6% of Gafgyt rows are distinct.
   After the float32 downcast, only **110** distinct rows remain in the entire
   dataset. This single fact explains the DBSCAN result, the 1D-CNN failure, and
   the duplicate-leakage problem below.

3. **32.8% of the holdout is not really unseen.** Because duplicates were retained,
   99.9% of Gafgyt holdout rows are exact copies of training rows. Only 102 Gafgyt
   rows are genuinely novel. Any holdout score reported without this check is
   inflated.

4. **The Autoencoder failure was a preprocessing bug, not a model limitation.**
   Scaled on all three classes it scores 0.1989 macro-F1 with 0.0066 attack recall.
   Scaled on benign traffic alone, holding everything else fixed, it scores
   **0.9832** with **0.9999** attack recall. A scaler fitted on attack data
   normalises attacks into the benign range before the model sees them.

5. **Throughput and latency give opposite answers.** Random Forest costs 1.43 µs
   per sample in batches of 10,000, but **3,656 µs** per packet at batch=1. Only
   the batch=1 figure describes inline detection.

---

## Repository contents

```
.
├── IoT_Botnet_Detection_Analysis.ipynb   # full analysis notebook
├── iot_botnet_detection_analysis.py      # script export
├── save_models.py                        # saves every trained model + scaler
├── requirements.txt
├── .gitignore
└── saved_models/
    ├── scaler.joblib                     # REQUIRED with any sklearn model
    ├── logistic_regression.joblib
    ├── decision_tree.joblib              # the recommended production model
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

The report points here for detail kept out of the write-up.

| Report section | Notebook section | Detail held here |
|---|---|---|
| 3.2 Redundancy | EDA | Correlation heatmap, PCA scree plot, full variance table |
| 3.5 Gafgyt convergence | Gafgyt diagnostic | float32 vs float64 uniqueness test, per-component spread |
| 5.1 Tuning | Tuning | **Full hyperparameter grids and every stage winner** |
| 5.2 Results | Final models | Per-class classification reports for all eleven models |
| 5.4 Robustness | Robustness | Per-fold cross-validation scores, clustering stability runs |
| 7.2 Generalisation | §9 Production eval | Duplicate-leakage check, full holdout scoring |
| 7.4 Autoencoder | §9 Production eval | Scaler A/B experiment, reconstruction error distributions |
| 7.5 Deployment | §9 Production eval | Latency percentiles, batch sweep, model footprints |

---

## Dataset

The three CSV files are **not included**, as they exceed GitHub's size limits.

- `benign_traffic.csv` (62,154 rows)
- `gafgyt_attacks_udp.csv` (104,011 rows)
- `mirai_attacks_udp.csv` (156,248 rows)

Download from the UCI Machine Learning Repository (N-BaIoT) and place in the
project root.

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook IoT_Botnet_Detection_Analysis.ipynb
```

Run top to bottom. `RANDOM_STATE = 42` throughout, so results reproduce exactly.
The deep learning sections need TensorFlow and are far faster on a GPU runtime.

---

## Using the production model

The scaler must be loaded with the model. Predictions made on unscaled features
are meaningless.

```python
import joblib

model  = joblib.load("saved_models/decision_tree.joblib")
scaler = joblib.load("saved_models/scaler.joblib")

preds = model.predict(scaler.transform(X_new))
# 0 = benign, 1 = gafgyt_udp, 2 = mirai_udp
```

For the Autoencoder, note finding 4 above: fit the scaler on **benign traffic
only**, never on mixed classes.

```python
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import numpy as np

ae = keras.models.load_model("saved_models/autoencoder.keras")
benign_scaler = StandardScaler().fit(X_benign_reference)   # benign rows ONLY

Xs  = benign_scaler.transform(X_new)
err = np.mean((Xs - ae.predict(Xs))**2, axis=1)
is_attack = err > threshold          # threshold = 95th pct of benign training error
```

---

## Methodology notes

- **Sampling.** 6,000 rows per class (18,000 total), the largest that keeps
  Agglomerative Clustering and SVM workable on one CPU. Held-out test set 4,500
  rows. The remaining 304,413 rows are used for the production evaluation.
- **Scaling.** `StandardScaler` fitted on the training split only, refitted inside
  each fold during cross-validation via a `Pipeline`.
- **Tuning.** Staged search: most influential hyperparameter first, then others
  with earlier winners fixed. Assumes weak interaction, stated as a limitation.
- **Duplicates.** 34.4% of rows are exact duplicates, retained deliberately since
  removing them would alter the class-conditional density. The consequence for
  holdout evaluation is measured explicitly rather than ignored.

---

## Author

Barakah Bolaji Obileye 
