"""Export the ALS counterfactual ISDs and the model-vs-KM errors as CSV.

Reproduces notebooks/isd_counterfactual.ipynb and notebooks/model_vs_km.ipynb
from the survival-als project (same data split, trained MENSA model,
patient index 321 = "Mr. Smith", counterfactual values and log-rank tests).
It only READS from survival-als and writes CSVs to scripts/data/.

Usage (conda env py39-als):
    python scripts/export_als_curves.py C:/Users/cml/Desktop/survival-als
"""
import os
import sys
from pathlib import Path

ALS_ROOT = Path(sys.argv[1]).resolve()
OUT = Path(__file__).resolve().parent / "data"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ALS_ROOT / "src"))
os.chdir(ALS_ROOT / "src")

import joblib                                  # noqa: E402
import numpy as np                             # noqa: E402
import pandas as pd                            # noqa: E402
import random                                  # noqa: E402
import torch                                   # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402

import config as cfg                           # noqa: E402
from tools.data_loader import get_data_loader  # noqa: E402
from tools.preprocessor import Preprocessor    # noqa: E402
from utility.survival import make_time_bins    # noqa: E402

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cpu")

# ---------------------------------------------------------------- as in cell 0
dl = get_data_loader("proact").load_data()
train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                  test_size=0.2, random_state=0)
X_train = pd.DataFrame(train_dict["X"], columns=dl.columns)
X_test = pd.DataFrame(test_dict["X"], columns=dl.columns)
preprocessor = Preprocessor(cat_feat_strat="mode", num_feat_strat="mean", scaling_strategy="standard")
transformer = preprocessor.fit(X_train, cat_feats=dl.cat_features, num_feats=dl.num_features,
                               one_hot=True, fill_value=-1)
model = joblib.load(Path(cfg.MODELS_DIR) / "mensa_proact.pkl")
time_bins = make_time_bins(train_dict["T"], event=None, dtype=dtype).to(device)
time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
tb = time_bins.numpy()

EVENTS = ["Speech", "Swallowing", "Handwriting", "Walking", "Dyspnea"]
sample_idx = 321  # Mr. Smith
full = dl.get_data()[0]


def predict(x_df, event_id):
    x = torch.tensor(transformer.transform(x_df).to_numpy(), device=device, dtype=dtype)
    return np.asarray(model.predict(x, time_bins, risk=event_id)).mean(axis=0)


def logrank(g1_idx, g2_idx, event_id):
    r = logrank_test(test_dict["T"][g1_idx][:, event_id], test_dict["T"][g2_idx][:, event_id],
                     event_observed_A=test_dict["E"][g1_idx][:, event_id],
                     event_observed_B=test_dict["E"][g2_idx][:, event_id])
    return r.p_value


def scenario(feature, val_a, val_b, g1_mask, g2_mask, lab_a, lab_b, name):
    base = X_test.iloc[[sample_idx]]
    xa, xb = base.copy(), base.copy()
    xa[feature] = val_a
    xb[feature] = val_b
    g1, g2 = X_test.loc[g1_mask].index, X_test.loc[g2_mask].index
    rows, ps = [], []
    for k, ev in enumerate(EVENTS):
        sa, sb = predict(xa, k), predict(xb, k)
        for t, a, b in zip(tb, sa, sb):
            rows.append(dict(feature=name, event=ev, time=t, s_a=a, s_b=b))
        ps.append(dict(feature=name, event=ev, p_value=logrank(g1, g2, k),
                       label_a=lab_a, label_b=lab_b))
    return rows, ps


rows, ps = [], []
f = "Subject_used_Riluzole"
r, p = scenario(f, "Yes", "No", X_test[f] == "Yes", X_test[f] == "No",
                "Riluzole", "no Riluzole", "riluzole"); rows += r; ps += p
f = "Site_of_Onset"
r, p = scenario(f, "Limb", "Bulbar", X_test[f] == "Limb", X_test[f] == "Bulbar",
                "Limb onset", "Bulbar onset", "onset"); rows += r; ps += p
f = "FVC_Mean"
m, s = full[f].mean(), full[f].std()
r, p = scenario(f, m + s, m - s, X_test[f] >= m + s, X_test[f] < m - s,
                f"FVC = {m + s:.1f}", f"FVC = {m - s:.1f}", "fvc"); rows += r; ps += p
f = "ALSFRS_R_Total"
m, s = full[f].mean(), full[f].std()
r, p = scenario(f, m + s, m - s, X_test[f] >= m + s, X_test[f] < m - s,
                f"ALSFRS-R = {m + s:.1f}", f"ALSFRS-R = {m - s:.1f}", "alsfrs"); rows += r; ps += p

pd.DataFrame(rows).to_csv(OUT / "als_counterfactual_curves.csv", index=False)
pd.DataFrame(ps).to_csv(OUT / "als_counterfactual_logrank.csv", index=False)

# ---------------------------------------------- model vs KM (model_vs_km.ipynb)
res = pd.read_csv(Path(cfg.RESULTS_DIR) / "model_results.csv")
res = res[res["DatasetName"] == "proact"]
res[["ModelName", "EventId", "Seed", "MAEM", "MAEKM"]].to_csv(OUT / "als_model_vs_km.csv", index=False)
print("wrote", OUT / "als_counterfactual_curves.csv", "and", OUT / "als_model_vs_km.csv")
