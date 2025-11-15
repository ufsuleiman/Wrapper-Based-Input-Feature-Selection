import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# filenames and column names

poro_file = 'Porosity_Preprocessed_Data.xlsx'         
perm_file = 'Permeability_Preprocessed_Data.xlsx' 

poro_features = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO']
perm_features = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO','RD','RM','RT']

poro_target = 'PHIF'
perm_target = 'KLOGH'

# If permeability is highly skewed, set True to log-transform target during training

log_transform_perm = True

# General MLP hyperparams

hidden_layer_sizes_poro = (64, 32)
hidden_layer_sizes_perm = (128, 64)  
activation = 'relu'
solver = 'adam'   
learning_rate_init = 1e-3
alpha = 1e-4      
batch_size = 32

# training loop settings (manual epoch loop)

n_epochs_search = 200      
patience = 20              
random_state = 42

# output filenames

porosity_history_jpg = "porosity_training_history.jpg"
permeability_history_jpg = "permeability_training_history.jpg"
pred_vs_actual_jpg = "predicted_vs_actual_both.jpg"
poro_summary_txt = "porosity_model_summary.txt"
perm_summary_txt = "permeability_model_summary.txt"
poro_model_file = "porosity_mlp.joblib"
perm_model_file = "permeability_mlp.joblib"

# Utilities

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

def save_model_summary(filename, model, feature_list, notes=""):
    with open(filename, 'w') as f:
        f.write("Model summary\n")
        f.write("=================\n")
        f.write(f"Hidden layers: {model.hidden_layer_sizes}\n")
        f.write(f"Activation: {model.activation}\n")
        f.write(f"Solver: {model.solver}\n")
        f.write(f"Learning rate init: {model.learning_rate_init}\n")
        f.write(f"Alpha (L2): {model.alpha}\n")
        f.write(f"Batch size: {model.batch_size}\n")
        f.write(f"Max iter (per .fit call): {model.max_iter}\n")
        f.write(f"Warm start: {model.warm_start}\n")
        f.write(f"Random state: {model.random_state}\n")
        f.write(f"Features ({len(feature_list)}): {feature_list}\n")
        if notes:
            f.write("\nNotes:\n")
            f.write(notes + "\n")

# Function: manual epoch training with warm_start to record val metrics

def train_mlp_with_history(X, y, hidden_layers, model_name="model",
                           activation='relu', solver='adam', lr=1e-3, alpha=1e-4,
                           batch_size=32, max_epochs=200, patience=20, random_state=42,
                           verbose=False):
    """
    Returns: trained_model, history_dict (train_rmse_list, val_rmse_list), X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Splitted into train/val/test 70/15/15

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.17647, random_state=random_state)

    # Create MLPRegressor with warm_start=True and max_iter=1 to iterate epochs manually

    model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                         activation=activation,
                         solver=solver,
                         alpha=alpha,
                         batch_size=batch_size,
                         learning_rate_init=lr,
                         max_iter=1,          
                         warm_start=True,
                         random_state=random_state)

    train_rmse_list = []
    val_rmse_list = []
    best_val = np.inf
    best_epoch = -1
    best_weights = None
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.fit(X_train, y_train)   

        # Predictions

        y_tr_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_rmse = rmse(y_train, y_tr_pred)
        val_rmse = rmse(y_val, y_val_pred)

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

        if verbose:
            print(f"{model_name} Epoch {epoch}: train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}")

        # early stopping on validation RMSE

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_epoch = epoch
            epochs_since_improve = 0

            # storing model weights by saving temporarily to disk object

            joblib.dump(model, f"temp_{model_name}_best.joblib")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs). Best epoch {best_epoch}, best_val_rmse={best_val:.4f}")
            break

    # loading best model

    best_model = joblib.load(f"temp_{model_name}_best.joblib")

    # cleaning-up temp file
    try:
        os.remove(f"temp_{model_name}_best.joblib")
    except:
        pass

    history = {'train_rmse': train_rmse_list, 'val_rmse': val_rmse_list, 'best_epoch': best_epoch}

    return best_model, history, (X_train, X_val, X_test, y_train, y_val, y_test)

# Training porosity and permeability

def run_baseline_all():

    # loading files

    if not os.path.exists(poro_file):
        raise FileNotFoundError(f"Porosity file not found: {poro_file}")
    if not os.path.exists(perm_file):
        raise FileNotFoundError(f"Permeability file not found: {perm_file}")

    df_poro = pd.read_excel(poro_file)
    df_perm = pd.read_excel(perm_file)

    # ensuring if columns exist

    for c in poro_features + [poro_target]:
        if c not in df_poro.columns:
            raise KeyError(f"Column {c} not found in {poro_file}")
    for c in perm_features + [perm_target]:
        if c not in df_perm.columns:
            raise KeyError(f"Column {c} not found in {perm_file}")

    # preparing arrays

    X_poro = df_poro[poro_features].values
    y_poro = df_poro[poro_target].values.astype(float)

    X_perm = df_perm[perm_features].values
    y_perm = df_perm[perm_target].values.astype(float)
    if log_transform_perm:
        y_perm_trainable = np.log1p(y_perm) 
    else:
        y_perm_trainable = y_perm.copy()

    # Porosity training

    print("Training porosity MLP (all features)...")
    poro_model, poro_history, poro_splits = train_mlp_with_history(X_poro, y_poro,
                                                                   hidden_layers=hidden_layer_sizes_poro,
                                                                   activation=activation, solver=solver,
                                                                   lr=learning_rate_init, alpha=alpha,
                                                                   batch_size=batch_size,
                                                                   max_epochs=n_epochs_search,
                                                                   patience=patience,
                                                                   random_state=random_state,
                                                                   verbose=True)
    
    # unpacking

    X_tr_p, X_val_p, X_test_p, y_tr_p, y_val_p, y_test_p = poro_splits

    # final metrics on test

    y_test_pred_poro = poro_model.predict(X_test_p)
    poro_test_rmse = rmse(y_test_p, y_test_pred_poro)
    poro_test_r2 = r2_score(y_test_p, y_test_pred_poro)

    print(f"Porosity test RMSE: {poro_test_rmse:.4f}, R2: {poro_test_r2:.4f}")

    # saving model and summary

    joblib.dump(poro_model, poro_model_file)
    save_model_summary(poro_summary_txt, poro_model, poro_features, notes="Baseline porosity MLP (all features)")

    # plotting training history

    plt.figure(figsize=(6,4))
    plt.plot(poro_history['train_rmse'], label='train RMSE')
    plt.plot(poro_history['val_rmse'], label='val RMSE')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.title('Porosity Training History (RMSE)')
    plt.legend(); plt.grid(True)
    plt.savefig(porosity_history_jpg, dpi=200, bbox_inches='tight')
    plt.close()

    # Permeability training

    print("\nTraining permeability MLP (all features)...")
    perm_model, perm_history, perm_splits = train_mlp_with_history(X_perm, y_perm_trainable,
                                                                   hidden_layers=hidden_layer_sizes_perm,
                                                                   activation=activation, solver=solver,
                                                                   lr=learning_rate_init, alpha=alpha,
                                                                   batch_size=batch_size,
                                                                   max_epochs=n_epochs_search,
                                                                   patience=patience,
                                                                   random_state=random_state,
                                                                   verbose=True)
    X_tr_perm, X_val_perm, X_test_perm, y_tr_perm, y_val_perm, y_test_perm = perm_splits

    # reverse transforming permeability for reporting

    if log_transform_perm:
        y_test_perm_orig = np.expm1(y_test_perm)
        y_test_pred_perm_log = perm_model.predict(X_test_perm)
        y_test_pred_perm_orig = np.expm1(y_test_pred_perm_log)
        perm_test_rmse_orig = rmse(y_test_perm_orig, y_test_pred_perm_orig)
        perm_test_r2_orig = r2_score(y_test_perm_orig, y_test_pred_perm_orig)
        print(f"Permeability test RMSE (original units): {perm_test_rmse_orig:.4f}, R2: {perm_test_r2_orig:.4f}")
    else:
        y_test_pred_perm = perm_model.predict(X_test_perm)
        perm_test_rmse = rmse(y_test_perm, y_test_pred_perm)
        perm_test_r2 = r2_score(y_test_perm, y_test_pred_perm)
        print(f"Permeability test RMSE: {perm_test_rmse:.4f}, R2: {perm_test_r2:.4f}")

    # saving permeability model & summary

    joblib.dump(perm_model, perm_model_file)
    save_model_summary(perm_summary_txt, perm_model, perm_features, notes="Baseline permeability MLP (all features)")

    # plotting permeability training history

    plt.figure(figsize=(6,4))
    plt.plot(perm_history['train_rmse'], label='train RMSE')
    plt.plot(perm_history['val_rmse'], label='val RMSE')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.title('Permeability Training History (RMSE)')
    plt.legend(); plt.grid(True)
    plt.savefig(permeability_history_jpg, dpi=200, bbox_inches='tight')
    plt.close()

    # Combined predicted vs actual scatter (both poro & perm) on same plot

    plt.figure(figsize=(6,6))

    # porosity test scatter

    plt.scatter(y_test_p, y_test_pred_poro, s=12, alpha=0.6, label='Porosity')

    # permeability test scatter: using original units if log transformed

    if log_transform_perm:
        plt.scatter(y_test_perm_orig, y_test_pred_perm_orig, s=12, alpha=0.6, label='Permeability (orig units)')
    else:
        plt.scatter(y_test_perm, y_test_pred_perm, s=12, alpha=0.6, label='Permeability')

    # identity line (generic)

    mn = min(np.min(y_test_p), np.min(y_test_perm_orig) if log_transform_perm else np.min(y_test_perm))
    mx = max(np.max(y_test_p), np.max(y_test_perm_orig) if log_transform_perm else np.max(y_test_perm))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.title('Predicted vs Actual (Porosity & Permeability)')
    plt.grid(True)
    plt.savefig(pred_vs_actual_jpg, dpi=200, bbox_inches='tight')
    plt.close()

    # Printing final summary

    print("\nSaved files:")
    print("-", porosity_history_jpg)
    print("-", permeability_history_jpg)
    print("-", pred_vs_actual_jpg)
    print("-", poro_summary_txt)
    print("-", perm_summary_txt)
    print("-", poro_model_file)
    print("-", perm_model_file)

# run

if __name__ == "__main__":
    run_baseline_all()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Loading datasets (same files used for training)

df_poro = pd.read_excel('Porosity_Preprocessed_Data.xlsx')
df_perm = pd.read_excel('Permeability_Preprocessed_Data.xlsx')
poro_features = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO']
perm_features = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO','RD','RM','RT']
poro_target = 'PHIF'
perm_target = 'KLOGH'

# 1) Checking if any input column is identical (or nearly identical) to the target

for col in poro_features:
    if col in df_poro.columns:
        same = np.allclose(df_poro[col].fillna(0).values, df_poro[poro_target].fillna(0).values)
        corr = df_poro[col].corr(df_poro[poro_target])
        print(f"Porosity feature '{col}' corr with target: {corr:.4f}, identical? {same}")

for col in perm_features:
    if col in df_perm.columns:
        same = np.allclose(df_perm[col].fillna(0).values, df_perm[perm_target].fillna(0).values)
        corr = df_perm[col].corr(df_perm[perm_target])
        print(f"Perm feature '{col}' corr with target: {corr:.4f}, identical? {same}")

# 2) Checking basic stats: std and variance of the targets (if variance is tiny, R2 can be misleading)

print("\nPorosity target stats:")
print(df_poro[poro_target].describe())
print("\nPermeability target stats:")
print(df_perm[perm_target].describe())

# 3) Computing baseline RMSE (predicting the mean)

y = df_poro[poro_target].values
baseline_rmse = np.sqrt(np.mean((y - y.mean())**2))
print("\nPorosity baseline RMSE (predict mean):", baseline_rmse)

y = df_perm[perm_target].values
baseline_rmse_perm = np.sqrt(np.mean((y - y.mean())**2))
print("Permeability baseline RMSE (predict mean):", baseline_rmse_perm)

# 4) Confirming test/train rows are disjoint:

from sklearn.model_selection import train_test_split
X = df_poro[poro_features]
y = df_poro[poro_target]
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Making sure there are no exact duplicates between training and testing

n_dups = len(pd.merge(X_train_full.reset_index(), X_test.reset_index(), how='inner', on=poro_features))
print("\nNumber of exact duplicates between train and test (porosity):", n_dups)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Porosity baseline linear

X_rhob = df_poro[['RHOB']].values
y = df_poro['PHIF'].values
lr = LinearRegression().fit(X_rhob, y)
y_pred_lr = lr.predict(X_rhob)
print("Linear (RHOB) R2:", r2_score(y, y_pred_lr), "RMSE:", np.sqrt(mean_squared_error(y, y_pred_lr)))

from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
scores = -cross_val_score(model, df_poro[poro_features], df_poro[poro_target],
                          cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
print("5-fold CV RMSEs:", scores, "mean:", scores.mean())

from sklearn.inspection import permutation_importance
res = permutation_importance(poro_model, X_test_p, y_test_p, n_repeats=20, random_state=42)
feat_importances = sorted(zip(poro_features, res.importances_mean), key=lambda x: -x[1])
print("Permutation importances (porosity):", feat_importances)

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(df_poro[poro_features], df_poro[poro_target])
print("Multivariate linear R2:", lr.score(df_poro[poro_features], df_poro[poro_target]))
