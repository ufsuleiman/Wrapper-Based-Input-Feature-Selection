import os
import random
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.utils import set_random_seed

# Reproducibility

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED) 

# Helpers

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def percent_rmse(y_true, y_pred):
    rm = rmse(y_true, y_pred)
    mean_y = np.mean(y_true)
    return 100.0 * rm / mean_y if mean_y != 0 else np.nan

def build_mlp(input_dim, hidden_units=(64,32), activation='relu', dropout=0.0, optimizer='adam', lr=1e-3):
    """
    Build MLP using explicit Input() to avoid input_shape warning.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units[0], activation=activation))
    if dropout and dropout > 0:
        model.add(Dropout(dropout))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation))
        if dropout and dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    opt = Adam(learning_rate=lr) if optimizer == 'adam' else RMSprop(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=[])
    return model

def save_model_summary(model, filepath):
    """
    Save model.summary() to a text file using UTF-8 encoding.
    Falls back to writing with errors='replace' if needed.
    """
    # capturing summary to a string

    with io.StringIO() as buf:
        model.summary(print_fn=lambda s: buf.write(s + '\n'))
        summary_str = buf.getvalue()

    # trying to write with UTF-8

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_str)
    except Exception as e:

        # writing using system default but replacing non-encodable chars

        with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
            f.write(summary_str)

        # print warning to notice fallback happened

        print(f"Warning: writing model summary with replacement for non-UTF8 chars due to: {e}")


def check_overfitting(history, window=10, threshold=1.15):
    """
    Very simple check: compute mean train loss and val loss over last `window` epochs.
    If val_loss > threshold * train_loss => possible overfitting.
    Returns a tuple (is_overfitting, train_mean, val_mean, message)
    """
    train_losses = history.history['loss']
    val_losses = history.history['val_loss'] if 'val_loss' in history.history else None
    if val_losses is None:
        return (False, None, None, "No validation history available.")
    train_mean = np.mean(train_losses[-window:]) if len(train_losses) >= window else np.mean(train_losses)
    val_mean = np.mean(val_losses[-window:]) if len(val_losses) >= window else np.mean(val_losses)
    if val_mean > threshold * train_mean:
        msg = ("Possible overfitting: validation loss (%.5f) is > %.2fx training loss (%.5f) "
               "over the last %d epochs." % (val_mean, threshold, train_mean, window))
        return (True, train_mean, val_mean, msg)
    else:
        msg = ("No strong overfitting detected: validation loss (%.5f) vs training loss (%.5f) "
               "over the last %d epochs." % (val_mean, train_mean, window))
        return (False, train_mean, val_mean, msg)


# Random search

def random_search_hyperparams(X, y, param_dist, n_iter=20, val_size=0.15, test_size=0.15,
                              epochs_search=40, batch_size=32, patience=6, verbose=0):
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                      test_size=val_size/(1-test_size), random_state=SEED)

    def sample_params(dist):
        ps = {}
        for k, v in dist.items():
            ps[k] = random.choice(v)
        return ps

    results = []
    best = None
    best_score = np.inf

    for i in range(n_iter):
        params = sample_params(param_dist)
        model = build_mlp(input_dim=X.shape[1],
                          hidden_units=params.get('hidden_units', (64,32)),
                          activation=params.get('activation','relu'),
                          dropout=params.get('dropout',0.0),
                          optimizer=params.get('optimizer','adam'),
                          lr=params.get('lr',1e-3))
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0)]
        if params.get('reduce_lr', False):
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3,patience//2), verbose=0))
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs_search, batch_size=batch_size,
                            callbacks=callbacks, verbose=verbose)
        y_val_pred = model.predict(X_val).flatten()
        val_rmse = rmse(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        results.append({'iter': i+1, 'params': params, 'val_rmse': val_rmse, 'val_r2': val_r2, 'history': history})
        if val_rmse < best_score:
            best_score = val_rmse
            best = {'params': params, 'val_rmse': val_rmse, 'val_r2': val_r2}
        K.clear_session()
        print(f"[Search {i+1}/{n_iter}] val_rmse={val_rmse:.4f} val_r2={val_r2:.4f} params={params}")
    return best, results, (X_train_full, X_test, y_train_full, y_test)


# Final train + save plots + model summary + metrics

def train_final_and_save_plots(X_train_full, X_test, y_train_full, y_test,
                               best_params, epochs_final=200, batch_size=32, patience=12,
                               save_prefix='model'):
    model = build_mlp(input_dim=X_train_full.shape[1],
                      hidden_units=best_params.get('hidden_units', (64,32)),
                      activation=best_params.get('activation','relu'),
                      dropout=best_params.get('dropout',0.0),
                      optimizer=best_params.get('optimizer','adam'),
                      lr=best_params.get('lr',1e-3))

    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3,patience//3), verbose=0)]

    X_tr, X_val, y_tr, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=SEED)

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs_final, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Evaluation on test

    y_test_pred = model.predict(X_test).flatten()
    test_rmse = rmse(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    pct_rmse = percent_rmse(y_test, y_test_pred)

    # Saving training history plot (RMSE style)
    # Note: trained on MSE loss; converted to RMSE for plotting (sqrt)

    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    train_rmse = np.sqrt(train_loss)
    val_rmse_hist = np.sqrt(val_loss)

    plt.figure(figsize=(8,5))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(val_rmse_hist, label='Val RMSE')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.title(f'{save_prefix} - Train vs Val RMSE')
    plt.legend(); plt.grid(True)
    hist_jpg = f"{save_prefix}_history.jpg"
    plt.savefig(hist_jpg, dpi=200, bbox_inches='tight')
    plt.close()

    # Predicted vs actual

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_test_pred, s=12, alpha=0.6)
    mn, mx = float(min(y_test.min(), y_test_pred.min())), float(max(y_test.max(), y_test_pred.max()))
    plt.plot([mn,mx],[mn,mx], 'r--')
    plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(f'{save_prefix} - Predicted vs Actual (Test)')
    pred_jpg = f"{save_prefix}_pred_vs_actual.jpg"
    plt.savefig(pred_jpg, dpi=200, bbox_inches='tight')
    plt.close()

    # Saving model in .keras format

    model_file = f"{save_prefix}_best_model.keras"
    model.save(model_file)

    # Saving model summary to text

    summary_file = f"{save_prefix}_model_summary.txt"
    save_model_summary(model, summary_file)

    # Saving metrics to CSV

    metrics_df = pd.DataFrame([{
        'prefix': save_prefix,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'percent_rmse': pct_rmse,
        'model_file': model_file,
        'summary_file': summary_file,
        'history_file': hist_jpg,
        'pred_file': pred_jpg
    }])
    metrics_csv = f"{save_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Quick overfitting check

    overfitting_flag, tr_mean, val_mean, msg = check_overfitting(history)
    print(msg)

    print(f"Final test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, %RMSE: {pct_rmse:.2f}%")
    print(f"Saved: {hist_jpg}, {pred_jpg}, {model_file}, {summary_file}, {metrics_csv}")

    return {'model': model, 'test_rmse': test_rmse, 'test_r2': test_r2, 'percent_rmse': pct_rmse,
            'history': history, 'history_jpg': hist_jpg, 'pred_jpg': pred_jpg,
            'model_file': model_file, 'summary_file': summary_file, 'metrics_csv': metrics_csv}


# Main execution

if __name__ == "__main__":
    poro_file = 'Porosity_Preprocessed_Data.xlsx'
    perm_file = 'Permeability_Preprocessed_Data.xlsx'

    poro_features = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO']
    perm_features = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO','RD','RM','RT']

    if not os.path.exists(poro_file):
        raise FileNotFoundError(f"Place your porosity dataset as '{poro_file}' or change path")
    df_poro = pd.read_excel(poro_file)
    X_poro = df_poro[poro_features].values
    y_poro = df_poro['PHIF'].values.astype(float)

    param_dist_poro = {
        'hidden_units': [(64,32), (128,64), (32,16)],
        'dropout': [0.0, 0.05, 0.1],
        'optimizer': ['adam', 'rmsprop'],
        'lr': [1e-3, 5e-4, 1e-4],
        'activation': ['relu'],
        'reduce_lr': [True, False]
    }

    best_poro, search_results_poro, splits_poro = random_search_hyperparams(X_poro, y_poro,
                                                                            param_dist=param_dist_poro,
                                                                            n_iter=15,
                                                                            epochs_search=40,
                                                                            batch_size=32,
                                                                            patience=6,
                                                                            verbose=0)
    print("Best search result (porosity):", best_poro)

    X_train_full_poro, X_test_poro, y_train_full_poro, y_test_poro = splits_poro
    final_poro = train_final_and_save_plots(X_train_full_poro, X_test_poro, y_train_full_poro, y_test_poro,
                                            best_poro['params'], epochs_final=150, batch_size=32, patience=12,
                                            save_prefix='porosity_baseline')

    # Permeability

    if os.path.exists(perm_file):
        df_perm = pd.read_excel(perm_file)
        X_perm = df_perm[perm_features].values
        y_perm = df_perm['KLOGH'].values.astype(float) 
        y_perm_log = np.log1p(y_perm)   

        param_dist_perm = {
            'hidden_units': [(128,64), (64,32), (256,128)],
            'dropout': [0.0, 0.05, 0.1],
            'optimizer': ['adam', 'rmsprop'],
            'lr': [1e-3, 5e-4, 1e-4],
            'activation': ['relu'],
            'reduce_lr': [True, False]
        }

        best_perm, search_results_perm, splits_perm = random_search_hyperparams(X_perm, y_perm_log,
                                                                                param_dist=param_dist_perm,
                                                                                n_iter=15,
                                                                                epochs_search=40,
                                                                                batch_size=32,
                                                                                patience=6,
                                                                                verbose=0)
        print("Best search result (permeability, log-target):", best_perm)

        X_train_full_perm, X_test_perm, y_train_full_perm_log, y_test_perm_log = splits_perm
        final_perm_log = train_final_and_save_plots(X_train_full_perm, X_test_perm, y_train_full_perm_log, y_test_perm_log,
                                                    best_perm['params'], epochs_final=200, batch_size=32, patience=12,
                                                    save_prefix='permeability_baseline')

        # inverting predictions to original units for reporting 

        model_perm = final_perm_log['model']
        y_test_pred_log = model_perm.predict(X_test_perm).flatten()
        y_test_pred_orig = np.expm1(y_test_pred_log)
        y_test_orig = np.expm1(y_test_perm_log)
        test_rmse_orig = rmse(y_test_orig, y_test_pred_orig)
        test_r2_orig = r2_score(y_test_orig, y_test_pred_orig)
        pct_rmse_orig = 100.0 * test_rmse_orig / np.mean(y_test_orig)

        # saving a small report

        pd.DataFrame([{
            'test_rmse_orig': test_rmse_orig, 'test_r2_orig': test_r2_orig, 'percent_rmse_orig': pct_rmse_orig
        }]).to_csv('permeability_original_units_metrics.csv', index=False)
        print(f"Permeability test RMSE (original units): {test_rmse_orig:.4f}, R2: {test_r2_orig:.4f}, %RMSE: {pct_rmse_orig:.2f}%")

    print("Auto hyperparameter search + final training complete. JPG plots, models and summaries saved.")
