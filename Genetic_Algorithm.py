# GA_Feature_Selection_and_Comparison.py

import os, io, random, math, sys
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

# CONFIG / SEED

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
set_random_seed(SEED)

# Files and features

PORO_FILE = 'Porosity_Preprocessed_Data.xlsx'
PERM_FILE = 'Permeability_Preprocessed_Data.xlsx'

PORO_FEATURES = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO']
PERM_FEATURES = ['RHOB','NPHI','DT','GR','PEF','CALI','DRHO','RD','RM','RT']

# Targets in files

PORO_TARGET = 'PHIF'
PERM_TARGET = 'KLOGH'   # note: script will log-transform permeability during modeling

# GA hyperparameters

POP_SIZE = 20
N_GENERATIONS = 20
CXPB = 0.8         
MUTPB = 0.15       
TOURN_SIZE = 3

# Fitness training hyperparameters
FIT_EPOCHS = 30
FIT_BATCH = 32
FIT_PATIENCE = 6

# Final training hyperparameters

FINAL_EPOCHS = 150
FINAL_BATCH = 32
FINAL_PATIENCE = 12

# Other settings

VERBOSE = 0
SAVE_DIR = './ga_results'
os.makedirs(SAVE_DIR, exist_ok=True)

# Utilities

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def percent_rmse(y_true, y_pred):
    mean_y = np.mean(y_true)
    return 100.0 * rmse(y_true, y_pred) / mean_y if mean_y != 0 else np.nan

def save_model_summary(model, filepath):
    with io.StringIO() as buf:
        model.summary(print_fn=lambda s: buf.write(s + '\n'))
        summary_str = buf.getvalue()
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary_str)
    except Exception:
        with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
            f.write(summary_str)

def build_mlp(input_dim, hidden_units=(128,64), activation='relu', dropout=0.0, optimizer='adam', lr=1e-3):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(hidden_units[0], activation=activation))
    if dropout and dropout>0: model.add(Dropout(dropout))
    for units in hidden_units[1:]:
        model.add(Dense(units, activation=activation))
        if dropout and dropout>0: model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))
    opt = Adam(learning_rate=lr) if optimizer=='adam' else RMSprop(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse')
    return model


# GA: chromosome helpers

def random_chromosome(n_bits):

    # ensuring at least one bit on
    
    while True:
        chrom = [random.choice([0,1]) for _ in range(n_bits)]
        if sum(chrom) > 0:
            return chrom

def crossover(parent1, parent2):
    n = len(parent1)
    if random.random() < CXPB:
        cxpoint = random.randint(1, n-1)
        child1 = parent1[:cxpoint] + parent2[cxpoint:]
        child2 = parent2[:cxpoint] + parent1[cxpoint:]
        return child1, child2
    else:
        return parent1[:], parent2[:]

def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < MUTPB:
            chrom[i] = 1 - chrom[i]
    if sum(chrom) == 0:

        # keeping at least one feature

        chrom[random.randint(0,len(chrom)-1)] = 1
    return chrom

def tournament_select(pop, fitnesses, k=TOURN_SIZE):
    selected = random.sample(range(len(pop)), k)
    best = min(selected, key=lambda idx: fitnesses[idx])
    return pop[best][:]  # return copy

# Fitness evaluation

def evaluate_chromosome(X, y, chrom, epochs=FIT_EPOCHS, batch_size=FIT_BATCH, patience=FIT_PATIENCE):

    # chrom is list of 0/1: ch
    mask = np.array(chrom, dtype=bool)
    X_sel = X[:, mask]

    # spliting into train/val

    X_tr, X_val, y_tr, y_val = train_test_split(X_sel, y, test_size=0.15, random_state=SEED)
    K.clear_session()
    model = build_mlp(input_dim=X_sel.shape[1], hidden_units=(64,32), dropout=0.05, optimizer='adam', lr=1e-3)
    callbacks = [EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3,patience//2), verbose=0)]
    model.fit(X_tr, y_tr, validation_data=(X_val,y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=VERBOSE)
    y_val_pred = model.predict(X_val).flatten()
    score = rmse(y_val, y_val_pred)
    K.clear_session()
    return score


# GA main loop

def run_ga_feature_selection(X, y, feature_names, pop_size=POP_SIZE, generations=N_GENERATIONS):
    n_bits = len(feature_names)

    # initial population
    pop = [random_chromosome(n_bits) for _ in range(pop_size)]
    fitnesses = [None]*pop_size

    # evaluating initial population

    for i,chrom in enumerate(pop):
        fitnesses[i] = evaluate_chromosome(X, y, chrom)
        print(f"Init individual {i+1}/{pop_size} fitness={fitnesses[i]:.4f}")

    best_per_gen = []
    best_chrom = None
    best_score = float('inf')

    for gen in range(generations):
        new_pop = []
        new_fitnesses = []

        # keeping best 1

        best_idx = int(np.argmin(fitnesses))
        elite = pop[best_idx][:]
        elite_score = fitnesses[best_idx]
        if elite_score < best_score:
            best_score = elite_score
            best_chrom = elite[:]
        best_per_gen.append(best_score)
        print(f"GA gen {gen+1}/{generations} best_val_rmse_so_far={best_score:.4f}")

        # produce new population

        while len(new_pop) < pop_size:

            # selection

            p1 = tournament_select(pop, fitnesses)
            p2 = tournament_select(pop, fitnesses)

            # crossover

            c1, c2 = crossover(p1, p2)

            # mutation

            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        # evaluating new population

        for i,chrom in enumerate(new_pop):
            fitness = evaluate_chromosome(X, y, chrom)
            new_fitnesses.append(fitness)
            print(f" Gen{gen+1} eval {i+1}/{len(new_pop)} fitness={fitness:.4f}")

        # replacing population but carrying over elite - ensuring elite is preserved, finding worst to replace if elite not present
       
        new_pop[0] = elite
        new_fitnesses[0] = elite_score
        pop = new_pop
        fitnesses = new_fitnesses

    # final best_chrom and best_score

    return {'best_chrom': best_chrom, 'best_score': best_score, 'best_per_gen': best_per_gen, 'feature_names': feature_names}


# Training final models and saving outputs

def train_and_save(X_train_full, X_test, y_train_full, y_test, feature_names, selected_mask, prefix, final_epochs=FINAL_EPOCHS):

    # selected_mask: list of 0/1 for which features to use

    sel_idx = np.where(np.array(selected_mask,dtype=bool))[0]
    sel_features = [feature_names[i] for i in sel_idx]
    eliminated = [feature_names[i] for i in range(len(feature_names)) if i not in sel_idx]

    # Training baseline GA and GA-ANN

    results = {}

    for mode in ['baseline','ga_selected']:
        if mode == 'baseline':
            X_full = X_train_full.copy()
            feat_names = feature_names
            prefix_mode = f"{prefix}_baseline"
        else:
            X_full = X_train_full[:, sel_idx]
            feat_names = sel_features
            prefix_mode = f"{prefix}_ga"

        # split train_full into train/val for final training

        X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_train_full, test_size=0.10, random_state=SEED)
        K.clear_session()
        model = build_mlp(input_dim=X_tr.shape[1], hidden_units=(128,64), dropout=0.05, optimizer='adam', lr=1e-3)
        callbacks = [EarlyStopping(monitor='val_loss', patience=FINAL_PATIENCE, restore_best_weights=True, verbose=0),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=max(3,FINAL_PATIENCE//3), verbose=0)]
        history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=final_epochs, batch_size=FINAL_BATCH, callbacks=callbacks, verbose=1)

        # predictions on test

        if mode == 'baseline':
            X_test_mode = X_test.copy()
        else:
            X_test_mode = X_test[:, sel_idx]
        y_pred = model.predict(X_test_mode).flatten()

        # computING metrics

        test_rmse = rmse(y_test, y_pred)
        test_r2 = r2_score(y_test, y_pred)
        pct_rmse = percent_rmse(y_test, y_pred)

        # saving history plot (RMSE)

        train_loss = np.array(history.history['loss'])
        val_loss = np.array(history.history['val_loss'])
        train_rmse = np.sqrt(train_loss)
        val_rmse_hist = np.sqrt(val_loss)
        plt.figure(figsize=(8,5))
        plt.plot(train_rmse, label='Train RMSE')
        plt.plot(val_rmse_hist, label='Val RMSE')
        plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.title(f'{prefix_mode} Train vs Val RMSE')
        plt.legend(); plt.grid(True)
        hist_jpg = os.path.join(SAVE_DIR, f"{prefix_mode}_history.jpg")
        plt.savefig(hist_jpg, dpi=200, bbox_inches='tight'); plt.close()

        # saving predicted vs actual

        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, s=12, alpha=0.6)
        mn, mx = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        plt.plot([mn,mx],[mn,mx],'r--')
        plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title(f'{prefix_mode} Pred vs Actual')
        pred_jpg = os.path.join(SAVE_DIR, f"{prefix_mode}_pred_vs_actual.jpg")
        plt.savefig(pred_jpg, dpi=200, bbox_inches='tight'); plt.close()

        # residual distribution

        residuals = y_test - y_pred
        plt.figure(figsize=(6,4))
        plt.hist(residuals, bins=40, alpha=0.7)
        plt.xlabel('Residual (Actual - Predicted)'); plt.ylabel('Count'); plt.title(f'{prefix_mode} Residuals')
        resid_jpg = os.path.join(SAVE_DIR, f"{prefix_mode}_residuals.jpg")
        plt.savefig(resid_jpg, dpi=200, bbox_inches='tight'); plt.close()

        # saving model summary and model file

        model_file = os.path.join(SAVE_DIR, f"{prefix_mode}_model.keras")
        model.save(model_file)
        summary_file = os.path.join(SAVE_DIR, f"{prefix_mode}_summary.txt")
        save_model_summary(model, summary_file)

        results[mode] = {'model': model, 'history': history, 'test_rmse': test_rmse, 'test_r2': test_r2, 'pct_rmse': pct_rmse,
                         'hist_jpg': hist_jpg, 'pred_jpg': pred_jpg, 'resid_jpg': resid_jpg, 'model_file': model_file,
                         'summary_file': summary_file, 'feat_names': feat_names}
        K.clear_session()

    # returning results and feature lists

    return results, sel_features, eliminated

# GA to run the whole flow for one target (porosity or permeability)

def run_full_flow(data_file, feature_list, target_col, target_transform=None, prefix='porosity'):

    # load dataset

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    df = pd.read_excel(data_file)
    X = df[feature_list].values
    y = df[target_col].values.astype(float)
    if target_transform == 'log1p':
        y = np.log1p(y)

    # creating a consistent split for final eval

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)

    # Running GA

    print(f"Running GA for {prefix} feature selection...")
    ga_out = run_ga_feature_selection(X_train_full, y_train_full, feature_list, pop_size=POP_SIZE, generations=N_GENERATIONS)

    # Saving GA convergence plot

    best_per_gen = ga_out['best_per_gen']
    plt.figure(figsize=(8,5))
    plt.plot(best_per_gen, marker='o')
    plt.xlabel('Generation'); plt.ylabel('Best validation RMSE'); plt.title(f'GA convergence ({prefix})')
    ga_conv_jpg = os.path.join(SAVE_DIR, f"{prefix}_ga_convergence.jpg")
    plt.grid(True); plt.savefig(ga_conv_jpg, dpi=200, bbox_inches='tight'); plt.close()

    # best mask and features

    best_mask = ga_out['best_chrom']
    sel_features = [f for f,bit in zip(feature_list,best_mask) if bit==1]
    elim_features = [f for f,bit in zip(feature_list,best_mask) if bit==0]

    # exported selected/eliminated features to excel

    excel_rows = []
    for f in feature_list:
        excel_rows.append({'feature': f, 'selected': 1 if f in sel_features else 0})
    df_feats = pd.DataFrame(excel_rows)
    feats_xlsx = os.path.join(SAVE_DIR, f"{prefix}_selected_features.xlsx")
    df_feats.to_excel(feats_xlsx, index=False, engine='openpyxl')

    # Train baseline and GA-ANN final models (with saved test split)

    results, sel_feats_list, elim = train_and_save(X_train_full, X_test, y_train_full, y_test, feature_list, best_mask, prefix)

    # computing predicted vs actual in original units if log1p was used, invert when saving CSVs and Saving comparison metrics to excel
   
    comp_rows = []

    # baseline

    b = results['baseline']
    comp_rows.append({'prefix': prefix, 'model_type': 'baseline', 'test_rmse': b['test_rmse'], 'test_r2': b['test_r2'], 'percent_rmse': b['pct_rmse']})

    # GA

    g = results['ga_selected']
    comp_rows.append({'prefix': prefix, 'model_type': 'ga_selected', 'test_rmse': g['test_rmse'], 'test_r2': g['test_r2'], 'percent_rmse': g['pct_rmse']})
    df_comp = pd.DataFrame(comp_rows)
    comp_xlsx = os.path.join(SAVE_DIR, f"{prefix}_comparison_metrics.xlsx")
    df_comp.to_excel(comp_xlsx, index=False, engine='openpyxl')

    return {'ga_out': ga_out, 'ga_conv_jpg': ga_conv_jpg, 'feats_xlsx': feats_xlsx, 'results': results, 'comp_xlsx': comp_xlsx}


# Run for porosity and permeability

if __name__ == "__main__":

    # POROSITY

    porosity_outputs = run_full_flow(PORO_FILE, PORO_FEATURES, PORO_TARGET, target_transform=None, prefix='porosity')
    print("Porosity GA+ANN done. Files saved to", SAVE_DIR)

    # PERMEABILITY (use log-transform for skew)

    perm_outputs = run_full_flow(PERM_FILE, PERM_FEATURES, PERM_TARGET, target_transform='log1p', prefix='permeability')
    print("Permeability GA+ANN done. Files saved to", SAVE_DIR)

    # Combine porosity & permeability comparison tables into one Excel for thesis
    
    por_comp = pd.read_excel(porosity_outputs['comp_xlsx'])
    perm_comp = pd.read_excel(perm_outputs['comp_xlsx'])
    combined = pd.concat([por_comp, perm_comp], ignore_index=True)
    combined.to_excel(os.path.join(SAVE_DIR,'combined_models_comparison.xlsx'), index=False, engine='openpyxl')
    print("Combined comparison saved.")
