"""
eSports Sensors Analysis 3.0 – VERSIÓN DEFINITIVA
=================================================
• Imputación KNN para recuperar registros perdidos  
• Validación por jugador dinámica (GroupKFold / LOGO)  
• Pipeline ElasticNetCV con QuantileTransformer  
• Optimización bayesiana (Optuna) para XGBoost  
• GridSearchCV de XGBoost  
• SHAP Beeswarm para interpretabilidad  
• Permutation feature importance  
• Stacking final de modelos  
• Clasificación ordinal binaria (bajo vs alto rendimiento)  
• Clasificación subestima vs sobreestima (RandomForest)  
• Análisis de errores por rol funcional  
• Clustering comparativo (HDBSCAN, DBSCAN, GMM, KMeans)  
• PCA previo a t-SNE para visualización  
• R² ajustado y ANOVA entre clusters  
Autor: Francisco Enríquez
"""

from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def save_fig(name: str):
    """Guarda la figura más reciente en analysis_outputs."""
    fig_path = Path(BASE_PATH) / "analysis_outputs" / f"{name}.png"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura guardada: {fig_path}")

# ───────────── 0. FILTRADO DE WARNINGS ─────────────
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignora UserWarning, ConvergenceWarning y mensajes de n_quantiles
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*n_quantiles.*")

import logging
# Silencia logs de xgboost y sklearn
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

# ───────────── 1. IMPORTS ─────────────
from sklearn.ensemble import RandomForestClassifier
import os, json, datetime, platform, getpass
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# scikit-learn modelado
from sklearn.model_selection import (
    GroupKFold, cross_validate, train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNetCV, RidgeCV, LogisticRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import (
    make_scorer, mean_absolute_error, r2_score, mean_squared_error,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, classification_report
)
from sklearn.inspection import permutation_importance

# Clustering y reducción de dimensión
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Interpretabilidad
import shap

# AutoML bayesiano
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
import joblib

# Clustering HDBSCAN
import hdbscan
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# ───────────── 2. RUTAS y CONSTANTES ─────────────
BASE_PATH     = r"C:/Users/franc/Desktop/Colima/eSports_Sensors_Dataset-master"
METADATA_FILE = "players_info.csv"

# ───────────── 3. MÉTRICAS PERSONALIZADAS ─────────────
def kendall_tau(y_true, y_pred):
    """Kendall Tau para métrica ordinal."""
    return stats.kendalltau(y_true, y_pred)[0]

SCORERS = {
    "R2":      "r2",
    "MAE":     make_scorer(mean_absolute_error, greater_is_better=False),
    "Kendall": make_scorer(kendall_tau, greater_is_better=True),
}

# ───────────── 4. HELPERS ─────────────
def load_json(path: str) -> dict:
    """Carga JSON o devuelve dict vacío."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def print_header(msg: str):
    """Imprime título de sección."""
    print(f"\n{'='*10} {msg.upper()} {'='*10}")

# ───────────── 5. INGESTA CRUDA DE PARTIDAS ─────────────
players_info = pd.read_csv(os.path.join(BASE_PATH, METADATA_FILE))
matches_dir  = os.path.join(BASE_PATH, "matches")

rows: List[dict] = []
report_keys = set()

for match in os.listdir(matches_dir):
    mdir = os.path.join(matches_dir, match)
    if not os.path.isdir(mdir):
        continue

    for pdir_name in os.listdir(mdir):
        pdir = os.path.join(mdir, pdir_name)
        if not (os.path.isdir(pdir) and pdir_name.startswith("player_")):
            continue

        row = {"match": match, "player": pdir_name}

        for fname in os.listdir(pdir):
            fpath = os.path.join(pdir, fname)
            if fname.endswith(".csv"):
                df_tmp = pd.read_csv(fpath)
                base   = fname.replace(".csv", "")
                row[f"{base}_mean"] = df_tmp.mean(numeric_only=True).mean()
                row[f"{base}_std"]  = df_tmp.std(numeric_only=True).mean()
            elif fname == "player_report.json":
                jdata = load_json(fpath)
                for k, v in jdata.items():
                    col = f"report_{k}"
                    row[col] = v
                    report_keys.add(col)

        rows.append(row)

df_raw = pd.DataFrame(rows)
df_raw["player_id"] = (
    df_raw["player"]
      .str.extract(r"(\d+)$")[0]
      .astype(float)
      .astype("Int64")
)
df_raw = df_raw.merge(players_info, how="left", on="player_id")

print_header("Claves JSON encontradas")
print(sorted(report_keys))

# ───────────── 6. IMPUTACIÓN KNN (recuperar filas) ─────────────
sensor_means = [c for c in df_raw.columns if c.endswith("_mean")]
imputer      = KNNImputer(n_neighbors=3)
df_raw[sensor_means] = imputer.fit_transform(df_raw[sensor_means])

# ───────────── 7. FEATURE ENGINEERING DERIVADO ─────────────
def safe_ratio(num: pd.Series, denom: pd.Series):
    return num / (denom.replace(0, np.nan) + 1e-6)

df_raw["emg_gsr_ratio"]      = safe_ratio(df_raw["emg_mean"],       df_raw["gsr_mean"])
df_raw["heart_emg_ratio"]    = safe_ratio(df_raw["heart_rate_mean"], df_raw["emg_mean"])
df_raw["gsr_spo2_diff"]      = df_raw["gsr_mean"] - df_raw["spo2_mean"]
df_raw["emg_keyboard_ratio"] = safe_ratio(df_raw["emg_mean"],       df_raw["keyboard_mean"])
df_raw["stress_index"]       = (
    df_raw["heart_rate_mean"] + df_raw["gsr_mean"] - df_raw["spo2_mean"]
)

sensor_feats = sensor_means + [
    "emg_gsr_ratio",
    "heart_emg_ratio",
    "gsr_spo2_diff",
    "emg_keyboard_ratio",
    "stress_index",
]

print_header("DataFrame después de imputación")
print(df_raw.shape)

# ───────────── 8. MAPEO ORDENAL (Likert) ─────────────
likert_map = {
    "not at all":   0,
    "a little bit": 1,
    "somewhat":     2,
    "very much":    3,
    "extremely":    4,
}
for col in df_raw.select_dtypes(include="object").columns:
    if df_raw[col].dropna().isin(likert_map).all():
        df_raw[col] = df_raw[col].map(likert_map)

# ───────────── 9. DEFINICIÓN DE X, y y GRUPOS ─────────────
TARGET      = "report_performance_evaluation"
feature_cols = sensor_feats
df_mod      = df_raw.dropna(subset=[TARGET]).copy()

X_base      = df_mod[feature_cols]
y_base      = df_mod[TARGET].astype(float)
groups_base = df_mod["player_id"].astype(int)

print_header("Muestra antes de SMOTER")
print(X_base.shape, "  jugadores únicos:", groups_base.nunique())

# ───────────── 10. AUGMENTACIÓN CON RUIDO GAUSSIANO ─────────────
rng      = np.random.default_rng(42)

# Reiniciar índices en X_base, y_base y groups_base
X_base      = X_base.reset_index(drop=True)
y_base      = y_base.reset_index(drop=True)
groups_base = groups_base.reset_index(drop=True)

# Guardamos índices originales para reconstruir el DF aumentado
orig_indices   = list(range(len(X_base)))
extra_indices  = []

# Empezamos el set aumentado replicando X_base, y_base
X_aug = X_base.copy()
y_aug = y_base.copy()

# Creamos +70% muestras sintéticas con ruido
n_extra = int(0.7 * len(X_base))
sigma_x = 0.01 * X_base.std(axis=0)
sigma_y = 0.05

for _ in range(n_extra):
    idx = int(rng.integers(len(X_base)))
    extra_indices.append(idx)
    x0 = X_base.iloc[idx].to_numpy()
    y0 = y_base.iloc[idx]

    x_noise = x0 + rng.normal(0, sigma_x, size=x0.shape)
    y_noise = y0 + rng.normal(0, sigma_y)

    X_aug.loc[len(X_aug)] = x_noise
    y_aug.loc[len(y_aug)] = y_noise

# Reconstruir vector de grupos para los datos aumentados
groups_aug = pd.concat([
    groups_base,
    pd.Series([groups_base.iloc[i] for i in extra_indices],
              index=range(len(X_base), len(X_aug)))
]).reset_index(drop=True)

# Índice maestro para mapear de vuelta al DataFrame original
sample_indices_ = orig_indices + extra_indices

# ───────────── 11. HELPERS DE VALIDACIÓN ─────────────
def make_cv(g_series, max_splits: int = 5):
    """GroupKFold con splits = min(n_groups, max_splits)."""
    n = g_series.nunique()
    if n < 2:
        raise ValueError("Se requieren al menos 2 jugadores.")
    return GroupKFold(n_splits=min(max_splits, n))

CV_GLOBAL = make_cv(groups_aug)

# ───────────── 12. PIPELINE ELASTICNET + TRANSFORMED TARGET ─────────────
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import QuantileTransformer

# Construye el pipeline:
#   1) Estandarizado
#   2) Selección de k mejores features
#   3) ElasticNetCV sobre la variable objetivo transformada a distribución normal
en_pipe = Pipeline([
    ("scale",  StandardScaler()),
    ("select", SelectKBest(score_func=f_regression, k=12)),
    ("model",  TransformedTargetRegressor(
        regressor=ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9],
            alphas=np.logspace(-2, 2, 50),
            max_iter=10000,    # más iteraciones para evitar warnings de convergencia
            tol=5e-3,          # tolerancia algo más laxa
            random_state=42,
        ),
        transformer=QuantileTransformer(
            n_quantiles=200,
            output_distribution="normal",
        )
    )),
])

# ───────────── 12bis. MATRIZ DE CORRELACIÓN GLOBAL ─────────────
print_header("Matriz de correlación – todas las features vs evaluación")
# Construimos df_corr con todas las features y el target
df_corr = df_mod[sensor_feats + ["report_performance_evaluation"]].dropna()
corr_scores = df_corr.corr()["report_performance_evaluation"].sort_values(ascending=False)
print(corr_scores.to_string())
# (Opcional: sns.heatmap(df_corr.corr(), cmap="coolwarm", center=0))

# ───────────── 13. VALIDACIÓN ElasticNetCV ─────────────
cv_results_en = cross_validate(
    en_pipe,
    X_aug,
    y_aug,
    groups=groups_aug,
    cv=CV_GLOBAL,
    scoring=SCORERS,
    n_jobs=-1,
    return_train_score=False,
)

print_header("Resultados ElasticNetCV (media ± est)")
for metric in SCORERS:
    vals = cv_results_en[f"test_{metric}"]
    # El scorer de MAE devuelve negativo, lo invertimos para presentar
    display_val = -vals if metric == "MAE" else vals
    print(f"{metric:7s}: {display_val.mean():>6.3f} ± {display_val.std():.3f}")

# ───────────── 14. PIPELINE XGBOOST BASE ─────────────
xgb_pipe = Pipeline([
    ("scale",  StandardScaler()),
    ("select", SelectKBest(score_func=f_regression, k=10)),
    ("model",  xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    ))
])

# ───────────── 15. VALIDACIÓN XGBOOST POR JUGADOR ─────────────
cv_results_xgb = cross_validate(
    xgb_pipe,
    X_aug,
    y_aug,
    groups=groups_aug,
    cv=CV_GLOBAL,
    scoring=SCORERS,
    n_jobs=-1,
    return_train_score=False,
)

print_header("Resultados XGBoost base (media ± est)")
for metric in SCORERS:
    vals = cv_results_xgb[f"test_{metric}"]
    # Invertimos signo para MAE
    display_val = -vals if metric == "MAE" else vals
    print(f"{metric:7s}: {display_val.mean():>6.3f} ± {display_val.std():.3f}")

# ───────────── 15bis. MODELADO BÁSICO (RidgeCV y RandomForest) ─────────────
print_header("Modelos base – RidgeCV y RandomForest (train/test split)")
X_train0, X_test0, y_train0, y_test0 = train_test_split(
    X_aug, y_aug, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestRegressor
ridge = RidgeCV(alphas=[0.1,1.0,10.0]).fit(X_train0, y_train0)
rf    = RandomForestRegressor(random_state=42).fit(X_train0, y_train0)
for name, model in [("RidgeCV", ridge), ("RandomForest", rf)]:
    ypred = model.predict(X_test0)
    r2 = r2_score(y_test0, ypred)
    mse = mean_squared_error(y_test0, ypred)
    print(f"{name} - R²: {r2:.3f}, MSE: {mse:.3f}")
    top5 = (pd.DataFrame({"true":y_test0, "pred":ypred})
              .assign(err=lambda df: abs(df.true-df.pred))
              .nlargest(5, "err"))
    print(" Top-5 errores:\n", top5)

# ───────────── 15bis.1. XGBOOST SIMPLE (TRAIN/TEST SPLIT) ─────────────
print_header("Modelo base – XGBoost Regressor (train/test split)")
xgb_simple = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
).fit(X_train0, y_train0)

y_pred_xgb0 = xgb_simple.predict(X_test0)
r2_xgb0     = r2_score(y_test0, y_pred_xgb0)
mse_xgb0    = mean_squared_error(y_test0, y_pred_xgb0)
print(f"XGBoost    - R²: {r2_xgb0:.3f}, MSE: {mse_xgb0:.3f}")

# Top-5 errores
top5_xgb = (
    pd.DataFrame({"true": y_test0, "pred": y_pred_xgb0})
      .assign(err=lambda df: abs(df.true - df.pred))
      .nlargest(5, "err")
)
print(" Top-5 errores XGB:\n", top5_xgb)

# ───────────── 15quater. VALIDACIÓN CRUZADA GLOBAL – XGBOOST SIMPLE ─────────────
print_header("Validación cruzada global – XGBoost simple (5-fold)")
from sklearn.model_selection import cross_val_score
# xgb_simple es el XGBRegressor que ya has ajustado en 15bis.1
cv_scores_xgb_global = cross_val_score(
    xgb_simple,
    X_aug,
    y_aug,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
print(f"R² global XGB simple (5-fold): {cv_scores_xgb_global.mean():.3f} ± {cv_scores_xgb_global.std():.3f}")

# ───────────── 15ter. R² Ajustado ejemplo (XGB Optuna) ─────────────
n, p = X_test0.shape
r2 = r2_score(y_test0, ridge.predict(X_test0))
r2_adj = 1 - (1-r2)*(n-1)/(n-p-1)
print(f"R² ajustado (RidgeCV): {r2_adj:.3f}")

# ───────────── 16. OPTUNA PARA XGBOOST ─────────────
def optuna_xgb_objective(trial):
    # Sugerencia de número de features k_best
    k_best = trial.suggest_int("k_best", 8, 15)
    # Espacio de hiperparámetros
    params = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 500, step=100),
        "max_depth":          trial.suggest_int("max_depth", 3, 6),
        "learning_rate":      trial.suggest_float("lr", 0.01, 0.1, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "objective":          "reg:squarederror",
        "random_state":       42,
        "verbosity":          0,
        "use_label_encoder":  False,
    }
    # Pipeline de prueba
    pipe_trial = Pipeline([
        ("scale",  StandardScaler()),
        ("select", SelectKBest(f_regression, k=k_best)),
        ("model",  xgb.XGBRegressor(**params))
    ])
    # Validación cruzada interna para optimizar MAE
    res = cross_validate(
        pipe_trial,
        X_aug,
        y_aug,
        groups=groups_aug,
        cv=CV_GLOBAL,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    # Optuna minimiza: devolvemos MAE positivo
    return -res["test_score"].mean()

# Lanzar la optimización
study_xgb = optuna.create_study(direction="minimize")
study_xgb.optimize(optuna_xgb_objective, n_trials=20, show_progress_bar=False)

print_header("Mejor MAE CV (XGB Optuna)")
print(f"MAE: {study_xgb.best_value:.3f}")
print("Params:", study_xgb.best_params)

# ───────────── 16bis. GRIDSEARCHCV XGBOOST ─────────────
print_header("Optimización con GridSearchCV – XGBoost")
param_grid = {
    "model__max_depth": [2,4,6],
    "model__learning_rate": [0.01, 0.1],
    "model__n_estimators": [100,200]
}
grid = GridSearchCV(
    xgb_pipe,
    param_grid,
    scoring="r2",
    cv=3,
    n_jobs=-1
)
grid.fit(X_aug, y_aug)
print(" GridSearchCV best params:", grid.best_params_)
print(" Best R²:", grid.best_score_)

# ───────────── 17. PIPELINE XGB OPTUNA FINAL ─────────────
# Extraemos k_best y parámetros óptimos
best_params_xgb = study_xgb.best_params.copy()
k_best_opt      = best_params_xgb.pop("k_best")

# Construcción y ajuste del pipeline definitivo
opt_xgb_pipe = Pipeline([
    ("scale",  StandardScaler()),
    ("select", SelectKBest(f_regression, k=k_best_opt)),
    ("model",  xgb.XGBRegressor(
        **best_params_xgb,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    ))
])
opt_xgb_pipe.fit(X_aug, y_aug)

# ───────────── 17bis. SERIALIZAR PIPELINE ∙─────────────
out_path = Path(BASE_PATH) / "analysis_outputs" / "xgb_optuna_pipeline.joblib"
out_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(opt_xgb_pipe, out_path)
print_header(f"Pipeline XGB Optuna guardado en {out_path}")

# ───────────── 17ter. VALIDACIÓN CRUZADA GLOBAL CON XGB OPTUNA ─────────────
print_header("Validación cruzada global – XGB Optuna")
# Creamos un XGBRegressor con los mejores hiperparámetros hallados
xgb_global = xgb.XGBRegressor(
    **best_params_xgb,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)
# Necesitamos escalar igual que en nuestro pipeline original
scaler_global = StandardScaler().fit(X_aug)
X_global_scaled = scaler_global.transform(X_aug)

# 5-fold CV sobre todo el conjunto aumentado
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    xgb_global,
    X_global_scaled,
    y_aug,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
print(f"R² global (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ───────────── 18. FUNCIÓN DE EVALUACIÓN DE PIPELINES ─────────────
def evaluate_pipeline(pipe, name, X, y, groups):
    res = cross_validate(pipe, X, y, groups=groups, cv=CV_GLOBAL,
                         scoring=SCORERS, n_jobs=-1)
    return {
        "Modelo": name,
        "R2":    res["test_R2"].mean(),
        "MAE":  -res["test_MAE"].mean(),     # invertimos signo
        "tau":   res["test_Kendall"].mean(),
    }

# ───────────── 19. COMPARATIVA DE MODELOS ─────────────
en_metrics  = evaluate_pipeline(en_pipe,   "ElasticNetCV", X_aug, y_aug, groups_aug)
xgb_metrics = evaluate_pipeline(xgb_pipe,  "XGB Base",    X_aug, y_aug, groups_aug)
opt_metrics = evaluate_pipeline(opt_xgb_pipe, "XGB Optuna", X_aug, y_aug, groups_aug)

# ───────────── 20. IMPRESIÓN DE LA TABLA RESUMEN ─────────────
df_compare = pd.DataFrame([en_metrics, xgb_metrics, opt_metrics])
print_header("Comparativa de pipelines (datos aumentados)")
print(df_compare.to_string(index=False, float_format="{:.3f}".format))

comparison_csv = Path(BASE_PATH) / "analysis_outputs" / "pipeline_comparison_augmented.csv"
comparison_csv.parent.mkdir(exist_ok=True, parents=True)
df_compare.to_csv(comparison_csv, index=False)
print(f"Tabla comparativa guardada en: {comparison_csv}")

# ───────────── 21. SHAP BEESWARM (XGB Optuna) ─────────────
print_header("SHAP Beeswarm – XGB Optuna Top Features")
shap.initjs()

# Extraer embedding (scale + select) para SHAP
X_opt_embed = opt_xgb_pipe[:-1].transform(X_aug)
model_opt   = opt_xgb_pipe.named_steps["model"]
explainer   = shap.TreeExplainer(model_opt)
shap_vals   = explainer.shap_values(X_opt_embed)

shap.summary_plot(
    shap_vals,
    features=X_opt_embed,
    feature_names=opt_xgb_pipe.named_steps["select"].get_feature_names_out(),
    max_display=10
)

# ───────────── 22. IMPORTANCIA POR PERMUTACIÓN ─────────────
print_header("Permutation Feature Importance (XGB Optuna sobre features seleccionados)")

# 1) Extraemos el embedding (scale + select) de X_aug
X_sel = opt_xgb_pipe[:-1].transform(X_aug)

# 2) Obtenemos el modelo puro
model_opt = opt_xgb_pipe.named_steps["model"]

# 3) Calculamos importancias por permutación sobre el embedding reducido
perm = permutation_importance(
    model_opt,
    X_sel,
    y_aug,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# 4) Generamos el DataFrame emparejando nombres e importancias
feat_names = opt_xgb_pipe.named_steps["select"].get_feature_names_out()
imp_df = (
    pd.DataFrame({
        "feature":    feat_names,
        "importance": perm.importances_mean
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

# 5) Mostramos las top 10
print(imp_df.head(10).to_string(index=False, float_format="{:.4f}".format))

# ───────────── 23. STACKING FINAL DE MODELOS ─────────────
print_header("Stacking de Modelos")
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

stack = StackingRegressor(
    estimators=[
        ("elastic", en_pipe),
        ("xgb",     xgb_pipe),
        ("opt_xgb", opt_xgb_pipe)
    ],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    n_jobs=-1
)
stack.fit(X_aug, y_aug)

stack_metrics = evaluate_pipeline(stack, "Stacking", X_aug, y_aug, groups_aug)
print(pd.DataFrame([stack_metrics]).to_string(index=False, float_format="{:.3f}".format))

# ───────────── 24. PERMUTATION FEATURE IMPORTANCE ─────────────
print_header("Permutation Feature Importance (Stacking)")
from sklearn.inspection import permutation_importance

# Calculamos la importancia por permutación sobre el set aumentado
perm_res = permutation_importance(
    stack, X_aug, y_aug,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Construimos DataFrame ordenado de importancias
feat_names = X_aug.columns if hasattr(X_aug, "columns") else feature_cols
perm_df = (
    pd.DataFrame({
        "feature": feat_names,
        "importance_mean": perm_res.importances_mean,
        "importance_std":  perm_res.importances_std
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

print(perm_df.head(10).to_string(index=False, float_format="{:.4f}".format))

# ───────────── 25. CLASIFICACIÓN ORDINAL BINARIA ─────────────
print_header("Clasificación Ordinal Binaria")

# Umbral: bajo (<3) vs alto (>=3)
y_bin = (y_aug >= 3).astype(int)

# División en train/test estratificando por jugador para mantener distribución
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_aug, y_bin,
    stratify=groups_aug,
    test_size=0.2,
    random_state=42
)

# Pipeline: escalado + LogisticRegression con más iteraciones
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf_ord_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42))
])

# Entrenamos
clf_ord_pipe.fit(X_train_bin, y_train_bin)

# Predicción
y_pred_bin = clf_ord_pipe.predict(X_test_bin)

# Informe de clasificación usando '>=3' en lugar de '≥3'
print(classification_report(
    y_test_bin,
    y_pred_bin,
    target_names=["Bajo (<3)", "Alto (>=3)"]
))

# Matriz de confusión
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)
disp_bin = ConfusionMatrixDisplay(
    confusion_matrix=cm_bin,
    display_labels=["Bajo (<3)", "Alto (>=3)"]
)
disp_bin.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión – Clasificación Ordinal Binaria")
plt.show()

def save_fig(name: str):
    """Guarda la figura más reciente en analysis_outputs."""
    fig_path = Path(BASE_PATH) / "analysis_outputs" / f"{name}.png"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura guardada: {fig_path}")

# ───────────── 25bis. CURVAS ROC Y PRECISION-RECALL – XGBClassifier ─────────────
print_header("Curvas ROC y Precision-Recall – XGBClassifier")

clf_xgb = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
).fit(X_train_bin, y_train_bin)

probs = clf_xgb.predict_proba(X_test_bin)[:, 1]
fpr, tpr, _         = roc_curve(y_test_bin, probs)
precision, recall, _ = precision_recall_curve(y_test_bin, probs)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
plt.plot([0,1], [0,1], "--", alpha=0.5)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()
save_fig("roc_curve_xgbclassifier")

plt.subplot(1,2,2)
plt.plot(recall, precision)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
save_fig("pr_curve_xgbclassifier")

plt.tight_layout()
plt.show()

# ───────────── 25bis.b. MATRIZ DE CONFUSIÓN – XGBClassifier ─────────────
print_header("Matriz de Confusión – XGBClassifier")
# predecimos clases
y_pred_xgb = clf_xgb.predict(X_test_bin)
# mostramos y guardamos
disp_xgb = ConfusionMatrixDisplay.from_predictions(
    y_test_bin,
    y_pred_xgb,
    display_labels=["Bajo (<3)", "Alto (>=3)"],
    cmap=plt.cm.Blues
)
plt.title("Confusion Matrix – XGBClassifier")
save_fig("confusion_matrix_xgbclassifier")
plt.show()

# ───────────── 25ter. SUBESTIMA vs SOBREESTIMA (RFC) ─────────────
print_header("Clasificación Subestima (true>pred) vs Sobreestima/Correcta")

y_class = (y_test_bin > y_pred_bin).astype(int)
rfc     = RandomForestClassifier(random_state=42).fit(X_test_bin, y_class)
y_pred_cl = rfc.predict(X_test_bin)

print(classification_report(y_class, y_pred_cl))
_ = ConfusionMatrixDisplay.from_predictions(
    y_class, y_pred_cl, cmap=plt.cm.Blues
)
plt.title("RFC Subestima vs Sobreestima")
plt.show()

# ───────────── 26. ANÁLISIS DE ERRORES POR ROL ─────────────
print_header("Análisis de Errores por Rol")

# Reconstrucción del DataFrame aumentado con las filas originales y sintéticas
df_mod_reset = df_mod.reset_index(drop=True)
df_aug = df_mod_reset.iloc[sample_indices_].copy()

# Predicciones completas del modelo apilado (stack)
y_pred_stack_full = stack.predict(X_aug)

# Añadimos las predicciones y el error absoluto
df_aug["pred_stack"] = y_pred_stack_full
df_aug["error_abs"]  = np.abs(y_aug.values - y_pred_stack_full)

# Agrupamos por rol y calculamos estadísticas de error
role_errors = (
    df_aug
    .groupby("report_role")["error_abs"]
    .agg(count="count", mean="mean", std="std")
)
print(role_errors.to_string(float_format="{:.4f}".format))

# ───────────── 26bis. MODELADO POR ROL ─────────────
print_header("Comparativa de modelos por rol (XGB Optuna)")

# 1) Reconstruimos df_aug con índices 0..N-1 exactamente paralelos a X_aug / y_aug
df_mod_reset = df_mod.reset_index(drop=True)
df_aug = df_mod_reset.iloc[sample_indices_].reset_index(drop=True).copy()

# 2) Para cada rol, filtramos por posición y hacemos un train/test split
for role in df_aug["report_role"].dropna().unique():
    mask = (df_aug["report_role"] == role).to_numpy()  # máscara posicional
    Xr = X_aug[mask]
    yr = y_aug[mask]
    if len(Xr) < 10:
        continue  # evitamos roles con pocas muestras

    # división train/test
    Xtr, Xte, ytr, yte = train_test_split(
        Xr, yr, test_size=0.2, random_state=42
    )

    # entrenamos un XGBRegressor con los mejores hiperparámetros
    mdl = xgb.XGBRegressor(
        **best_params_xgb,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    ).fit(Xtr, ytr)

    # evaluamos
    ypr = mdl.predict(Xte)
    print(f"Rol={role}: R²={r2_score(yte, ypr):.3f}, MSE={mean_squared_error(yte, ypr):.3f}")

# ───────────── 26ter. CORRELACIÓN ERROR ABSOLUTO vs FISIOLOGÍA ─────────────
print_header("Correlaciones error_abs vs perfiles fisiológicos")

# 1) Reconstruir df_aug alineado con X_aug / y_aug
df_mod_reset = df_mod.reset_index(drop=True)
df_aug = df_mod_reset.iloc[sample_indices_].reset_index(drop=True).copy()

# 2) Añadir la predicción completa y calcular error_abs
y_pred_full = stack.predict(X_aug)
df_aug["error_abs"] = np.abs(y_aug.values - y_pred_full)

# 3) Seleccionar sólo las columnas fisiológicas
physio_cols = [
    col for col in df_aug.columns
    if col.endswith("_mean") or "_ratio" in col or "_index" in col
]

# 4) Calcular correlaciones entre error_abs y esas variables
corr_err = df_aug[["error_abs"] + physio_cols] \
    .corr()["error_abs"] \
    .sort_values(ascending=False)

# 5) Mostrar las 10 más altas
print("Top 10 correlaciones entre error_abs y perfiles fisiológicos:")
print(corr_err.head(10).to_string(float_format="{:.4f}".format))

# ───────────── 26cuater. TOP 10 % ERRORES ─────────────
umbral = df_aug["error_abs"].quantile(0.9)
top_err = df_aug[df_aug["error_abs"] >= umbral]
print_header("Jugadores top 10% error absoluto")
print(top_err[["player","error_abs","report_role","report_performance_evaluation"]]
      .sort_values("error_abs", ascending=False).to_string(index=False))

# ───────────── 27. CLUSTERING COMPARATIVO ─────────────
print_header("Clustering Comparativo (DBSCAN, GMM, KMeans)")

# Usamos el mismo embedding que para HDBSCAN (scale + select)
X_embed = opt_xgb_pipe[:-1].transform(X_aug)

# 1) DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels_db = dbscan.fit_predict(X_embed)
print("DBSCAN clusters:", pd.Series(labels_db).value_counts().sort_index().to_dict())
if len(set(labels_db)) - ( -1 in labels_db ) > 1:
    sil_db = silhouette_score(X_embed[labels_db != -1], labels_db[labels_db != -1])
    print(f"Silhouette DBSCAN (core only): {sil_db:.3f}")

# 2) Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, random_state=42)
labels_gmm = gmm.fit_predict(X_embed)
print("GMM clusters:", pd.Series(labels_gmm).value_counts().sort_index().to_dict())
print(f"Silhouette GMM: {silhouette_score(X_embed, labels_gmm):.3f}")

# 3) KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
labels_km = kmeans.fit_predict(X_embed)
print("KMeans clusters:", pd.Series(labels_km).value_counts().sort_index().to_dict())
print(f"Silhouette KMeans: {silhouette_score(X_embed, labels_km):.3f}")

# ───────────── 27bis. CLUSTERING JERÁRQUICO ─────────────
print_header("Clustering Jerárquico – Ward")
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
Z = linkage(X_embed, method="ward")
labels_h = fcluster(Z, t=3, criterion="maxclust")
print(" Jerárquico clusters:", np.bincount(labels_h))
# (Opcional: dendrogram(Z))

# ───────────── 27bis.b. DENDROGRAMA – Ward ─────────────
print_header("Dendrograma – Clustering Jerárquico (Ward)")
plt.figure(figsize=(8, 5))
dendrogram(
    Z,
    truncate_mode='level',  # mostrar solo primeros niveles
    p=5,                    # hasta 5 ramas
    leaf_rotation=90,
    color_threshold=None
)
plt.title("Dendrograma Clustering Jerárquico (Ward)")
plt.xlabel("Muestra")
plt.ylabel("Distancia")
save_fig("dendrogram_ward")
plt.show()

# ───────────── 27ter. CLUSTERING HDBSCAN ─────────────
print_header("Clustering HDBSCAN")
hdb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5)
labels_hdb = hdb.fit_predict(X_embed)
print("HDBSCAN clusters:", pd.Series(labels_hdb).value_counts().sort_index().to_dict())
# opcional: silhouette solo si hay más de 1 cluster
if len(set(labels_hdb)) > 1:
    sil_hdb = silhouette_score(X_embed[labels_hdb != -1], labels_hdb[labels_hdb != -1])
    print(f"Silhouette HDBSCAN: {sil_hdb:.3f}")

# ───────────── 27ter. MÉTRICAS DE VALIDACIÓN DE CLUSTERS ─────────────
print_header("Validación KMeans – CH y DB scores")
print(" Calinski-Harabasz:", calinski_harabasz_score(X_embed, labels_km))
print(" Davies-Bouldin   :", davies_bouldin_score(X_embed, labels_km))

# ───────────── 28. PERMUTATION FEATURE IMPORTANCE ─────────────
print_header("Permutation Feature Importance (XGB Optuna)")

# Calculamos la importancia por permutación sobre el set aumentado
perm_res = permutation_importance(
    opt_xgb_pipe,
    X_aug,
    y_aug,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

# Usamos como índice los nombres de las features originales (X_aug tiene shape (n_samples, n_features))
feat_names = feature_cols  # lista de los nombres de las columnas de X_aug

# Creamos una serie con la media de las importancias, indexada por nombre de feature
perm_imp_ser = pd.Series(perm_res.importances_mean, index=feat_names)

# Mostramos las top 10 features más importantes
print(perm_imp_ser.sort_values(ascending=False).head(10))

# ───────────── 28bis. ANOVA CLUSTERS vs EVALUACIÓN ─────────────
print_header("ANOVA entre clusters KMeans y evaluación real")
from scipy.stats import f_oneway
groups_eval = [df_aug["report_performance_evaluation"][labels_km==i].dropna()
               for i in np.unique(labels_km)]
f, p = f_oneway(*groups_eval)
print(f"F-statistic: {f:.3f}, p-value: {p:.3e}")

# ───────────── 29. CLUSTERING COMPARATIVO ─────────────
print_header("Clustering Comparativo: DBSCAN, GMM, KMeans")

# Obtenemos embeddings de las features seleccionadas
X_embed = opt_xgb_pipe[:-1].transform(X_aug)

# DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_embed)
print("DBSCAN clusters:", np.bincount(db + 1))  # +1 para contar -1

# Gaussian Mixture Model (3 componentes)
gmm = GaussianMixture(n_components=3, random_state=42).fit_predict(X_embed)
print("GMM clusters:", np.bincount(gmm))

# KMeans (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42).fit_predict(X_embed)
print("KMeans clusters:", np.bincount(kmeans))

# ───────────── 30. PCA PREVIO A t-SNE ─────────────
print_header("PCA previo a t-SNE")

# Embedding de features seleccionadas (scale + select)
X_embed = opt_xgb_pipe[:-1].transform(X_aug)

# Número de características en el embedding
n_features = X_embed.shape[1]
# Queremos reducir hasta 10, pero no más que n_features
n_pca = min(10, n_features)

# Reducimos a n_pca dimensiones antes de t-SNE para acelerar cómputo
pca = PCA(n_components=n_pca, random_state=42)
X_pca = pca.fit_transform(X_embed)

# t-SNE en 2D con embeddings reducidos
tsne_pca = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_pca)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    x=tsne_pca[:, 0],
    y=tsne_pca[:, 1],
    hue=kmeans,      # reutiliza etiquetas de KMeans
    palette="tab10",
    legend="brief",
    s=50,
)
plt.title("t-SNE (post-PCA) coloreado por clusters KMeans")
plt.show()

# ───────────── 31. GUARDADO DE FIGURAS FINALES ─────────────
print_header("Guardado de Figuras Finales")

def save_fig(name: str):
    """Guarda la figura más reciente en analysis_outputs."""
    fig_path = Path(BASE_PATH) / "analysis_outputs" / f"{name}.png"
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Figura guardada: {fig_path}")

# Ahora llamamos a save_fig para cada figura que queramos conservar
save_fig("clusters_comparative_dbscan")
save_fig("clusters_comparative_gmm")
save_fig("clusters_comparative_kmeans")
save_fig("tsne_post_pca_kmeans")

save_fig("confusion_matrix_ordinal_aug")
save_fig("shap_beeswarm_aug")

# ───────────── 32. METADATOS DE EJECUCIÓN ─────────────
print_header("Metadatos de Ejecución")
meta = {
    "executed_at": datetime.datetime.now().isoformat(),
    "user":        getpass.getuser(),
    "python":      platform.python_version(),
    "platform":    platform.platform(),
}
meta_df = pd.Series(meta)
meta_path = Path(BASE_PATH) / "analysis_outputs" / "run_metadata_versión_definitiva.csv"
meta_df.to_csv(meta_path, header=False)
print(f"Metadatos guardados en: {meta_path}")

# ───────────── 23. STACKING FINAL DE MODELOS ─────────────
stack = StackingRegressor(
    estimators=[
        ("elastic", en_pipe),
        ("xgb",     xgb_pipe),
        ("opt_xgb", opt_xgb_pipe)
    ],
    final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
    n_jobs=-1
)
stack.fit(X_aug, y_aug)

stack_metrics = evaluate_pipeline(stack, "Stacking", X_aug, y_aug, groups_aug)
print(pd.DataFrame([stack_metrics]).to_string(index=False, float_format="{:.3f}".format))

# — EJEMPLO DE PREDICCIÓN VS REAL (primeras 10 muestras) —
y_pred_stack_full = stack.predict(X_aug)
df_example = pd.DataFrame({
    "true_eval": y_aug.reset_index(drop=True),
    "pred_eval": np.round(y_pred_stack_full, 3)
})
df_example["abs_error"] = np.round((df_example.true_eval - df_example.pred_eval).abs(), 3)
print(df_example.head(20))

# ───────────── 33. MAIN GUARD ─────────────
if __name__ == "__main__":
    print_header("¡Análisis V3 – VERSIÓN DEFINITIVA completado exitosamente!")