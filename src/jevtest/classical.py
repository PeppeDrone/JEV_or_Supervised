"""M2 (XGBoost) classical tabular baseline, ported from neurojev.evaluation.baselines.

Randomized search with inner stratified CV (log loss). Everything is fitted on
outer-training patients only. Native XGBoost missing-value handling; categoricals
are one-hot encoded after constant imputation.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from jevtest.data import predictor_matrix
from jevtest.models import save_pipeline, write_manifest


ESTIMATORS = {
    "M2": lambda seed, threads: XGBClassifier(
        random_state=seed, n_jobs=threads, tree_method="hist", eval_metric="mlogloss",
    ),
}
SEARCH = {
    "M2": {
        "model__max_depth": [2, 3, 5], "model__learning_rate": [0.02, 0.05, 0.1],
        "model__n_estimators": [100, 200, 400],
        "model__subsample": [0.7, 1.0], "model__colsample_bytree": [0.7, 1.0],
        "model__min_child_weight": [1, 5, 10],
        "model__reg_alpha": [0, 0.1, 1], "model__reg_lambda": [1, 5, 10],
    },
}


def preprocessing(X, model="M2"):
    """Column prep. ``model=="M2"``: numeric passthrough; otherwise impute+scale (e.g. ICL)."""
    numerical = list(X.select_dtypes(include=np.number).columns)
    categorical = [x for x in X if x not in numerical]
    numeric = "passthrough" if model == "M2" else Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    category = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("numeric", numeric, numerical), ("category", category, categorical)],
                             verbose_feature_names_out=True)


def fit_fold(X, y, train, model, study, fold):
    """Fit on one outer-training set; returns the predictor."""
    inner = min(study["inner_folds"], int(np.bincount(y[train]).min()))
    cv = StratifiedKFold(inner, shuffle=True, random_state=study["seed"] + fold)
    pipe = Pipeline([("preprocess", preprocessing(X, model)), ("model", ESTIMATORS[model](study["seed"], study["threads"]))])
    search = RandomizedSearchCV(pipe, SEARCH[model], n_iter=study["search_iterations"], scoring="neg_log_loss", cv=cv,
                                random_state=study["seed"], n_jobs=1, error_score="raise", refit=True)
    search.fit(X.iloc[train], y[train])
    return search.best_estimator_


def out_of_fold(run, X, model, name=None):
    P = np.zeros((len(run.y), len(run.classes)))
    for fold, (train, test) in enumerate(run.splits):
        run.progress(stage=name or model, fold=fold)
        fitted = fit_fold(X, run.y, train, model, run.study, fold)
        save_pipeline(run, name or model, fold, fitted, train, {"columns": list(X.columns)})
        P[test] = fitted.predict_proba(X.iloc[test])
    return P


def run_classical(run):
    X = predictor_matrix(run)
    write_manifest(run)
    run.save_predictions("M2", out_of_fold(run, X, "M2"))
