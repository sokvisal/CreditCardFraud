import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from lightgbm import LGBMClassifier


def load_dataframe():
    df = pd.read_csv("./data/creditcard.csv")

    features = df.drop(["Time", "Class"], axis=1)
    labels = df["Class"]

    # train and test split
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    return x_train, x_test, y_train, y_test


def get_model(**kwargs):
    model = LGBMClassifier(
        objective="binary",
        is_unbalance=True,
        subsample=None,
        subsample_freq=None,
        colsample_bytree=None,
        **kwargs
    )
    return model


def search_parameters(x_train, y_train):

    model = get_model()

    opt = BayesSearchCV(
        model,
        {
            "boosting_type": Categorical(["gbdt", "dart"]),
            "learning_rate": Real(1e-6, 0.3, prior="log-uniform"),
            "num_leaves": Integer(1, 32),
            "n_estimators": Integer(100, 300),
            "min_split_gain": Real(0, 1),
            "reg_alpha": Real(1e-6, 10, prior="log-uniform"),
            "reg_lambda": Real(1e-6, 10, prior="log-uniform"),
            "feature_fraction": Real(0.6, 1),
            "bagging_fraction": Real(0.6, 1),
            "bagging_freq": Integer(0, 7),
            "min_child_samples": Integer(50, 150),
            "min_child_weight": Real(1e-3, 3, prior="log-uniform"),
            "skip_drop": Real(0.5, 1),
            "drop_rate": Real(0.1, 0.5),
            "max_drop": Integer(25, 75),
        },
        scoring=make_scorer(f1_score),
        n_jobs=4,
        n_iter=20,
        cv=5,
        random_state=42,
        verbose=1,
    )

    # executes bayesian optimization
    _ = opt.fit(x_train, y_train)

    # rank of test score
    results = pd.DataFrame(opt.cv_results_).sort_values(by="rank_test_score")
    # best params
    best_params = opt.best_params_

    return results, best_params


def main():
    x_train, x_test, y_train, y_test = load_dataframe()

    results, best_params = search_parameters(x_train, y_train)

    # fit model using best params
    lgbm_model = get_model(**best_params)
    best_model = lgbm_model.fit(x_train, y_train)

    # predict label
    y_pred = best_model.predict(x_test)

    return best_params, results, y_pred, y_test


if __name__ == "__main__":
    main()
