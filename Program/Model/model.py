def build_models(pos_weight):
    models = {}

    # 1. Logistic Regression (baseline)
    from sklearn.linear_model import LogisticRegression
    models["logreg"] = LogisticRegression(
        max_iter=2000, C=0.5,
        class_weight="balanced",
        solver="lbfgs", penalty="l2"
    )

    # 2. Random Forest (log2 features)
    from sklearn.ensemble import RandomForestClassifier
    models["random_forest"] = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        max_features="log2",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1, random_state=42
    )

    # 3. XGBoost
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1, random_state=42,
            verbosity=0
        )
    except ImportError:
        print("xgboost tidak terpasang, skip")

    # 4. LightGBM
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=400,
            num_leaves=63,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            is_unbalance=True,
            n_jobs=-1, random_state=42,
            verbose=-1
        )
    except ImportError:
        print("lightgbm tidak terpasang, skip")

    return models
