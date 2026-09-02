import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import fill_missing, build_train_features, build_predict_features, get_model


# ─── Train ───────────────────────────────────────────────────────────────────

def train(df):
    """
    df:   timestamp-indexed DataFrame，包含 'consumption' 列
    返回: model, slot_means
    """
    df, slot_means = fill_missing(df, mode="train")
    X, y, _ = build_train_features(df)

    model = get_model()
    model.fit(X, y)

    return model, slot_means


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict(df, model, slot_means, target_date=None):
    """
    df:   timestamp-indexed DataFrame，包含 'consumption' 列
    返回: base DataFrame，含目标日期和预测列 'predicted'
    """
    df, _ = fill_missing(df, slot_means=slot_means, mode="predict")
    X, base, target_date = build_predict_features(df, target_date=target_date)
    base["predicted"] = model.predict(X)

    return base, target_date
