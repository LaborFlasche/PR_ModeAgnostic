import os
import numpy as np
import pandas as pd

# R_HOME muss gesetzt sein, damit rpy2 R findet
# (wird vom Setup-Skript gesetzt, oder manuell via: export R_HOME=$(R RHOME))
try:
    from shaprpy import explain
except ImportError as e:
    raise ImportError(
        "shaprpy nicht gefunden. Bitte zuerst setup_shapr_env.sh ausführen.\n"
        f"Original-Fehler: {e}"
    )


def compute_shapr_explanations(
    model,
    x_train: pd.DataFrame,
    x_explain: pd.DataFrame,
    phi0: float = 0.0,
    approach: str = "empirical",
    seed: int = 42,
) -> np.ndarray:
    """
    Berechnet SHAPr-Shapley-Werte für x_explain.

    Parameter
    ----------
    model      : trainiertes sklearn/xgboost/keras-Modell
    x_train    : Trainingsdaten (pd.DataFrame) – für Hintergrundverteilung
    x_explain  : Erklärungsdaten (pd.DataFrame) – diese Punkte werden erklärt
    phi0       : Baseline-Vorhersage (z.B. Trainings-Mittelwert)
    approach   : SHAPr-Approach (Standard: "empirical")
    seed       : Zufalls-Seed für Reproduzierbarkeit

    Rückgabe
    --------
    np.ndarray mit Shape (n_explain, n_features) – Shapley-Werte
    """

    # sicherstellen dass x_train und x_explain DataFrames sind
    if not isinstance(x_train, pd.DataFrame):
        x_train = pd.DataFrame(x_train)
    if not isinstance(x_explain, pd.DataFrame):
        x_explain = pd.DataFrame(x_explain)

    explanation = explain(
        model=model,
        x_train=x_train,
        x_explain=x_explain,
        approach=approach,
        phi0=phi0,
        seed=seed,
    )

    results = explanation.get_results()
    shapley_values = results["shapley_est"]

    # in numpy-Array konvertieren (ohne phi0-Spalte, falls vorhanden)
    if isinstance(shapley_values, pd.DataFrame):
        # erste Spalte ist oft "none" (phi0), entfernen falls vorhanden
        feature_cols = [c for c in shapley_values.columns if c != "none"]
        return shapley_values[feature_cols].values
    return np.array(shapley_values)


# ──────────────────────────────────────────────────────────────
# Selbsttest: direkt ausführen zum Testen
# python Benchmarking/backends/shapr_backend.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" shapr_backend.py – Selbsttest")
    print("=" * 55)

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    print("\n▶ Lade Beispiel-Datensatz (Diabetes)...")
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    # Für den Test nur 5 Erklärungspunkte
    X_explain = X_test.iloc[:5]

    print("▶ Trainiere RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    phi0 = float(y_train.mean())
    print(f"▶ phi0 (Trainings-Mittelwert): {phi0:.4f}")

    print("▶ Berechne SHAPr-Erklärungen (approach='empirical')...")
    shap_values = compute_shapr_explanations(
        model=model,
        x_train=X_train,
        x_explain=X_explain,
        phi0=phi0,
        approach="empirical",
        seed=42,
    )

    print(f"\n✅ Shapley-Werte berechnet!")
    print(f"   Shape: {shap_values.shape}  (n_explain × n_features)")
    print(f"\n   Feature-Namen: {list(X_train.columns)}")
    print(f"\n   Erste Zeile (Erklärung für Punkt 0):")
    for feat, val in zip(X_train.columns, shap_values[0]):
        print(f"     {feat:>10s}: {val:+.4f}")

    # Sanity-Check: Summe der Shapley-Werte ≈ Vorhersage - phi0
    preds = model.predict(X_explain)
    for i in range(len(X_explain)):
        shapley_sum = shap_values[i].sum() + phi0
        pred = preds[i]
        diff = abs(shapley_sum - pred)
        status = "✅" if diff < 5 else "⚠️ "
        print(f"\n   Punkt {i}: Vorhersage={pred:.2f}, "
              f"phi0+Σshapley={shapley_sum:.2f}, Diff={diff:.4f} {status}")

    print("\n" + "=" * 55)
    print(" Selbsttest abgeschlossen!")
    print("=" * 55)
