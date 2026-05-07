"""
cs2_kill_predictor/trainer.py

Trenuje model XGBoost/RandomForest do przewidywania killi w CS2.
Zapisuje model jako .pkl.

Użycie:
    python trainer.py --demos ./dema/ --output model.pkl
    python trainer.py --demos ./dema/ --output model.pkl --model xgboost
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from demo_parser import CS2DemoParser, FeatureBuilder


# ─────────────────────────────────────────────
# Kolumny używane do trenowania (bez metadanych)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "health", "armor", "is_crouching", "is_in_air", "is_scoped",
    "flash_duration", "ammo", "weapon_cat", "spotted",
    "speed", "speed_z",
    "dist_nearest_enemy", "dist_mean_enemy",
    "enemies_alive", "enemies_low_hp",
    "teammates_alive", "team_total_hp", "team_min_hp",
]


# ─────────────────────────────────────────────────────────────
# Główna klasa trenera
# ─────────────────────────────────────────────────────────────

class KillPredictor:
    """
    Opakowuje cały pipeline: parsowanie dem → budowanie datasetu → trening → zapis .pkl
    """

    def __init__(
        self,
        seconds_before_kill: float = 5.0,
        tick_rate: int = 64,
        sample_every: int = 4,
        model_type: str = "xgboost",
    ):
        self.seconds_before = seconds_before_kill
        self.tick_rate = tick_rate
        self.sample_every = sample_every
        self.model_type = model_type

        self.parser = CS2DemoParser(seconds_before_kill, tick_rate)
        self.feature_builder = FeatureBuilder()
        self.pipeline = None
        self.feature_importance = None
        self.threshold = 0.5  # można dostroić po treningu

    # ──────────────────────────────────────────
    def build_dataset(self, demo_dir: str) -> pd.DataFrame:
        """Parsuje dema i buduje pełny dataset."""
        print("\n" + "═"*60)
        print("  KROK 1: Parsowanie dem")
        print("═"*60)

        ticks_df, kills_df = self.parser.parse_demo_directory(demo_dir)

        print("\n" + "═"*60)
        print("  KROK 2: Budowanie cech")
        print("═"*60)

        all_features = []
        demo_ids = ticks_df["demo_id"].unique()

        for demo_id in demo_ids:
            print(f"\n  Przetwarzanie demo {demo_id}...")
            feats_df = self.feature_builder.build_features_from_ticks(
                ticks_df, kills_df, demo_id,
                ticks_before_kill=int(self.seconds_before * self.tick_rate),
                sample_every=self.sample_every,
            )
            all_features.append(feats_df)

        dataset = pd.concat(all_features, ignore_index=True)
        print(f"\n  ✓ Dataset: {len(dataset):,} próbek | Kille: {dataset['label'].sum():,} ({dataset['label'].mean()*100:.1f}%)")
        return dataset

    # ──────────────────────────────────────────
    def _build_model(self, class_weight_ratio: float):
        """Buduje model (XGBoost lub RandomForest)."""

        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            print("  Model: XGBoostClassifier")
            model = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_weight_ratio,  # obsługa nierównych klas
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boost":
            print("  Model: GradientBoostingClassifier")
            model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            print("  Model: RandomForestClassifier")
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model),
        ])
        return pipeline

    # ──────────────────────────────────────────
    def train(self, dataset: pd.DataFrame, output_path: str = "model.pkl"):
        """Trenuje model i zapisuje do .pkl."""

        print("\n" + "═"*60)
        print("  KROK 3: Trening modelu")
        print("═"*60)

        # Przygotowanie danych
        X = dataset[FEATURE_COLS].fillna(0).astype(np.float32)
        y = dataset["label"].values

        # Proporcja klas (kille są rzadsze)
        neg, pos = (y == 0).sum(), (y == 1).sum()
        class_weight_ratio = neg / max(pos, 1)
        print(f"\n  Klasy: 0={neg:,} | 1={pos:,} | ratio={class_weight_ratio:.2f}")

        # Podział train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

        # Budowa i trening modelu
        self.pipeline = self._build_model(class_weight_ratio)

        print("\n  Trenowanie...")
        self.pipeline.fit(X_train, y_train)

        # ── Ewaluacja ──
        print("\n" + "═"*60)
        print("  KROK 4: Ewaluacja")
        print("═"*60)

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]

        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Brak killa", "Kill wkrótce"]))

        auc = roc_auc_score(y_test, y_proba)
        print(f"  ROC-AUC: {auc:.4f}")

        # Dobór threshold (maksymalizuj F1 dla klasy 1)
        prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        print(f"  Optymalny threshold: {self.threshold:.3f} (F1={f1_scores[best_idx]:.3f})")

        # Feature importance
        self._extract_feature_importance()

        # Cross-validation
        print("\n  Cross-validation (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Zapis modelu
        self._save(output_path, auc, cv_scores.mean())
        self._plot_results(y_test, y_pred, y_proba, output_path)

        return auc

    # ──────────────────────────────────────────
    def _extract_feature_importance(self):
        """Wyciąga ważność cech z modelu."""
        try:
            clf = self.pipeline.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importance = clf.feature_importances_
                self.feature_importance = pd.Series(importance, index=FEATURE_COLS).sort_values(ascending=False)
                print("\n  Top 10 najważniejszych cech:")
                for feat, imp in self.feature_importance.head(10).items():
                    bar = "█" * int(imp * 50)
                    print(f"    {feat:<30} {bar} {imp:.4f}")
        except Exception as e:
            print(f"  [WARN] Nie można wyciągnąć feature importance: {e}")

    # ──────────────────────────────────────────
    def _save(self, output_path: str, auc: float, cv_auc: float):
        """Zapisuje model + metadane jako .pkl."""
        model_data = {
            "pipeline": self.pipeline,
            "feature_cols": FEATURE_COLS,
            "threshold": self.threshold,
            "feature_importance": self.feature_importance,
            "metadata": {
                "seconds_before_kill": self.seconds_before,
                "tick_rate": self.tick_rate,
                "model_type": self.model_type,
                "roc_auc": auc,
                "cv_roc_auc": cv_auc,
                "trained_at": datetime.now().isoformat(),
            }
        }
        joblib.dump(model_data, output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  ✓ Model zapisany: {output_path} ({size_mb:.1f} MB)")

    # ──────────────────────────────────────────
    def _plot_results(self, y_test, y_pred, y_proba, output_path: str):
        """Generuje wykresy ewaluacji."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("CS2 Kill Predictor — Ewaluacja modelu", fontsize=14)

        # 1. Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[0],
                    xticklabels=["Brak killa", "Kill"],
                    yticklabels=["Brak killa", "Kill"],
                    cmap="Blues")
        axes[0].set_title("Macierz konfuzji")
        axes[0].set_ylabel("Prawdziwa etykieta")
        axes[0].set_xlabel("Predykcja")

        # 2. ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("Krzywa ROC")
        axes[1].legend()

        # 3. Feature importance
        if self.feature_importance is not None:
            top10 = self.feature_importance.head(10)
            axes[2].barh(top10.index[::-1], top10.values[::-1])
            axes[2].set_title("Top 10 cech")
            axes[2].set_xlabel("Ważność")

        plt.tight_layout()
        plot_path = output_path.replace(".pkl", "_eval.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Wykres zapisany: {plot_path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Trener modelu AI do przewidywania killi w CS2"
    )
    p.add_argument("--demos", required=True, help="Katalog z plikami .dem")
    p.add_argument("--output", default="cs2_kill_model.pkl", help="Ścieżka wyjściowa modelu .pkl")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="Ile sekund przed killem oznaczamy jako 'ryzykowny' (domyślnie: 5)")
    p.add_argument("--model", choices=["xgboost", "random_forest", "gradient_boost"],
                   default="xgboost", help="Typ modelu")
    p.add_argument("--sample-every", type=int, default=4,
                   help="Próbkuj co N ticków (domyślnie: 4 = ~16 próbek/s przy 64 tick)")
    p.add_argument("--save-dataset", default=None,
                   help="Opcjonalnie zapisz dataset do pliku .csv")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═"*60)
    print("  CS2 KILL PREDICTOR — TRENER")
    print("═"*60)
    print(f"  Katalog z demami : {args.demos}")
    print(f"  Wyjście modelu   : {args.output}")
    print(f"  Typ modelu       : {args.model}")
    print(f"  Sekund przed     : {args.seconds}s")

    predictor = KillPredictor(
        seconds_before_kill=args.seconds,
        model_type=args.model,
        sample_every=args.sample_every,
    )

    # Buduj dataset
    dataset = predictor.build_dataset(args.demos)

    if args.save_dataset:
        dataset.to_csv(args.save_dataset, index=False)
        print(f"  Dataset zapisany: {args.save_dataset}")

    # Trenuj
    auc = predictor.train(dataset, args.output)

    print(f"\n{'═'*60}")
    print(f"  GOTOWE! ROC-AUC: {auc:.4f}")
    print(f"  Model: {args.output}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
