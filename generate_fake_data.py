"""
cs2_kill_predictor/generate_fake_data.py

Generuje syntetyczny dataset do testowania modelu BEZ prawdziwych dem.
Symuluje realistyczne dane gry CS2.

Uruchom: python generate_fake_data.py
"""

import numpy as np
import pandas as pd
import joblib
import argparse
from pathlib import Path

from trainer import KillPredictor, FEATURE_COLS
from sklearn.model_selection import train_test_split


WEAPON_CATS = [0, 1, 2, 3, 4, 5, 6]


def generate_synthetic_dataset(n_samples: int = 50000, kill_ratio: float = 0.15) -> pd.DataFrame:
    """
    Generuje syntetyczny dataset imitujący dane z dem CS2.
    
    Heurystyki oparte na rzeczywistej mechanice CS2:
    - Gracze bliżej wroga → wyższe prawdopodobieństwo killa
    - Gracze z AWP + scope → wyższe przy dużym dystansie
    - Niskie HP → niższe (mogą dać killa, ale też zginąć)
    - Gracz spotted przez wroga → wyższe ryzyko
    """
    np.random.seed(42)
    n_kill = int(n_samples * kill_ratio)
    n_no_kill = n_samples - n_kill

    def gen_samples(n: int, is_kill: bool) -> pd.DataFrame:
        rows = []
        for _ in range(n):
            is_sniper = np.random.random() < 0.15
            weapon_cat = 2 if is_sniper else np.random.choice([1, 3, 4, 5], p=[0.4, 0.25, 0.2, 0.15])
            is_scoped = int(is_sniper and np.random.random() < 0.8) if is_kill else int(is_sniper and np.random.random() < 0.3)

            if is_kill:
                # Kill moment — gracz blisko wroga lub dobry snajper
                if is_sniper and is_scoped:
                    dist_nearest = np.random.uniform(300, 1500)
                else:
                    dist_nearest = np.random.uniform(50, 400)
                health = np.random.randint(40, 100)
                speed = np.random.uniform(0, 100) if not is_scoped else np.random.uniform(0, 30)
                spotted = int(np.random.random() < 0.4)
                flash = np.random.uniform(0, 0.5)
            else:
                # Brak killa — różne sytuacje
                dist_nearest = np.random.uniform(200, 3000)
                health = np.random.randint(1, 100)
                speed = np.random.uniform(0, 300)
                spotted = int(np.random.random() < 0.3)
                flash = np.random.uniform(0, 3.0)

            row = {
                "health": health,
                "armor": np.random.randint(0, 100),
                "is_crouching": int(np.random.random() < (0.4 if is_kill else 0.15)),
                "is_in_air": int(np.random.random() < 0.05),
                "is_scoped": is_scoped,
                "flash_duration": flash,
                "ammo": np.random.randint(0, 30),
                "weapon_cat": weapon_cat,
                "spotted": spotted,
                "speed": speed,
                "speed_z": np.random.uniform(0, 50),
                "dist_nearest_enemy": dist_nearest,
                "dist_mean_enemy": dist_nearest + np.random.uniform(0, 500),
                "enemies_alive": np.random.randint(1, 6),
                "enemies_low_hp": np.random.randint(0, 3),
                "teammates_alive": np.random.randint(0, 5),
                "team_total_hp": np.random.randint(0, 500),
                "team_min_hp": np.random.randint(0, 100),
                "label": int(is_kill),
                "demo_id": np.random.randint(0, 10),
                "tick": np.random.randint(1000, 100000),
                "player_id": f"player_{np.random.randint(1, 11)}",
                "team": np.random.choice(["CT", "T"]),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    kill_df = gen_samples(n_kill, is_kill=True)
    no_kill_df = gen_samples(n_no_kill, is_kill=False)

    dataset = pd.concat([kill_df, no_kill_df], ignore_index=True).sample(frac=1, random_state=42)
    print(f"  ✓ Wygenerowano {len(dataset):,} próbek (kille: {dataset['label'].sum():,} = {dataset['label'].mean()*100:.1f}%)")
    return dataset


def main():
    p = argparse.ArgumentParser(description="Generuj syntetyczny dataset i trenuj model testowy")
    p.add_argument("--output", default="cs2_kill_model_test.pkl", help="Ścieżka wyjściowa modelu")
    p.add_argument("--samples", type=int, default=50000, help="Liczba próbek")
    p.add_argument("--model", default="xgboost", choices=["xgboost", "random_forest", "gradient_boost"])
    args = p.parse_args()

    print("\n" + "═"*60)
    print("  GENEROWANIE SYNTETYCZNEGO DATASETU")
    print("═"*60)

    dataset = generate_synthetic_dataset(args.samples)

    print("\n" + "═"*60)
    print("  TRENING NA DANYCH SYNTETYCZNYCH")
    print("═"*60)

    predictor = KillPredictor(model_type=args.model)
    auc = predictor.train(dataset, args.output)

    print(f"\n  ✓ Model testowy gotowy: {args.output}")
    print(f"  ROC-AUC: {auc:.4f}")
    print("\n  Możesz teraz przetestować predykcję (wymaga prawdziwego .dem):")
    print(f"  python predictor.py --model {args.output} --demo twoj_mecz.dem")


if __name__ == "__main__":
    main()
