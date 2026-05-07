"""
cs2_kill_predictor/predictor.py

Używa wytrenowanego modelu .pkl do przewidywania killi w czasie rzeczywistym.
Przy wysokim prawdopodobieństwie killa sugeruje przełączenie kamery.

Użycie:
    python predictor.py --model cs2_kill_model.pkl --demo mecz.dem
    python predictor.py --model cs2_kill_model.pkl --demo mecz.dem --output highlights.json
"""

import joblib
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

from demo_parser import CS2DemoParser, FeatureBuilder


class LiveKillPredictor:
    """
    Przetwarza dema i przewiduje momenty killi.
    Generuje listę 'highlight moments' z sugestią kamery.
    """

    def __init__(self, model_path: str):
        print(f"  Ładowanie modelu: {model_path}")
        model_data = joblib.load(model_path)

        self.pipeline = model_data["pipeline"]
        self.feature_cols = model_data["feature_cols"]
        self.threshold = model_data.get("threshold", 0.5)
        self.metadata = model_data.get("metadata", {})

        seconds = self.metadata.get("seconds_before_kill", 5.0)
        tick_rate = self.metadata.get("tick_rate", 64)

        self.parser = CS2DemoParser(seconds, tick_rate)
        self.feature_builder = FeatureBuilder()

        print(f"  Model: {self.metadata.get('model_type', 'unknown')}")
        print(f"  ROC-AUC: {self.metadata.get('roc_auc', '?'):.4f}")
        print(f"  Threshold: {self.threshold:.3f}")
        print(f"  Trenowany: {self.metadata.get('trained_at', '?')}")

    # ──────────────────────────────────────────
    def predict_demo(
        self,
        demo_path: str,
        sample_every: int = 4,
        cooldown_ticks: int = 128,
    ) -> list[dict]:
        """
        Przewiduje momenty killi w całym demie.

        Args:
            demo_path:      Ścieżka do pliku .dem
            sample_every:   Co ile ticków liczymy predykcję
            cooldown_ticks: Minimalna przerwa między sugestiami kamery (żeby nie migać)

        Returns:
            Lista słowników z momentami sugestii kamery:
            {
                "tick": int,
                "time_seconds": float,
                "player_id": str,
                "player_name": str,
                "team": str,
                "kill_probability": float,
                "camera_switch_recommended": bool
            }
        """
        print(f"\n  Parsowanie: {demo_path}")
        ticks_df, kills_df = self.parser.parse_demo(demo_path)
        ticks_df["demo_id"] = 0
        kills_df["demo_id"] = 0

        tick_rate = self.metadata.get("tick_rate", 64)
        highlights = []

        # Cooldown per gracz (żeby nie sugerować przełączenia co 4 ticki)
        last_switch_tick: dict[str, int] = {}

        unique_ticks = sorted(ticks_df["tick"].unique())
        sampled_ticks = unique_ticks[::sample_every]

        print(f"  Analiza {len(sampled_ticks):,} ticków...")

        for tick in sampled_ticks:
            tick_data = ticks_df[ticks_df["tick"] == tick]

            ct_players = tick_data[tick_data["team_num"] == 3] if "team_num" in tick_data.columns else tick_data.head(0)
            t_players = tick_data[tick_data["team_num"] == 2] if "team_num" in tick_data.columns else tick_data.head(0)

            for _, player_row in tick_data.iterrows():
                if (player_row.get("health", 0) or 0) <= 0:
                    continue

                player_id = str(player_row.get("steamid", player_row.get("name", "unknown")))
                player_name = str(player_row.get("name", player_id))
                team_num = player_row.get("team_num", 0)

                if team_num == 3:
                    team_data = ct_players[ct_players.index != player_row.name]
                    enemy_data = t_players
                else:
                    team_data = t_players[t_players.index != player_row.name]
                    enemy_data = ct_players

                # Buduj wektor cech
                feats = self.feature_builder.build_player_features(player_row, team_data, enemy_data)
                X = pd.DataFrame([feats])[self.feature_cols].fillna(0).astype(np.float32)

                # Predykcja
                kill_prob = float(self.pipeline.predict_proba(X)[0, 1])
                recommend = kill_prob >= self.threshold

                # Cooldown — nie dodawaj jeśli ostatnio sugerowaliśmy tę kamerę
                if recommend:
                    last_for_player = last_switch_tick.get(player_id, -9999)
                    if tick - last_for_player < cooldown_ticks:
                        recommend = False
                    else:
                        last_switch_tick[player_id] = tick

                if kill_prob > 0.3:  # loguj wszystkie powyżej 30%
                    highlights.append({
                        "tick": int(tick),
                        "time_seconds": round(tick / tick_rate, 2),
                        "player_id": player_id,
                        "player_name": player_name,
                        "team": "CT" if team_num == 3 else "T",
                        "kill_probability": round(kill_prob, 4),
                        "camera_switch_recommended": recommend,
                    })

        # Sortuj po prawdopodobieństwie (malejąco) w obrębie czasu
        highlights.sort(key=lambda x: (x["tick"], -x["kill_probability"]))
        return highlights

    # ──────────────────────────────────────────
    def get_camera_schedule(self, highlights: list[dict], min_prob: float = None) -> list[dict]:
        """
        Z listy highlights buduje finalny harmonogram kamery —
        który gracz powinien być na ekranie w jakim momencie.

        Args:
            highlights: Wynik predict_demo()
            min_prob:   Minimalny próg (domyślnie: threshold modelu)

        Returns:
            Lista momentów przełączenia kamery posortowana chronologicznie.
        """
        if min_prob is None:
            min_prob = self.threshold

        schedule = [h for h in highlights if h["camera_switch_recommended"] and h["kill_probability"] >= min_prob]
        schedule.sort(key=lambda x: x["tick"])

        print(f"\n  Harmonogram kamery: {len(schedule)} przełączeń")
        for entry in schedule[:20]:  # pokaż pierwsze 20
            t = entry["time_seconds"]
            minutes = int(t // 60)
            seconds = t % 60
            prob = entry["kill_probability"] * 100
            print(f"    [{minutes:02d}:{seconds:05.2f}] {entry['player_name']:<20} ({entry['team']}) — {prob:.1f}% kill")

        if len(schedule) > 20:
            print(f"    ... i {len(schedule) - 20} więcej")

        return schedule


# ─────────────────────────────────────────────────────────────
# Symulator kamery — pokazuje jak wyglądałoby przełączanie
# ─────────────────────────────────────────────────────────────

class CameraSimulator:
    """
    Symuluje system kamer — na podstawie harmonogramu decyduje,
    kogo kamera powinna śledzić w danym ticku.
    """

    def __init__(self, schedule: list[dict], default_player: str = ""):
        self.schedule = sorted(schedule, key=lambda x: x["tick"])
        self.current_camera = default_player
        self._idx = 0

    def get_camera_at(self, tick: int) -> tuple[str, float]:
        """
        Zwraca (player_id, kill_probability) dla danego ticku.
        """
        for entry in reversed([e for e in self.schedule if e["tick"] <= tick]):
            return entry["player_id"], entry["kill_probability"]
        return self.current_camera, 0.0

    def export_to_json(self, path: str):
        """Eksportuje harmonogram do JSON (można wczytać w innych narzędziach)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.schedule, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Harmonogram kamery zapisany: {path}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Predyktor killi CS2 — sugestie kamery"
    )
    p.add_argument("--model", required=True, help="Ścieżka do modelu .pkl")
    p.add_argument("--demo", required=True, help="Ścieżka do pliku .dem")
    p.add_argument("--output", default="camera_schedule.json",
                   help="Plik wyjściowy z harmonogramem kamery")
    p.add_argument("--min-prob", type=float, default=None,
                   help="Minimalny próg prawdopodobieństwa (domyślnie: z modelu)")
    p.add_argument("--cooldown", type=int, default=128,
                   help="Cooldown między przełączeniami kamery (w tickach)")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═"*60)
    print("  CS2 KILL PREDICTOR — PREDYKCJA")
    print("═"*60)

    predictor = LiveKillPredictor(args.model)

    highlights = predictor.predict_demo(
        args.demo,
        cooldown_ticks=args.cooldown,
    )

    schedule = predictor.get_camera_schedule(highlights, args.min_prob)

    cam_sim = CameraSimulator(schedule)
    cam_sim.export_to_json(args.output)

    print(f"\n{'═'*60}")
    print(f"  GOTOWE! Znaleziono {len(schedule)} momentów do przełączenia kamery.")
    print(f"  Plik: {args.output}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
