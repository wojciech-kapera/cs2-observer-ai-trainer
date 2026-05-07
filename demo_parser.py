"""
cs2_kill_predictor/demo_parser.py

Parsuje dema CS2 i wyciąga dane do trenowania modelu.
Wymaga: pip install demoparser2==0.41.2
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def safe_float(v, default=0.0):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except:
        return default


def safe_int(v, default=0):
    try:
        if pd.isna(v):
            return default
        return int(v)
    except:
        return default

try:
    from demoparser2 import DemoParser
    DEMOPARSER_AVAILABLE = True
except ImportError:
    DEMOPARSER_AVAILABLE = False
    print("[WARN] Zainstaluj: pip install demoparser2==0.41.2")


# Pola pobierane per tick per gracz
PLAYER_TICK_FIELDS = [
    "X", "Y", "Z",
    "pitch", "yaw",
    "velocity_X", "velocity_Y", "velocity_Z",
    "health",
    "armor_value",
    "is_crouching",
    "is_in_air",
    "active_weapon_name",
    "active_weapon_ammo",
    "is_scoped",
    "flash_duration",
    "team_num",
]


class CS2DemoParser:
    def __init__(self, seconds_before_kill: float = 5.0, tick_rate: int = 64):
        self.seconds_before = seconds_before_kill
        self.tick_rate = tick_rate
        self.ticks_before = int(seconds_before_kill * tick_rate)

    def parse_demo(self, demo_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parsuje jedno demo.
        Returns:
            ticks_df  — dane per tick per gracz
            kills_df  — eventy player_death
        """
        if not DEMOPARSER_AVAILABLE:
            raise RuntimeError("Zainstaluj demoparser2: pip install demoparser2==0.41.2")

        parser = DemoParser(demo_path)

        print(f"  [+] Parsowanie ticków...")
        # POPRAWNE API: parse_ticks(lista_pol)
        ticks_df: pd.DataFrame = parser.parse_ticks(PLAYER_TICK_FIELDS)

        print(f"  [+] Parsowanie killi...")
        # parse_event(nazwa_eventu, player=[...], other=[...])
        # player= → dodaje kolumny z prefiksem attacker_ (dla zabójcy) i user_ (dla ofiary)
        kills_df: pd.DataFrame = parser.parse_event(
            "player_death",
            player=["X", "Y", "Z", "health", "team_num"],
            other=["total_rounds_played"],
        )

        print(f"  [✓] Ticków: {len(ticks_df):,} | Killi: {len(kills_df):,}")

        # Debug — pokaż kolumny żeby łatwiej debugować
        print(f"  [i] Kolumny ticks ({len(ticks_df.columns)}): {list(ticks_df.columns)}")
        print(f"  [i] Kolumny kills ({len(kills_df.columns)}): {list(kills_df.columns)}")

        return ticks_df, kills_df

    def parse_demo_directory(self, demo_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Parsuje wszystkie .dem w katalogu."""
        demo_files = list(Path(demo_dir).glob("*.dem"))
        if not demo_files:
            raise FileNotFoundError(f"Brak plików .dem w: {demo_dir}")

        all_ticks, all_kills = [], []
        for i, dp in enumerate(demo_files):
            print(f"\n[Demo {i+1}/{len(demo_files)}] {dp.name}")
            try:
                t, k = self.parse_demo(str(dp))
                t["demo_id"] = i
                k["demo_id"] = i
                all_ticks.append(t)
                all_kills.append(k)
            except Exception as e:
                print(f"  [ERR] Pominięto {dp.name}: {e}")
                import traceback; traceback.print_exc()

        if not all_ticks:
            raise RuntimeError("Żadne demo nie zostało sparsowane.")

        return pd.concat(all_ticks, ignore_index=True), pd.concat(all_kills, ignore_index=True)


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

class FeatureBuilder:

    WEAPON_CATS = {
        1: ["ak47", "m4a1", "sg556", "aug", "famas", "galil", "m4a1_silencer"],
        2: ["awp", "ssg08", "scar20", "g3sg1"],
        3: ["glock", "usp_silencer", "p250", "deagle", "tec9", "fiveseven", "cz75a", "revolver"],
        4: ["mp9", "mac10", "mp5sd", "mp7", "p90", "ump45", "bizon"],
        5: ["nova", "xm1014", "mag7", "sawedoff", "m249", "negev"],
        6: ["knife"],
        7: ["hegrenade", "flashbang", "smokegrenade", "molotov", "incgrenade", "decoy"],
    }

    def weapon_to_category(self, name) -> int:
        if not isinstance(name, str):
            return 0
        nl = name.lower()
        for cat, weapons in self.WEAPON_CATS.items():
            if any(w in nl for w in weapons):
                return cat
        return 0

    
    def build_player_features(self, row, team_rows, enemy_rows) -> dict:
        f = {}

        f["health"]         = safe_float(row.get("health"), 100)
        f["armor"]          = safe_float(row.get("armor_value"), 0)

        f["is_crouching"]   = safe_int(bool(row.get("is_crouching", False)))
        f["is_in_air"]      = safe_int(bool(row.get("is_in_air", False)))
        f["is_scoped"]      = safe_int(bool(row.get("is_scoped", False)))

        f["flash_duration"] = safe_float(row.get("flash_duration"), 0)
        f["ammo"]           = safe_float(row.get("active_weapon_ammo"), 0)

        f["weapon_cat"]     = self.weapon_to_category(
            row.get("active_weapon_name", "")
        )

        f["spotted"] = 0

        vx = safe_float(row.get("velocity_X"), 0)
        vy = safe_float(row.get("velocity_Y"), 0)
        vz = safe_float(row.get("velocity_Z"), 0)

        f["speed"]   = float(np.sqrt(vx**2 + vy**2 + vz**2))
        f["speed_z"] = abs(vz)

        px = safe_float(row.get("X"), 0)
        py = safe_float(row.get("Y"), 0)
        pz = safe_float(row.get("Z"), 0)

        if len(enemy_rows) > 0:
            ex = enemy_rows["X"].fillna(0).astype(float)
            ey = enemy_rows["Y"].fillna(0).astype(float)
            ez = enemy_rows["Z"].fillna(0).astype(float)

            dists = np.sqrt((ex - px)**2 + (ey - py)**2 + (ez - pz)**2)

            f["dist_nearest_enemy"] = float(dists.min())
            f["dist_mean_enemy"]    = float(dists.mean())
            f["enemies_alive"]      = int(len(enemy_rows))

            f["enemies_low_hp"] = int(
                (enemy_rows["health"].fillna(100).astype(float) < 50).sum()
            )
        else:
            f["dist_nearest_enemy"] = 9999.0
            f["dist_mean_enemy"]    = 9999.0
            f["enemies_alive"]      = 0
            f["enemies_low_hp"]     = 0

        f["teammates_alive"] = int(len(team_rows))

        if len(team_rows) > 0:
            hp = team_rows["health"].fillna(0).astype(float)

            f["team_total_hp"] = float(hp.sum())
            f["team_min_hp"]   = float(hp.min())
        else:
            f["team_total_hp"] = 0.0
            f["team_min_hp"]   = 0.0

        return f



    def build_features_from_ticks(
        self,
        ticks_df: pd.DataFrame,
        kills_df: pd.DataFrame,
        demo_id: int,
        ticks_before_kill: int,
        sample_every: int = 4,
    ) -> pd.DataFrame:

        demo_ticks = ticks_df[ticks_df["demo_id"] == demo_id].copy()
        demo_kills = kills_df[kills_df["demo_id"] == demo_id].copy()

        demo_ticks = demo_ticks.replace([np.inf, -np.inf], np.nan)
        demo_ticks = demo_ticks.fillna(0)

        demo_kills = demo_kills.replace([np.inf, -np.inf], np.nan)
        demo_kills = demo_kills.fillna(0)

        # Znajdź kolumnę z ID zabójcy (attacker_steamid lub attacker_name)
        attacker_col = next(
            (c for c in ["attacker_steamid", "attacker_name"] if c in demo_kills.columns),
            None
        )
        if attacker_col is None:
            print(f"  [WARN] Brak kolumny zabójcy. Dostępne: {list(demo_kills.columns)}")
            # Fallback — użyj pierwszej kolumny która wygląda jak id
            attacker_col = demo_kills.columns[0]

        # Buduj słownik killer_id -> [tick1, tick2, ...]
        killer_ticks: dict[str, list[int]] = {}
        for _, kr in demo_kills.iterrows():
            kid  = str(kr.get(attacker_col, "?"))
            tick = int(kr.get("tick", 0))
            killer_ticks.setdefault(kid, []).append(tick)

        # Kolumna identyfikatora gracza w ticks
        pid_col = "steamid" if "steamid" in demo_ticks.columns else "name"

        all_rows = []
        sampled = sorted(demo_ticks["tick"].unique())[::sample_every]

        for tick in tqdm(sampled, desc=f"  Demo {demo_id}", leave=False):
            td = demo_ticks[demo_ticks["tick"] == tick]

            if "team_num" in td.columns:
                ct = td[td["team_num"] == 3]
                t  = td[td["team_num"] == 2]
            else:
                ct = td.iloc[:0]
                t  = td.iloc[:0]

            for _, prow in td.iterrows():
                if safe_float(prow.get("health"), 0) <= 0:
                    continue

                pid      = str(prow.get(pid_col, "?"))
                team_num = safe_int(prow.get("team_num"), 0)

                teammates = (ct if team_num == 3 else t)
                teammates = teammates[teammates.index != prow.name]
                enemies   = t if team_num == 3 else ct

                feats = self.build_player_features(prow, teammates, enemies)
                feats["tick"]      = int(tick)
                feats["player_id"] = pid
                feats["demo_id"]   = demo_id
                feats["team"]      = "CT" if team_num == 3 else "T"

                pk = killer_ticks.get(pid, [])
                feats["label"] = int(any(tick <= kt <= tick + ticks_before_kill for kt in pk))

                all_rows.append(feats)

        return pd.DataFrame(all_rows)
