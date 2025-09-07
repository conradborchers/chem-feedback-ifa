import os
import numpy as np
import pandas as pd
from collections import defaultdict
from cmdstanpy import CmdStanModel

CSV = "d_afm_stoich.csv"
STAN_FILE = "bkt_constrained.stan"
OUT_PARAMS = "chem-bkt_constrained_params.csv"
OUT_LOG    = "chem-bkt_constrained_fitlog.csv"
OUT_PREDS  = "chem-bkt_constrained_preds.csv"

TIME_CANDIDATES = ["time", "Time", "timestamp", "ms_first_response", "step_start_time", "StartTime", "start_time"]

def pick_student_col(df):
    candidates = [
        "student_id","Anon Student Id","user_id","User","student",
        "Student Id","studentId","stu_id"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pick_time_col(df):
    for c in TIME_CANDIDATES:
        if c in df.columns:
            return c
    return None

def build_sequences(df):
    """Return dict: skill -> list of np.arrays (each is a 0/1 correctness sequence), sorted by time."""
    id_col = pick_student_col(df)
    tcol   = pick_time_col(df)

    # sort globally by time if present (helps groupby stability)
    if tcol:
        df = df.sort_values(by=[tcol])
    else:
        print("[SORT] no time column found; using input order globally")

    seqs = defaultdict(list)

    # group by skill
    for skill, dsk in df.groupby("skill_name", sort=False):
        if id_col:
            # within skill, group by student; sort each group by time if available
            if tcol:
                for _, g in dsk.groupby(id_col, sort=False):
                    g = g.sort_values(by=[tcol])
                    arr = g["correct"].astype(int).to_numpy()
                    if len(arr) > 0:
                        seqs[skill].append(arr)
            else:
                for _, g in dsk.groupby(id_col, sort=False):
                    arr = g["correct"].astype(int).to_numpy()
                    if len(arr) > 0:
                        seqs[skill].append(arr)
        else:
            # no student id; treat all rows for this skill as one sequence (sorted if possible)
            if tcol:
                dsk = dsk.sort_values(by=[tcol])
            arr = dsk["correct"].astype(int).to_numpy()
            if len(arr) > 0:
                seqs[skill].append(arr)
    return seqs

def pack_sequences(seq_list):
    """Pack a list of 1D arrays into Stan data (K, N, T[K], y[N])."""
    T = np.array([len(s) for s in seq_list], dtype=int)
    y = np.concatenate(seq_list).astype(int)
    return dict(K=len(seq_list), N=int(y.size), T=T.tolist(), y=y.tolist())

def main():
    # --- load & prep ---
    print("[LOAD] reading", CSV)
    df = (pd.read_csv(CSV)
            .rename(columns={"outcome":"correct", "kc_default":"skill_name"}))
    df = df.dropna(subset=["skill_name"]).copy()
    df["correct"] = df["correct"].astype(int)

    # sort by a time column if present (global preview)
    tcol = pick_time_col(df)
    if tcol:
        df = df.sort_values(by=[tcol])
        print(f"[SORT] sorted globally by {tcol}")
    else:
        print("[SORT] no obvious time column; proceeding without global sort")

    id_col = pick_student_col(df)
    if id_col is None:
        # create a dummy student so prediction grouping logic is uniform
        id_col = "_dummy_student"
        df[id_col] = "ALL"

    print(f"[LOAD] rows={len(df):,} | skills={df['skill_name'].nunique():,} | id_col={id_col}")

    # --- sequences per skill (with per-group time sort) ---
    skill_seqs = build_sequences(df)

    # --- compile stan ---
    print("[STAN] compiling model ...")
    model = CmdStanModel(stan_file=STAN_FILE)
    print("[STAN] compiled.")

    rows = []
    logs = []
    skill_params = {}  # map: skill -> dict(prior, learn, guess, slip)

    # --- fit per skill (MAP optimization for speed) ---
    for i, (skill, seq_list) in enumerate(skill_seqs.items(), 1):
        data = pack_sequences(seq_list)
        fit = model.optimize(data=data, algorithm="lbfgs")

        prior = float(fit.stan_variable("prior"))
        learn = float(fit.stan_variable("learn"))
        guess = float(fit.stan_variable("guess"))
        slip  = float(fit.stan_variable("slip"))

        skill_params[skill] = dict(prior=prior, learns=learn, guesses=guess, slips=slip)

        rows.extend([
            {"skill": skill, "param": "prior",   "class": "default", "value": prior},
            {"skill": skill, "param": "learns",  "class": "default", "value": learn},
            {"skill": skill, "param": "guesses", "class": "default", "value": guess},
            {"skill": skill, "param": "slips",   "class": "default", "value": slip},
            {"skill": skill, "param": "forgets", "class": "default", "value": 0.0},
        ])
        logs.append({
            "skill": skill,
            "n_sequences": data["K"],
            "N_total": data["N"],
            "prior": prior, "learn": learn,
            "guess": guess, "slip": slip,
            "g_plus_s": guess + slip
        })

        if i <= 8 or i % 25 == 0:
            print(f"[{i:>4}/{len(skill_seqs)}] {skill[:64]} | "
                  f"prior={prior:.3f} learn={learn:.3f} g={guess:.3f} s={slip:.3f} g+s={guess+slip:.3f}")

    # --- write params & logs ---
    params_df = pd.DataFrame(rows)
    params_df.to_csv(OUT_PARAMS, index=False)
    pd.DataFrame(logs).to_csv(OUT_LOG, index=False)

    # --- hard verification of bounds ---
    wide = params_df.pivot_table(index="skill", columns="param", values="value")
    bad = []
    for sk, r in wide.iterrows():
        p, l, g, s = float(r["prior"]), float(r["learns"]), float(r["guesses"]), float(r["slips"])
        if not (0.01 <= p <= 0.90): bad.append((sk,"prior",p))
        if not (0.01 <= l <= 0.90): bad.append((sk,"learn",l))
        if not (0.01 <= g <= 0.50): bad.append((sk,"guess",g))
        if not (0.01 <= s <= 0.50): bad.append((sk,"slip",s))
        if g + s > 1.0 + 1e-9:      bad.append((sk,"sum",g+s))
    print("[VERIFY] violations:", len(bad))
    if bad[:8]: print(" examples:", bad[:8])

    # --- PREDICTIONS over original rows ---
    # Compute pre-mastery, predicted p(correct), post-mastery for each row in time order
    print("[PRED] generating row-wise predictions and over-practice stats ...")
    if tcol:
        sort_cols = [tcol]
    else:
        # if no time, keep original row order within each group
        sort_cols = None

    preds = []
    over_count = 0
    total_rows = len(df)

    # group by skill then student to follow the same sequence structure
    for skill, dsk in df.groupby("skill_name", sort=False):
        if skill not in skill_params:
            continue  # should not happen, but guard
        p = skill_params[skill]
        prior, learn, guess, slip = p["prior"], p["learns"], p["guesses"], p["slips"]

        for sid, g in dsk.groupby(id_col, sort=False):
            if sort_cols:
                g = g.sort_values(by=sort_cols)

            L = prior  # mastery before first step of this (student,skill) sequence
            for idx, row in g.iterrows():
                pre_mastery = L
                p_correct = pre_mastery * (1.0 - slip) + (1.0 - pre_mastery) * guess
                y = int(row["correct"])

                # over-practice check (threshold 0.95 on pre-mastery)
                if pre_mastery >= 0.95:
                    over_count += 1

                # posterior given observation
                denom = p_correct if y == 1 else (1.0 - p_correct)
                # numerical safety
                denom = max(denom, 1e-12)
                if y == 1:
                    numer = pre_mastery * (1.0 - slip)
                else:
                    numer = pre_mastery * slip
                post = numer / denom

                # learning update (no forgetting)
                L = post + (1.0 - post) * learn

                preds.append({
                    "index": idx,
                    "student": sid,
                    "skill_name": skill,
                    "pre_mastery": pre_mastery,
                    "pred_p_correct": p_correct,
                    "post_mastery": L,
                    "guess": guess,
                    "slip": slip,
                    "prior": prior,
                    "learn": learn,
                    "observed_correct": y
                })

    preds_df = pd.DataFrame(preds).sort_values(by="index")
    preds_df.to_csv(OUT_PREDS, index=False)

    # --- Over-practice summary ---
    over_pct = (over_count / total_rows) * 100.0 if total_rows > 0 else 0.0
    print(f"[OVERPRACTICE] steps with pre_mastery ≥ 0.95: {over_count:,} / {total_rows:,} "
          f"({over_pct:.2f}%)")

    print("[DONE] wrote:")
    print("  -", OUT_PARAMS)
    print("  -", OUT_LOG)
    print("  -", OUT_PREDS)

if __name__ == "__main__":
    main()

