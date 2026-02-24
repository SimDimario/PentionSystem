"""
Script di validazione del preprocessing interferenti.
Confronta le predizioni con e senza preprocessing su tutto il dataset,
classe per classe, e produce un report con i risultati.

Uso:
    python validate_preprocessing.py

Requisiti:
    pip install pandas requests tqdm
"""

import pandas as pd
import numpy as np
import requests
import json
from collections import defaultdict

# ── Configurazione ──────────────────────────────────────────────────────────
API_URL   = "http://localhost:8000"
CSV_PATH  = "C:/Users/simon/PycharmProjects/Pention-System/ClassificatoreNPS/datasetNPS/1-s2.0-S2468170923000358-mmc1.csv"  # metti il path corretto
BATCH_SIZE = 20   # spettri per chiamata API
# ────────────────────────────────────────────────────────────────────────────

legends = {
    0: 'Cathinone analogues',
    1: 'Cannabinoid analogues',
    2: 'Phenethylamine analogues',
    3: 'Piperazine analogues',
    4: 'Tryptamine analogues',
    5: 'Fentanyl analogues',
    6: 'Other compounds'
}


def predict_batch(spectra_list: list, endpoint: str, preprocessing: bool) -> list:
    """Chiama l'API e ritorna le predizioni."""
    payload = {
        "spectra": spectra_list,
        "apply_preprocessing": preprocessing
    }
    r = requests.post(f"{API_URL}/{endpoint}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["predictions"]


def run_validation(endpoint: str = "predict_dnn"):
    print(f"\n{'='*60}")
    print(f"  VALIDAZIONE PREPROCESSING — endpoint: /{endpoint}")
    print(f"{'='*60}\n")

    # Carica dataset
    df = pd.read_csv(CSV_PATH)
    spectra = df.iloc[:, 1:601].values.tolist()   # colonne m/z 1-600
    labels  = df['label'].values
    names   = df['Name'].values

    n = len(spectra)
    print(f"Dataset caricato: {n} spettri\n")

    # Raccogli predizioni con e senza preprocessing in batch
    preds_with    = []
    preds_without = []

    for start in range(0, n, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, n)
        batch = spectra[start:end]
        print(f"  Elaboro spettri {start+1}-{end}/{n}...", end="\r")

        preds_with    += predict_batch(batch, endpoint, preprocessing=True)
        preds_without += predict_batch(batch, endpoint, preprocessing=False)

    print(f"\n  Predizioni completate.\n")

    # ── Analisi risultati ──
    correct_with    = 0
    correct_without = 0
    changed         = 0   # casi in cui il preprocessing ha cambiato la predizione
    improved        = 0   # cambiati E corretti con preprocessing
    worsened        = 0   # cambiati E sbagliati con preprocessing

    # Per classe
    class_stats = defaultdict(lambda: {
        'total': 0,
        'correct_with': 0,
        'correct_without': 0,
        'changed': 0
    })

    disagreements = []   # casi interessanti da analizzare

    for i in range(n):
        true_label  = legends[labels[i]]
        pred_with   = preds_with[i]
        pred_without = preds_without[i]

        ok_with    = (pred_with    == true_label)
        ok_without = (pred_without == true_label)

        if ok_with:    correct_with    += 1
        if ok_without: correct_without += 1

        class_stats[true_label]['total']           += 1
        class_stats[true_label]['correct_with']    += int(ok_with)
        class_stats[true_label]['correct_without'] += int(ok_without)

        if pred_with != pred_without:
            changed += 1
            class_stats[true_label]['changed'] += 1

            if ok_with and not ok_without:
                improved += 1
                tag = "✅ MIGLIORATO"
            elif not ok_with and ok_without:
                worsened += 1
                tag = "❌ PEGGIORATO"
            else:
                tag = "↔ CAMBIATO (entrambi sbagliati)"

            disagreements.append({
                'name':          names[i],
                'true':          true_label,
                'pred_with':     pred_with,
                'pred_without':  pred_without,
                'tag':           tag
            })

    # ── Stampa risultati globali ──
    acc_with    = correct_with    / n * 100
    acc_without = correct_without / n * 100

    print("─" * 60)
    print(f"  ACCURACY SENZA preprocessing : {acc_without:.2f}%  ({correct_without}/{n})")
    print(f"  ACCURACY CON    preprocessing : {acc_with:.2f}%  ({correct_with}/{n})")
    delta = acc_with - acc_without
    sign  = "+" if delta >= 0 else ""
    print(f"  DELTA                         : {sign}{delta:.2f}%")
    print("─" * 60)
    print(f"\n  Spettri dove la predizione è cambiata : {changed}/{n}")
    print(f"    → Migliorati dal preprocessing       : {improved}")
    print(f"    → Peggiorati  dal preprocessing      : {worsened}")

    # ── Stampa risultati per classe ──
    print(f"\n{'─'*60}")
    print(f"  DETTAGLIO PER CLASSE")
    print(f"{'─'*60}")
    print(f"  {'Classe':<28} {'Tot':>4}  {'Senza':>6}  {'Con':>6}  {'Δ':>5}  {'Cambiati':>8}")
    print(f"  {'-'*28} {'-'*4}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*8}")

    for label_id in range(7):
        cls   = legends[label_id]
        stats = class_stats[cls]
        tot   = stats['total']
        if tot == 0:
            continue
        a_wo  = stats['correct_without'] / tot * 100
        a_w   = stats['correct_with']    / tot * 100
        d     = a_w - a_wo
        sign  = "+" if d >= 0 else ""
        print(f"  {cls:<28} {tot:>4}  {a_wo:>5.1f}%  {a_w:>5.1f}%  {sign}{d:>4.1f}%  {stats['changed']:>8}")

    # ── Stampa casi di disaccordo ──
    if disagreements:
        print(f"\n{'─'*60}")
        print(f"  CASI IN CUI IL PREPROCESSING HA CAMBIATO LA PREDIZIONE ({len(disagreements)})")
        print(f"{'─'*60}")
        for d in disagreements:
            print(f"\n  {d['tag']}")
            print(f"    Sostanza   : {d['name']}")
            print(f"    Vera classe: {d['true']}")
            print(f"    Senza prep : {d['pred_without']}")
            print(f"    Con prep   : {d['pred_with']}")
    else:
        print("\n  ℹ️  Il preprocessing non ha cambiato nessuna predizione su questo dataset.")
        print("     Questo può indicare che:")
        print("     1. Il modello è già robusto agli interferenti presenti in questo dataset")
        print("     2. Gli spettri del dataset sono già relativamente puliti")
        print("     3. Il preprocessing agisce ma non cambia la classe predetta (margini di confidenza ampi)")

    print(f"\n{'='*60}\n")
    return {
        'acc_with': acc_with,
        'acc_without': acc_without,
        'delta': delta,
        'changed': changed,
        'improved': improved,
        'worsened': worsened,
        'disagreements': disagreements
    }


if __name__ == "__main__":
    # Testa entrambi gli endpoint
    results_dnn = run_validation("predict_dnn")
    results_brf = run_validation("predict_brf")