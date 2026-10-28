"""Can a text-only model tell the preferred answer from the not-preferred one without reading the question? If it can, DPO will learn
that surface tell (a phrase, a layout, a length) instead of the mistake.

    python pairs/audit.py --data data/raw/dpo_pairs.jsonl [--max-pairs 6000]

    audit(rows) -> (report lines, accuracy per domain, top tell words)

A logistic regression on word and character features of the ANSWER only, scored by 5-fold cross-validation (the same question's two
answers always fall in the same fold). 50% = no tell. Above about 65% = a tell worth removing before training. CPU only, a minute.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step_root  # noqa: E402,F401  (shared sft_common + this step as config root)

TELL_ACCURACY = 0.65


def _answer(text: str) -> str:
    return text.split("</think>")[-1].strip()


def audit(rows: list[dict], max_pairs: int = 6000, seed: int = 0) -> tuple[list[str], dict, list[tuple[str, float]]]:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from scipy.sparse import hstack

    from pairs.shape import domain_of
    rng = random.Random(seed)
    rows = rng.sample(rows, max_pairs) if len(rows) > max_pairs else rows
    texts, labels, groups, domains = [], [], [], []
    for i, r in enumerate(rows):
        for text, y in ((_answer(r["preferred"]), 1), (_answer(r["not_preferred"]), 0)):
            texts.append(text)
            labels.append(y)
            groups.append(i)
            domains.append(domain_of(r))
    y, groups, domains = np.array(labels), np.array(groups), np.array(domains)
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, lowercase=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, sublinear_tf=True, max_features=60000)
    X = hstack([words.fit_transform(texts), chars.fit_transform(texts)]).tocsr()      # vocabulary fitted on all: no labels used
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pred[te] = LogisticRegression(max_iter=300, C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    # pairwise: does the classifier rank the preferred answer above the not-preferred one of the SAME pair?
    n = len(rows)
    p_c, p_r = pred[0::2], pred[1::2]                                              # texts were appended chosen, rejected, chosen, ...
    len_c = np.array([len(_answer(r["preferred"]).split()) for r in rows])
    len_r = np.array([len(_answer(r["not_preferred"]).split()) for r in rows])
    dom = domains[0::2]
    matched = np.abs(np.log((len_c + 1) / (len_r + 1))) < 0.2                      # answers within ~20% of each other in length
    win = p_c > p_r
    acc = {"all": float(win.mean()), "length_only": float((len_c > len_r).mean()), "matched_share": float(matched.mean()),
           "matched": float(win[matched].mean()) if matched.any() else float("nan")}
    per = {d: (float(win[dom == d].mean()), float((len_c > len_r)[dom == d].mean()), float(matched[dom == d].mean())) for d in sorted(set(dom))}
    top = LogisticRegression(max_iter=300).fit(words.transform(texts), y)
    names = np.array(words.get_feature_names_out())
    order = np.argsort(top.coef_[0])
    tells = [(names[i], float(top.coef_[0][i])) for i in list(order[:8]) + list(order[-8:])]
    lines = [f"[tells audit] {n} pairs, answer text only (the question is never shown). The classifier ranks the preferred answer above the "
             f"not-preferred one of the same pair in {acc['all']:.1%} of pairs (50% = no tell).",
             f"    just picking the longer answer would win {acc['length_only']:.1%}; on the {acc['matched_share']:.0%} of pairs whose answers are within "
             f"~20% in length, the classifier still wins {acc['matched']:.1%}  <- the tells that are NOT length",
             f"    {'domain':24} {'classifier':>10} {'longer wins':>12} {'length-matched share':>21}"]
    for d, (a, ln, ms) in per.items():
        lines.append(f"    {d:24} {a:>10.1%} {ln:>12.1%} {ms:>21.0%}" + ("   <-- tell beyond length" if a - ln > 0.10 and a > TELL_ACCURACY else ""))
    lines.append("    strongest words (negative = marks the not-preferred answer): " + ", ".join(f"{w}({c:+.1f})" for w, c in tells))
    acc["by_domain"] = per
    return lines, acc, tells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-pairs", type=int, default=6000)
    ap.add_argument("--raw", action="store_true", help="audit the file as it is, without the cleaning the trainer applies")
    a = ap.parse_args()
    from pairs.shape import shape
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    if not a.raw:
        rows, _ = shape(rows, {"clean_remarks": True, "split_by": "template"})        # what the trainer would see
    print("\n".join(audit(rows, a.max_pairs)[0]))


if __name__ == "__main__":
    main()
