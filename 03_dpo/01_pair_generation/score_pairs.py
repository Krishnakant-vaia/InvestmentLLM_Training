"""Check and score DPO pairs. NOTHING IS DROPPED: every pair is kept, with the judge's scores, a list of flags (every problem the
checks found) and a `passes_all_gates` marker, so filtering is a later choice.

1. Build the judge input (code decides which answer is "a" and which is "b", the judge is not told):
       python score_pairs.py build pairs.jsonl --out judge_in.jsonl --key judge_key.json
   Run DPO_pair_judge_prompt.txt over judge_in.jsonl (judge.py) and save the "scores" objects, one per line, to judge_out.jsonl.
2. Attach verdicts, flags and the score to every pair:
       python score_pairs.py select pairs.jsonl --judge judge_out.jsonl --key judge_key.json --out scored_pairs.jsonl

Input pairs: JSONL in the qa_prompt.txt output schema, plus `document` (the chunk text; the judge needs it).

Flags (all informational):
  * code checks: <think> wrapping, thinking and answer lengths, defect allowed for the type, tells, numeric_check recomputes;
  * judge on the preferred answer: score below 8, or any failed gate;
  * judge on the not-preferred answer: it passes (the defect did not land), or fails gates beyond the defect's own (G8 ignored).
A pair with no flags has passes_all_gates = true. Each pair also carries the judge's own verdicts: `judge_preferred` and
`judge_not_preferred`, each {score, reason, failed_gates}. There is no combined score and no ranking: filter on those fields.
"""
import argparse
import ast
import json
import operator
import re
import random
from collections import Counter, defaultdict

import config
from schema import normalize_row

# Gates each defect may fail. The judge fails G2 for ANY claim that contradicts or goes beyond the document (that is its definition of a
# contradicted claim), so every defect that puts a wrong claim into the answer is allowed G2 as well as its own gate; measured on the
# smoke test, where such rejected answers failed G2 whatever their defect. G8 is excluded: it fails whenever any other gate fails.
EXPECTED_GATES = {
    "headline_number": {"G2", "G3", "G4"}, "wrong_adjustment": {"G2", "G3"}, "calculation_slip": {"G2", "G3"},
    "no_workings": {"G2", "G3", "G5"}, "off_by_one": {"G2", "G3", "G4"}, "omitted_element": {"G5"},
    "advice_not_draft": {"G5", "G7"}, "generic_content": {"G2", "G5"}, "unsupported_claim": {"G2"},
    "wrong_definition": {"G2", "G4"}, "single_driver": {"G5"}, "overstated_certainty": {"G2", "G4", "G7"},
    "padded": {"G7"},
    # management-presentation set (DPO_pair_judge_prompt_deck.txt): G4 = stage and narrative order, G7 = slide type fit and density
    "wrong_order": {"G4"}, "wrong_slide_type": {"G7"}, "overstuffed_slide": {"G7"},
    # CIM set (DPO_pair_judge_prompt_cim.txt): a statement that contradicts another section fails G6 (consistency); G2 when it also contradicts a stated fact
    "cross_section_conflict": {"G6", "G2"},
}
# Gates each defect fails under DPO_pair_judge_prompt_cim.txt, which assigns every error class to one gate (G2 grounding, G3 numbers,
# G4 basis and certainty, G5 completeness, G6 consistency, G7 trainability). Used for rows whose folder is config.CIM_FOLDER.
EXPECTED_GATES_CIM = {
    "omitted_element": {"G5"}, "advice_not_draft": {"G5", "G7"}, "generic_content": {"G5", "G2"}, "unsupported_claim": {"G2"},
    "overstated_certainty": {"G4"}, "headline_number": {"G3", "G4"}, "wrong_adjustment": {"G3", "G4"}, "wrong_definition": {"G4", "G3"},
    "calculation_slip": {"G3"}, "off_by_one": {"G3", "G4"}, "no_workings": {"G3", "G7"}, "padded": {"G7"}, "cross_section_conflict": {"G6"},
}
# Which defects each question type may be paired with. The generation script draws from this list, so an
# impossible pairing (e.g. padded + Calculation) is never sent; code_flags flags any that slip through.
COMPATIBLE = {
    "Calculation/Numerical": ["calculation_slip", "no_workings", "off_by_one", "unsupported_claim"],
    "Adjusted Metric": ["headline_number", "wrong_adjustment", "calculation_slip", "no_workings", "wrong_definition"],
    "Rating/Notching": ["off_by_one", "calculation_slip", "no_workings", "wrong_definition"],
    "Waterfall/Distribution": ["calculation_slip", "no_workings", "off_by_one", "omitted_element", "wrong_definition"],
    "Drafting": ["omitted_element", "advice_not_draft", "generic_content", "unsupported_claim", "overstated_certainty",
                 "headline_number", "wrong_adjustment", "calculation_slip", "wrong_definition"],   # the last four: sections that carry the file's own figures and definitions
    # CIM set (DPO_content_generation_prompts_cim.txt): Drafting plus cross_section_conflict, kept apart so the other prompts never get a defect they do not describe
    "CIM Drafting": ["omitted_element", "advice_not_draft", "generic_content", "unsupported_claim", "overstated_certainty",
                     "headline_number", "wrong_adjustment", "calculation_slip", "wrong_definition", "cross_section_conflict"],
    "High-Order Reasoning": ["single_driver", "unsupported_claim", "overstated_certainty", "omitted_element"],
    "Error Identification": ["omitted_element", "unsupported_claim", "overstated_certainty", "generic_content"],
    "Applied/Scenario": ["omitted_element", "generic_content", "overstated_certainty", "unsupported_claim"],
    "Interpretation & Analysis": ["unsupported_claim", "overstated_certainty", "single_driver", "generic_content"],
    "Comparative Ranking": ["off_by_one", "omitted_element", "single_driver", "unsupported_claim"],
    "Terminology": ["wrong_definition", "generic_content", "omitted_element", "unsupported_claim"],
    "Short Answer": ["unsupported_claim", "wrong_definition", "off_by_one", "padded"],
    "Slide Drafting": ["omitted_element", "wrong_slide_type", "overstuffed_slide", "advice_not_draft", "generic_content",
                       "unsupported_claim", "off_by_one", "overstated_certainty"],
    "Deck Outline": ["omitted_element", "wrong_order", "advice_not_draft", "generic_content", "unsupported_claim", "off_by_one"],
}
ONE_PAIR_PER_CALL = {"Slide Drafting", "Deck Outline", "Drafting", "CIM Drafting", "Calculation/Numerical", "Adjusted Metric", "Rating/Notching", "Waterfall/Distribution"}
MAX_PAIRS_PER_CALL = 3                # every other type; the long ones above get a call each (output size, JSON breakage)
THINK_WORDS = (20, 350)               # the prompt asks for 70-150 (up to 250 for numerical types); gpt-oss-120b writes ~60 and ignores word counts
THINK_RATIO = (0.4, 2.5)              # rejected_thinking words / chosen_thinking words: a far shorter rejected trace is a surface tell
MIN_CHOSEN_WORDS = 10


def allowed_defects(question_type: str) -> list[str]:
    """The defects the generation script may assign to this question type."""
    return COMPATIBLE[question_type]


LENGTH_RULE = {"no_workings": (0.5, 0.8), "omitted_element": (0.5, 0.8), "single_driver": (0.5, 0.8), "padded": (1.3, 1.8)}
DEFAULT_LENGTH = (0.8, 1.25)
LENGTH_SLACK = 0.15                  # the generator counts loosely; the trainer's own limit is 2x
NUMERIC_TYPES = {"Calculation/Numerical", "Adjusted Metric"}     # single final figure; waterfalls and ratings are not checked here
NUMERIC_TOLERANCE = 0.01
CHOSEN_MIN_SCORE = 8
TELLS = ("as an ai", "i cannot", "i can't", "constraint", "from memory", "hypothetical", "illustrative")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def quote_in_document(quote, doc: str, folder: str | None = None) -> bool:
    """`doc` is whitespace-normalised. A quote is verbatim, except in a tear sheet, where a table's labels and values sit in different
    rows (header row, then value row): a quote like "Our investment | $897,000" is then accepted when each of its pipe-separated parts
    is in the sheet. A part that is not there (a changed or invented figure) still fails."""
    q = " ".join(str(quote).split())
    if q in doc:
        return True
    if folder != config.TEARSHEET_FOLDER:
        return False
    plain = " ".join(doc.replace("*", " ").split())
    nl = chr(10)
    parts = [" ".join(x.replace("*", " ").split()) for x in re.split(r"[|" + nl + "]", str(quote).strip("| " + nl))]
    return all(x and x in plain for x in parts)


TRIVIAL_LITERALS = {0, 1, 2, 3, 4, 5, 10, 100, 1000}


def figure_flags(row: dict) -> list[str]:
    """CIM pairs (rows with `facts`, the numbers of the CIM's structured JSON): every number the preferred answer's workings use must be a
    figure of the document or of the structured data, within the rounding of the number as written. Pure arithmetic constants are free."""
    facts = row.get("facts")
    if not facts:
        return []
    check = row.get("numeric_check") or {}
    doc_numbers = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", row.get("document", ""))]
    flags = []
    for lit in re.findall(r"\d+(?:\.\d+)?", str(check.get("expression", ""))):
        v = float(lit)
        if v in TRIVIAL_LITERALS:
            continue
        tol = 0.5 * 10 ** -(len(lit.split(".")[1]) if "." in lit else 0) + 1e-9
        if not any(abs(v - f) <= tol for f in doc_numbers) and not any(abs(v - f) <= tol for f in facts):
            flags.append(f"numeric: input {lit} is in neither the document nor the structured data")
            break
    return flags


def final_matches(value: float, final: float, unit=None) -> bool:
    """The expression's value equals the stated final figure, within the tolerance or within the rounding of the figure as written
    (1.235 stated as 1.2); a percentage may be written as a ratio in the expression (0.195 for 19.5 %)."""
    decimals = len(repr(float(final)).split(".")[1].rstrip("0")) if "." in repr(float(final)) else 0
    slack = max(NUMERIC_TOLERANCE * max(abs(final), 1e-9), 0.5 * 10 ** -decimals + 1e-9)
    candidates = [value] + ([value * 100] if unit and "%" in str(unit) else [])
    return any(abs(v - final) <= slack for v in candidates)


def evaluate(expression: str) -> float:
    """Value of an arithmetic expression (numbers and + - * / ( ) ** only); anything else raises."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.operand))
        raise ValueError("not plain arithmetic")
    return walk(ast.parse(expression, mode="eval").body)


def strip_think(text: str) -> str:
    """The final answer only: the judge never sees the reasoning trace."""
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text.strip()


# ---- step 0: code checks (flags only) -------------------------------------------------------

TEXT_FIELDS = ("document", "question", "preferred", "not_preferred", "preferred_thinking", "not_preferred_thinking", "defect_type", "question_type")


def has_text(row: dict) -> bool:
    return all(isinstance(row.get(k), str) and row[k].strip() for k in TEXT_FIELDS)


def code_flags(row: dict) -> list[str]:
    """Every problem the code checks find in a pair; an empty list means clean. Nothing is dropped because of them."""
    if not has_text(row):
        return ["missing field"]
    flags = []
    if row["defect_type"] not in EXPECTED_GATES:
        flags.append("unknown defect_type")
    elif row["defect_type"] not in COMPATIBLE.get(row["question_type"], []):
        flags.append("defect_type not allowed for this question_type")
    if row["preferred"].strip() == row["not_preferred"].strip():
        flags.append("preferred identical to not-preferred")
    if not all(row[k].strip().startswith("<think>") and row[k].strip().endswith("</think>")
               for k in ("preferred_thinking", "not_preferred_thinking")):
        flags.append("thinking not wrapped in <think>")
    think = [len(row[k].split()) for k in ("preferred_thinking", "not_preferred_thinking")]
    if not all(THINK_WORDS[0] <= n <= THINK_WORDS[1] for n in think):
        flags.append("thinking length outside the SFT-style range")
    if think[0] and not THINK_RATIO[0] <= think[1] / think[0] <= THINK_RATIO[1]:
        flags.append("not-preferred thinking much shorter or longer than preferred")
    if len(row["preferred"].split()) < MIN_CHOSEN_WORDS:
        flags.append("preferred answer too short to teach anything")
    low, high = LENGTH_RULE.get(row["defect_type"], DEFAULT_LENGTH)
    ratio = len(row["not_preferred"].split()) / max(1, len(row["preferred"].split()))
    if not low - LENGTH_SLACK <= ratio <= high + LENGTH_SLACK:
        flags.append("length outside the defect's rule")
    blob = " ".join(row[k].lower() for k in ("not_preferred", "not_preferred_thinking"))
    if any(t in blob for t in TELLS):
        flags.append("not-preferred answer has a tell")
    if "evidence" in row:                                  # content prompt only: each quote must be a verbatim part of the document
        doc = " ".join(row["document"].split())
        quotes = row["evidence"] if isinstance(row["evidence"], list) else []
        if not quotes:
            flags.append("evidence missing")
        elif any(not quote_in_document(q, doc, row.get("folder")) for q in quotes):
            flags.append("evidence quote not found in the document")
    if row["question_type"] in NUMERIC_TYPES:
        check = row.get("numeric_check") or {}
        final, rejected = check.get("final_value"), check.get("not_preferred_value", check.get("rejected_value"))
        if not isinstance(final, (int, float)) or isinstance(final, bool):
            flags.append("numeric: final_value is not a number")
        else:
            if final == rejected:
                flags.append("numeric: not-preferred equals preferred")
            try:
                value = evaluate(str(check.get("expression", "")))
                if not final_matches(value, final, check.get("unit")):
                    flags.append("numeric: expression does not equal the final value")
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                flags.append("numeric: expression does not evaluate")
        flags += figure_flags(row)
    return flags


# ---- step 1: judge input -------------------------------------------------------------------

def build_judge_input(rows: list[dict], seed: int = 3407):
    """(judge lines, key). `rows` are the pairs that have all their text fields; the key says, per line, which side holds the
    chosen answer."""
    rng, lines, key = random.Random(seed), [], []
    for i, r in enumerate(rows):
        chosen_first = rng.random() < 0.5
        a, b = (r["preferred"], r["not_preferred"]) if chosen_first else (r["not_preferred"], r["preferred"])
        lines.append({"id": i, "document": r["document"], "question": r["question"],
                      "answer_a": strip_think(a), "answer_b": strip_think(b)})
        key.append({"preferred_side": "a" if chosen_first else "b"})
    return lines, key


# ---- step 2: judge flags and score ---------------------------------------------------------

def judge_flags(row: dict, chosen: dict, rejected: dict) -> list[str]:
    """Problems the judge's verdicts show. Each verdict: {score, reason, failed_gates}."""
    flags = []
    if chosen.get("score", 0) < CHOSEN_MIN_SCORE or chosen.get("reason") != "PASS_ALL_GATES" or chosen.get("failed_gates"):
        gates = sorted(set(chosen.get("failed_gates") or []) - {"G8"})
        flags.append(f"judge: preferred answer fails ({chosen.get('reason')}; gates {gates})")
    failed = set(rejected.get("failed_gates") or []) - {"G8"}
    table = EXPECTED_GATES_CIM if row.get("folder") == config.CIM_FOLDER else EXPECTED_GATES
    allowed = table.get(row.get("defect_type"), set())
    if rejected.get("score", 10) >= 7 or not failed:
        flags.append("judge: not-preferred answer passes (defect did not land)")
    elif failed - allowed:
        flags.append(f"judge: not-preferred answer also fails {sorted(failed - allowed)}")
    return flags


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cmd_build(a):
    rows = [normalize_row(r) for r in read_jsonl(a.pairs)]          # old files (chosen / rejected) are read as preferred / not_preferred
    judgeable = [i for i, r in enumerate(rows) if has_text(r)]
    lines, key = build_judge_input([rows[i] for i in judgeable])
    for k, i in zip(key, judgeable):
        k["row"] = i                                           # which pair of the full list this judge line is
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    with open(a.key, "w", encoding="utf-8") as f:
        json.dump({"pairs": rows, "key": key}, f, ensure_ascii=False)
    flagged = Counter(flag for r in rows for flag in code_flags(r))
    print(f"read {len(rows)} pairs, all kept; {len(lines)} go to the judge ({len(rows) - len(lines)} lack a text field) -> {a.out}")
    for why, n in flagged.most_common():
        print(f"    flagged {n:>5}  {why}")


def cmd_select(a):
    saved = json.load(open(a.key, encoding="utf-8"))
    rows, key = saved["pairs"], saved["key"]
    verdicts = {v["id"]: v for v in read_jsonl(a.judge)}
    for j, k in enumerate(key):
        v = verdicts.get(j)
        if v:
            side = k["preferred_side"]
            rows[k["row"]]["judge_preferred"], rows[k["row"]]["judge_not_preferred"] = v[side], v["b" if side == "a" else "a"]
    for r in rows:
        judged = judge_flags(r, r["judge_preferred"], r["judge_not_preferred"]) if "judge_preferred" in r else ["no judge verdict"]
        r["flags"] = code_flags(r) + judged
        r["passes_all_gates"] = not r["flags"]
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    clean = sum(r["passes_all_gates"] for r in rows)
    print(f"{len(rows)} pairs, ALL kept -> {a.out}; {clean} pass every gate, {len(rows) - clean} carry flags")
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r.get("question_type")][0] += 1
        by_type[r.get("question_type")][1] += r["passes_all_gates"]
    for t, (n, ok) in sorted(by_type.items(), key=lambda x: str(x[0])):
        print(f"    {str(t):28} pairs {n:>5}  clean {ok:>5}")
    for why, n in Counter(f.split(" (")[0].split(" also fails")[0] for r in rows for f in r["flags"]).most_common():
        print(f"    flagged {n:>5}  {why}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the judge input (every pair is kept)")
    b.add_argument("pairs")
    b.add_argument("--out", required=True)
    b.add_argument("--key", required=True)
    b.set_defaults(fn=cmd_build)
    s = sub.add_parser("select", help="attach the judge's verdicts, flags and score to every pair")
    s.add_argument("pairs")
    s.add_argument("--judge", required=True)
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_select)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
