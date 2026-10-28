"""One vocabulary for every QA pair: preferred / not_preferred.

Old files used chosen / rejected (document runs, the training file) or accepted / rejected (the numerical Excel sets). Every pair now carries

    preferred, not_preferred                       the two answers
    preferred_thinking, not_preferred_thinking     their reasoning (document pairs)
    judge_preferred, judge_not_preferred           the judge's verdict on each
    preferred_side                                 which side (a or b) the judge saw the preferred answer on

    normalize_row(row, accepted_is_answer)         any old or new row -> the new names (idempotent)
    to_old_names(row, kind)                        the inverse, used only to prove a migration lost nothing

`accepted` is an answer only in the numerical pair files. In rewrite results it is a yes/no outcome, so it is renamed only when
`accepted_is_answer` is True. Flag texts that used the old words are renamed with the same mapping.
"""
KEY_RENAMES = {
    "chosen": "preferred", "rejected": "not_preferred",
    "chosen_thinking": "preferred_thinking", "rejected_thinking": "not_preferred_thinking",
    "judge_chosen": "judge_preferred", "judge_rejected": "judge_not_preferred",
    "chosen_judge": "judge_preferred", "rejected_judge": "judge_not_preferred",
    "chosen_side": "preferred_side",
}
ACCEPTED_RENAME = {"accepted": "preferred"}

# Flag texts (exact phrases; "judge: chosen answer fails (...)" keeps its tail).
FLAG_RENAMES = [
    ("chosen identical to rejected", "preferred identical to not-preferred"),
    ("chosen answer too short to teach anything", "preferred answer too short to teach anything"),
    ("rejected thinking much shorter or longer than chosen", "not-preferred thinking much shorter or longer than preferred"),
    ("numeric: rejected equals chosen", "numeric: not-preferred equals preferred"),
    ("judge: chosen answer fails", "judge: preferred answer fails"),
]

# What each kind of file used before, for the inverse (verification only).
OLD_NAMES_BY_KIND = {
    "training": {"preferred": "chosen", "not_preferred": "rejected", "judge_preferred": "chosen_judge", "judge_not_preferred": "rejected_judge"},
    "scored": {"preferred": "chosen", "not_preferred": "rejected", "preferred_thinking": "chosen_thinking",
               "not_preferred_thinking": "rejected_thinking", "judge_preferred": "judge_chosen", "judge_not_preferred": "judge_rejected"},
    "raw": {"preferred": "chosen", "not_preferred": "rejected", "preferred_thinking": "chosen_thinking", "not_preferred_thinking": "rejected_thinking"},
    "numerical": {"preferred": "accepted", "not_preferred": "rejected"},
}


def rename_flag(text: str) -> str:
    for old, new in FLAG_RENAMES:
        if text.startswith(old):
            return new + text[len(old):]
    return text


def _flag_to_old(text: str) -> str:
    for old, new in FLAG_RENAMES:
        if text.startswith(new):
            return old + text[len(new):]
    return text


def normalize_row(row: dict, accepted_is_answer: bool = False) -> dict:
    """The row with the new names, in the same key order. Rows already in the new names come back unchanged."""
    renames = {**KEY_RENAMES, **(ACCEPTED_RENAME if accepted_is_answer else {})}
    out = {}
    for k, v in row.items():
        new = renames.get(k, k)
        if new in out or (new != k and new in row):          # never overwrite a value that is already there
            raise ValueError(f"row has both {k!r} and {new!r}")
        out[new] = [rename_flag(f) for f in v] if k == "flags" and isinstance(v, list) else v
    return out


def to_old_names(row: dict, kind: str) -> dict:
    """Invert a migration for one kind of file (training, scored, raw, numerical)."""
    inv = OLD_NAMES_BY_KIND[kind]
    flags_had_old_words = kind != "numerical"          # the numerical files already used "judge: preferred answer fails": nothing to restore
    return {inv.get(k, k): ([_flag_to_old(f) for f in v] if flags_had_old_words and k == "flags" and isinstance(v, list) else v)
            for k, v in row.items()}
