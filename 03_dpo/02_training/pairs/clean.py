"""Remove commentary from answers, the SAME way on both sides.

    clean_answer(text) -> text

The Excel pairs' not-preferred answers often explain their own mistake ("(cure period mis-converted)", "; pool ignored"), and the
preferred answers carry bracketed remarks of their own ("(disallowed valuation allowance release)"). If only one side is cleaned, the
other side's brackets become the tell instead (this happened once: brackets stayed on 55% of preferred answers and 0% of not-preferred
ones). So every remark of two or more words in brackets, and every trailing "; ... ignored." note, goes from BOTH sides.
What stays: one-word brackets ("(Tue)", "(6)"), brackets with no words ("(3,4,5)"), and the "Given:" line of worked solutions.
`think` blocks are cleaned too (their text is part of the answer the model would learn).
"""
import re

ERROR_WORDS = ("inverted", "omitted", "ignored", "reversed", "unweighted", "misapplied", "flipped", "skipped", "swapped", "double-counted",
               "mis-converted", "miscounted", "misread", "wrongly", "incorrectly")
_PAREN = re.compile(r"[ \t]*\(([^()]*)\)")
_TRAILING = re.compile(r";[ \t]*[\w ,/-]{0,40}\b(?:" + "|".join(ERROR_WORDS) + r")\b\.?")


def _is_remark(inner: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", inner)
    return len(words) >= 2 or (len(words) == 1 and words[0].lower() in ERROR_WORDS)


def clean_answer(text: str) -> str:
    out = []
    for line in str(text).split("\n"):
        if line.startswith("Given:"):
            out.append(line)
            continue
        line = _PAREN.sub(lambda m: "" if _is_remark(m.group(1)) else m.group(0), line)
        out.append(_TRAILING.sub("", line).rstrip())
    return "\n".join(out)
