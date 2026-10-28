IC_MEMO_GENERATION_PROMPT = """\
ROLE
You are a senior PE deal-team principal generating preference data for an investment LLM.

DOCUMENT
<document>
{chunk_text}
</document>

The document (possibly DOC-1, DOC-2) and the formulas below are the only evidence. No outside facts or assumptions. Keep source units.

ASSIGNMENT
- Sections, in order: {question_types}
- Defects, in order: {defect_types}
- Answer layout: {style}

Pair the i-th section with the i-th defect; max 4 pairs. Skip unsupported pairs; return [] if nothing is decision-useful.

PROCEDURE
1. Find the entity, periods, figures, adjustments, risks and the document's own metric definitions.
2. Pick 2-4 supported sub-headings, including one the defect edits.
3. Write the question and chosen answer, then copy it with only the defect applied.

SECTION TEMPLATES (CAPITAL heading, one bottom-line sentence, bold sub-headings verbatim and in order, listed columns)
F1 Post=Pre+Round; PPS=Pre/pre-round FD shares; FD%=Round/Post
F2 CAGR=(End/Start)^(1/n)-1, never YoY over a missing year; haircut=1-Base/Target
F3 Weighted mid=sum(w*mid), weights sum 100%; multiple=Pre/Revenue (LTM or forward)
F4 MOIC=Exit EV*exit ownership/Investment; IRR=MOIC^(1/n)-1; Expected=sum(p*value); breakeven EV=Investment/exit ownership
1 Executive_Summary: Investment Ask (F1); Return Profile (Scenario|Probability|MOIC|IRR, Expected); Top Risks; Recommendation and Conditions
2 Company_and_Traction: Revenue Trajectory (Period|Revenue|Growth|Status; F2); Capital Efficiency (source definition); Governance Gaps
3 Financial_Projections: Scenario Projections (Metric|Bull|Base|Bear); Earnings Adjustments (reported, each add-back accepted/rejected with source reason, final); Base Case Haircut (F2); Funding Need
4 Valuation_and_Pricing: Weighted Valuation (Method|Basis|Low|Mid|High|Weight; F3); Implied Multiples; Comps Adjustments (sum vs stated net)
5 Deal_Terms_and_Capitalisation: Round Economics (F1); Post-Round Cap Table (Holder|Pre FD%|Post FD%|Change pp); Control Thresholds (as stated)
6 Returns_and_Downside: Scenario Returns (Scenario|Probability|Exit Year|Exit EV|MOIC|IRR; F4); Expected Value; Breakeven Multiple (vs lowest comp); Preference Floor
7 Key_Risks_and_Mitigants: Risk Register (Severity|Risk|Evidence|Impact|Mitigant; HIGH first); Severity Count (HIGH/total); Reference Calls (completed/total, refusals)

QUESTION RULES
- 2-5 sentences, 40-140 words, one entity, transaction and basis.
- Request 2-4 sub-headings in order, paraphrased, answerable from the document (3-4 for missing_subheading).
- Reveal no answer figures; restate at most two inputs; copy at most eight consecutive words.
- Start with a verb or question phrase, never "Based on"/"According to"; vary openings.
- Never ask for your own buy/sell/hold view ("Recommendation and Conditions" = the memo's documented one).
- Avoid conflicting figures unless the document says which governs; make the defect a plausible analyst mistake.

DEFECT RULES
Apply exactly the assigned defect inside a fitting sub-heading:
- wrong_headline_figure: swap a headline figure for another source figure.
- misapplied_adjustment: wrongly include or exclude one adjustment, scenario or weight.
- arithmetic_error: in one calculation line, one denominator, unit, sign, period or n error.
- missing_derivation: delete calculations, state the wrong result as fact; 50-80% length.
- adjacent_value_error: outside calculations, shift one count, period or n to its neighbour.
- mixed_basis: mix LTM/forward, basic/FD, reported/adjusted or pre/post.
- missing_subheading: delete 1-2 adverse or decision sub-headings, keep at least 2; 50-80%.
- advocacy_bias: delete or soften every adverse item; 70-100%.
- generic_prose: generic phrases replace specifics in over half the sentences.
- fabricated_fact: state one plausible fact absent from the document.
- target_as_base_case: present a target, interpolated or illustrative figure as base or actual.
- dropped_qualifier: strip every qualifier (unaudited, estimated, illustrative).
- conditions_dropped: drop all conditions or state approval as decided.
- non_source_definition: use a textbook definition, not the source's.
- advice_not_memo: replace the section with what to verify or negotiate, no template headings.

CHOSEN ANSWER
- Every claim traces to the document or rechecked arithmetic; full precision, round at the end; narrative and figures agree.
- Show "Label: formula = result"; if it differs from the source, report both.
- Label figures (actual, estimate, target, illustrative); prefer base case, target and haircut separate; keep bases distinct (pre/post, basic/FD, LTM/forward, reported/adjusted).
- Use the document's definitions and adjustments, not headline figures. Conflicts: governing figure, else flag both. Gaps: "Not disclosed in the source".
- Recommendation only as the documented view with all conditions; other judgments marked inference; name specific risks.
- 60-200 words per sub-heading, 250-800 per section; no preamble, filler or citations.

REJECTED ANSWER
Copy the chosen answer; change only the defect and dependent figures. Fluent, unflagged, detectable only against the document. Length within 25% of chosen unless stated.

THINKING TRACES
First-person <think>...</think> per answer, 80-180 words, within 25% of each other, ending in a qualified conclusion. The rejected trace treats its defect as correct.

NUMERIC CHECK
When the defect changes a figure: {{"expression":"numbers and + - * / ( ) ** only, giving the chosen figure","final_value":0,"unit":"","rejected_value":0}} (rejected_value differs). Otherwise {{}}.

OUTPUT
Return only a valid JSON array; each object has exactly: question_type, defect_type (assigned labels), question, chosen_thinking, chosen, rejected_thinking, rejected, numeric_check, evidence (3-6 verbatim document quotes, max 25 words each). Double quotes, escaped quotes, \\n in strings; no fences, trailing commas, nulls or extra keys.
"""
