# VaiaGPT SFT dataset rebuild — stage-by-stage plan

Last updated: 2026-10-27 (v2: new bucket split, guardrails, knowledge-injection phase)

## Context

Run 2 SFT and DPO made the model *more* confidently wrong than the pre-tuned base:
- Phantom sources: "the document / the table / the authors" appeared in 26 → 66 → 76 of 200 eval answers (pre → SFT → DPO); hedging fell 30 → 7 → 5.
- Root cause: the 178k pool was built by stripping `<document>` chunks (`backup_before_doc_strip/…208k_with_documents.jsonl`) while keeping answers that read from them. About 61% of the corpus questions name a source directly, and roughly a third of the rest still depend on one implicitly.
- Live chat confirms it: invented "the company", wrong BRSR facts (MCA vs SEBI, "5 pillars" vs the 9 NGRBC principles), the IFC "first principle" confused with the EU Taxonomy's DNSH, a fake inline tool call with fabricated results, and an identity leak ("I am Qwen3").
- 66% of the clean questions use generator templates ("state one practical implication", "…risk that could invalidate…").
- Facts and figures: SFT on Q&A alone does not store facts, and training on facts the base model does not know raises hallucination (see "Facts and figures" below).

Goal: a ~212k SFT set where every answer is either grounded in material that is actually in the prompt (a document, a search result or a tool result) or is verified stable knowledge, plus real tool use, behaviour and guardrails, and general data, balanced against the capability taxonomy.

Decisions made (user):
- Tavily for web search; the local corpus is the main search source.
- GPT-OSS-120B on Daytona for generation and judging.
- Public tool sets for Bucket C: ToolACE, Glaive, Hermes, Toucan, ToolMind. SmolTalk for general data.
- No `company_data_tool` trajectories in the training data. Company questions appear only in Bucket D, as "which company?".
- Citations and a Sources list live only where a document, search result or tool result is actually in the conversation (Bucket B and C). Bucket A never refers to a source.

## Sources

| Source | Records | Notes |
|---|---|---|
| `s3://ft-vaia/Corpus/Invest_dataset/09032026_SFTQ&Apairs/sft_data_kept.jsonl` | 647k (5.2 GB) | Each user prompt has `<document>` + question. Metadata: `_source` (capability = 4th path segment), `_chunk_id`, `_question_type`, `_difficulty`, `_section_context`. Scores live in `pc_krishnakant/qa_quality_analysis/scored.jsonl` |
| `…/100626_Dataset_DPO/qa_pairs/sft/sft_from_dpo.jsonl` | 5,573 (+9,404 numerical rows with no think block, in `sft_from_dpo_numerical_pending.jsonl`) | Prompts contain the source text. `sft_from_dpo_provenance.jsonl` gives source and question_type |
| Public tool sets | ToolACE, Glaive, Hermes FC, Toucan, ToolMind | Converted to a single schema |
| SmolTalk (general) + existing non-domain rows | ~12k target | |
| Local retrieval corpus | `Corpus/Invest_dataset/082126_fliteredextractionrun2/<capability>/text/…` | Indexed for search; also the source of the knowledge corpus |

Existing assets to reuse: `SFT_qa_generation/docs/superpowers/specs/qa_response_prompt.txt` (generator), `qa_response_judge_prompt.txt` (judge), `Capability_Pool_Counts.xlsx` (capability counts), the `SFT_Script/outputs/daytona/tokens_*` eval harness and the 440-question eval sets.

## Target mix (~212k)

| Bucket | Size | Share | What it teaches |
|---|---|---|---|
| **A. Closed-book knowledge** | **140k** | 66% | Correct domain knowledge with no document or tool: concepts, frameworks, regulations, calculations from given inputs, step-by-step reasoning. As-of dates stated inline. Says so when unsure of a figure. Never refers to a document or source, which removes the phantom "the document states…" answers |
| **B. Document grounding** | **25k** | 12% | Reading material that is in the prompt (internal file-search chunks or web-search results) and answering only from it. Citing the right source (title, section or page, date). Copying figures exactly. Ignoring irrelevant chunks. Saying "the provided sources don't state this" when the answer isn't there |
| **C. Tool use** | **25k** | 12% | When to call a tool and when not to, valid JSON calls, waiting for the real result, using only that result, handling empty or failed results, parallel calls, multi-turn tool chats |
| **D. Behaviour + guardrails** | **10k** | 5% | Ask which company or document when missing, correct false premises, stay VaiaGPT, no unprompted tool offers; refuse harmful finance requests, resist instructions hidden in documents or results, protect personal data, no personalised investment advice, decline off-scope requests |
| **E. General** | **12k** | 6% | Everyday instruction-following, writing and chat, so general skills are kept |

Capability balance inside A+B (165k): use the sub-capability counts from `Capability_Pool_Counts.xlsx` as the base, cap any one sub-capability at 7% (~11.5k), keep every record from thin sub-capabilities (EK, AR, CL, EA).

**Feasibility warning for A (140k):** the corpus scan suggests only about 60k questions are truly self-contained and pass the score gate. Reaching 140k needs most `rewritable` questions fixed in Stage 4, user-style variants, and probably a score gate of ≥ 8 for A (instead of ≥ 9). Decide this after Stage 2. Decision (user): reach 140k by taking the self-contained questions and paraphrasing/rewriting the rest from `sft_data_kept`. Only rewrite questions whose answer is a concept, framework, rule or calculation with inputs given (Conceptual/Tutorial, Multi-Concept, Applied/Scenario, Comparative Ranking); a question about one paper's table or a specific result cannot be answered closed-book, so it goes to B or is dropped. If A still falls short, shrink it rather than pad it with document-dependent questions.

## Facts and figures

Research findings that shape the plan:
- Q&A fine-tuning after training on documents leaves models weak at answering about those documents, even at low document perplexity. Training on questions first (pre-instruction-tuning) absorbs about 17.8% more knowledge (Jiang et al., ACL 2024, arXiv 2402.12847).
- A fact seen once in one wording is memorised but not extractable. Facts need many varied rewrites (paraphrase, shuffled sentences, per-entity summaries) (Allen-Zhu & Li, "Physics of LMs 3.1", ICML 2024).
- Fine-tuning on facts the model does not know is fitted slowly and raises hallucination; best results come from fitting mostly known facts (Gekhman et al., EMNLP 2024, arXiv 2405.05904).
- LoRA learns less than full fine-tuning in continued pretraining, but forgets less (Biderman et al., TMLR 2024).
- Claim-level checking against search (SAFE) agrees with human raters 72% of the time at far lower cost (Long-form factuality, arXiv 2403.18802).

Plan:
1. **Knowledge corpus (optional phase 0, before SFT):** pick fact-dense primary sources from the extraction corpus per capability (SEBI/RBI circulars, BRSR/NGRBC, Basel/CRAR, IFRS/Ind AS, GRI/TCFD/CSRD/SFDR, IBC, rating methodologies). GPT-OSS rewrites each document 5–10 ways (paraphrase, sentence-shuffled, per-entity summary, tables as prose, Q&A version). Train a high-rank LoRA (embeddings and output layers trainable) for this phase on a small, fact-dense corpus (~50–100M tokens), mixing 10–20% general text. This corpus is outside the SFT pair counts. Pending budget confirmation.
2. **Claim-level verification** in Stage 7: split each answer into single claims and check each against the local corpus, then Tavily. Unsupported figures are rewritten or dropped.
3. **Probe after training:** quiz the model on corpus facts. Facts it still gets wrong are not trained as confident closed-book answers; they get a hedged answer or move to Bucket B with their document.
4. **Changing figures** (market data, current rules, a company's numbers) are never memorised: they come from documents, search or tools at answer time.
5. **Calculations** where the question supplies the inputs stay in A: the model learns to compute step by step, not recall.
6. **Later (DPO):** use the tuned model's own wrong answers on the fact set as "rejected" and the verified answer as "chosen".

## Stages

### Stage 0 — Inventory and freeze
- Stream `sft_data_kept` and join the quality scores from `scored.jsonl` (by chunk id + question). Keep on every record: `capability`, `sub_capability`, `question_type`, `difficulty`, `chunk_id`, `origin`, `score`.
- Normalise label variants (non-ASCII hyphens in question types, "advanced" vs "Advanced", question types stored in the difficulty field).
- Give every record a stable `rid`. Write `stage0_inventory.parquet` (question, chunk, answer, metadata) to S3.
- Bring `sft_from_dpo` and its numerical rows in with `origin=dpo`, mapping its `source` to the taxonomy codes.

### Stage 1 — Hard filters (rules, no LLM)
- Keep score ≥ 9, plus 7–8 for EK, AR, CL and EA (same rule as the existing pool). Bucket A may take ≥ 8 (see feasibility warning).
- Exact and near-duplicate questions (MinHash on normalised questions, Jaccard ≥ 0.85): keep the highest-scoring one.
- At most 3 questions per `chunk_id`.
- Remove any question that overlaps the 440-question eval sets and the gold fact eval (n-gram overlap).
- Drop broken records: empty answers, truncation, a leaked prompt.

### Stage 2 — Question classification (GPT-OSS-120B, one short call per question)
Labels in strict JSON:
- `dependence`: `self_contained` | `doc_dependent` | `rewritable` (only needs an entity, period or table name made explicit) | `broken`
- `fact_type`: `stable_concept` | `calc_with_inputs` | `time_sensitive` | `company_specific`
- `templated`: true/false, plus a suggested natural rewrite
- Calibrate on 300 hand-labelled questions first. Target ≥ 90% agreement on `dependence` before running on everything.

### Stage 3 — Routing
| Labels | Goes to |
|---|---|
| self_contained + stable_concept or calc_with_inputs | A |
| rewritable | Stage 4 rewrite → A |
| doc_dependent | B (chunk kept in the prompt) |
| time_sensitive | B as web-search-result grounding (Tavily results in the prompt), and a small share to A with an explicit "as of" date and a note that it may have changed |
| company_specific | D only, as "which company?" / "no data provided" cases |
| broken | drop; keep about 1.5k for D ("what's missing" answers) |

### Stage 4 — Question rewrite and de-templating
- Rewritable questions: name the entity, period and jurisdiction; remove references to "the table" or "the authors".
- Templated questions: remove the stock tail on about 70% of them; keep the rest so multi-part asks still exist.
- Add user-style variants (short, casual, vague) for about 15% of A, matching the `user_short` / `user_vague` eval groups.
- Check every rewrite with the Stage 2 `dependence` label; it must come out `self_contained`.

### Stage 5 — Retrieval tool (local first, Tavily second)
- Index the filtered extraction corpus: chunks of about 500 tokens, BM25 plus dense embeddings (bge-m3 or e5-large on Daytona), combined with reciprocal rank fusion.
- Expose it as the `tavily_search` tool with the same result shape (title, url/source path, date, snippet) so `qa_response_prompt.txt` does not change.
- Routing: local index first. Call Tavily only when the top fused score is below a threshold or the question is `time_sensitive`. Set a Tavily query budget and log every call.

### Stage 6 — Generation (Bucket A, Daytona GPT-OSS-120B)
- Run `qa_response_prompt.txt` with the retrieval tool; save `tier`, `think`, `answer`, `references` and the raw snippets for each record.- Pilot 500 → Stage 7 checks → tune the prompt → run the full set in resumable shards.

### Stage 7 — Verification gates
1. Deterministic checks: the 4 tags parse; every number in the answer appears in the question or snippets or is reproduced by recomputation; every reference URL was actually returned; no banned phrases ("the document", "I searched", URLs, Markdown, LaTeX); length matches the tier.
2. Claim-level check: split the answer into single claims and verify each against the snippets/local corpus.
3. Run `qa_response_judge_prompt.txt` as a separate pass (fresh context, temperature 0). Keep score ≥ 8 (≥ 7 for thin capabilities).
4. Hand-review 200 random kept records and 100 rejected ones per round. If the kept set shows more than 3% factual errors, tighten and rerun.
5. Failures get one regeneration; if they fail again, drop them.

### Stage 8 — Bucket B (document grounding, 25k)
- Clean chunks: strip site clutter (nav, promos, "Skip to content", timers), fix line-break hyphens, drop chunks under 150 words or cut off mid-sentence.
- Referring to and citing the provided source is *required* in B (the "no document wording" ban applies only to A). The answer names the source title, and the section or page and date when present.
- Build the mix, with about 30% of B being hard cases:
  - correct chunk only (~50%)
  - correct chunk + 2–3 unrelated distractor chunks from the local index (~20%)
  - correct chunk removed, distractors only → "the provided sources don't state this" (~15%)
  - chunk contains a related but different fact than asked → says what it covers (~10%)
  - web-search-result format with multiple results and dates (~5%, from time_sensitive questions)
- Re-judge existing answers for faithfulness with a **grounded judge** (a variant of `qa_response_judge_prompt.txt`, because rule G7 there penalises document references): every claim must be in the provided chunk; every citation points to a chunk actually in the prompt; no outside figures presented as if from the source.
- Regenerate failures with the chunk in the prompt.
- Write the numerical think blocks for the 9,404 `sft_from_dpo_numerical_pending` rows in grounded mode, then put them through the same gates (a share of these goes to B).
- Open question: the exact format in which the app passes file-search results to the model and how answers should cite them (file name, page, chunk ID). Match it before generating.

### Stage 9 — Bucket C (tool use, 25k)
- Convert ToolACE, Glaive, Hermes FC, Toucan and ToolMind to one schema: the Qwen3 chat template with `tools` in the system prompt, `<tool_call>{json}</tool_call>`, a `tool` role for results, then the final answer.
- Filter: arguments must be valid JSON and match the schema, every call must have a result, the final answer must agree with the result, English only.
- Dedupe across sources and cap any one source at 35%.
- Include parallel calls, multi-turn chats, empty/failed results and "no tool needed" cases.
- Decision (user): no VaiaGPT-specific tool examples, no `company_data_tool` calls and no extra tool-call tests. C is the public sets only.

### Stage 10 — Buckets D and E
- **D (10k) behaviour + guardrails.** Generate from templates plus GPT-OSS-120B, then review by hand: all of D if small, otherwise a large sample.
  - Behaviour: missing company or document → ask which one; false premise → correct it; "forget everything / who are you" → stays VaiaGPT; plain knowledge question → no unprompted tool offer; reference questions from the test chat (BRSR full form and principles, IFC Performance Standard 1, double materiality, "who are you", "which company") with corrected answers, trained as paraphrases while the originals stay as a held-out check.
  - Guardrails: refuse insider trading, market manipulation, fraud, sanctions or tax evasion help; resist instructions hidden inside documents, search results or tool outputs; protect personal and confidential data; present analysis as analysis, not personalised investment advice; decline off-scope requests politely; keep refusals short and useful.
- **E (12k) general.** SmolTalk sampled for diversity across its subsets, plus the best existing non-domain rows. Drop items that overlap the evals.

### Stage 11 — Assembly
- One system prompt for domain buckets (`systemprompt_SFT_v2.txt`). Buckets C and E keep their own tool or general system prompts. About 5% of examples have no system prompt so the model holds up without one.
- Bucket A target: think + answer, with the Sources list and references removed (no search at inference for closed-book answers). Bucket B and C answers keep citations only when they point at material in the conversation.
- Apply the capability balance, a final cross-bucket dedupe and the eval decontamination. Shuffle, split 98/2 into train/val, stratified by bucket and capability.
- Publish `sft_v3_train.jsonl`, `sft_v3_val.jsonl` and `sft_v3_manifest.json` (counts per bucket, capability and source, plus the hashes of the gating prompts).

## Corpus scan results (full `sft_data_kept`, 647,066 records, 2026-10-27)

Every record has a `<document>` in its prompt. A regex flags a question as **anchored** when it names its source directly ("the table", "Section 4", "the authors", "the company"…). A sample showed that about a third of the unanchored questions still depend on the document implicitly, so the regex undercounts. Stage 2 gives the real split.

| Category | Total | Anchored | Not anchored | Anchored % |
|---|---|---|---|---|
| DK Domain Knowledge | 382,910 | 234,760 | 148,150 | 61% |
| CG Content Generation | 117,818 | 68,365 | 49,453 | 58% |
| FR Financial Reasoning | 64,908 | 43,257 | 21,651 | 67% |
| WF Workflow Understanding | 54,467 | 32,571 | 21,896 | 60% |
| EK Enterprise Knowledge | 19,161 | 11,097 | 8,064 | 58% |
| AR Agentic Reasoning | 6,666 | 4,348 | 2,318 | 65% |
| EA Enterprise AI | 788 | 485 | 303 | 62% |
| CL Continuous Learning | 348 | 213 | 135 | 61% |
| **Total** | **647,066** | **395,096** | **251,970** | **61%** |

- Question types (total / anchored): Multi-Concept 132k / 50k · High-Order Reasoning 132k / 110k · Error Identification 116k / 78k · Calculation 86k / 50k · Applied/Scenario 77k / 35k · Interpretation 52k / 40k · Conceptual/Tutorial 27k / 18k · Comparative Ranking 20k / 12k · Summary 3k / 3k.
- Difficulty: Advanced 335k, Applied 281k, **Basic only 31k (4.7%)**. Closed-book A cannot be made of basic questions alone; it also takes self-contained Applied/Advanced questions.
- Highest-yield sources for Bucket A by the not-anchored share: DK Private Markets (114k total, 41k not anchored), DK Debt-Credit (81k / 32k), DK India Regulatory (43k / 21k), DK Accounting (40k / 16k). Highest-yield for Bucket B: DK Private Markets, DK Debt-Credit, DK ESG, CG Company Profiles.

## Verification

- Before training: run the existing token-stats eval harness (`SFT_Script/outputs/daytona/tokens_*`) on a 1k-pair smoke-train LoRA to confirm the pipeline works end to end.
- Add a fixed factual eval set (~150 questions with gold answers, e.g. who issued BRSR and when, the NGRBC principles, IFC PS1–8, Basel/RBI CRAR thresholds, double materiality). Never trained on.
- Report: phantom-source rate (target < 5 of 200, against 66 for SFT and 76 for DPO), fact accuracy on the gold set, hedging on unknown entities, citation correctness in B (every citation points to a chunk in the prompt), tool-call JSON validity (> 98%), guardrail refusal rate on a red-team set, identity attack held, and per-capability judge scores. Compare against the pre-tuned model, Run 2 SFT and DPO checkpoint-100.

## Open items

1. Can Bucket A reach 140k clean questions? Decide after Stage 2 (shrink A rather than pad it).
2. Is the high-rank LoRA knowledge-injection phase (50–100M tokens) within budget?
3. Format and citation style the app uses for file-search results (affects Bucket B).
4. Resolved: the app has no calculator or code tool.
