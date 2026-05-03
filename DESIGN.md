# PersonaMem-v3: Design Document

*Towards All-Day-Long Omni-Platform Personal Intelligence*

> Design rationale for the PersonaMem-v3 data generation pipeline. For implementation, see `persona_agent.py`, `prompts.py`, and `skill.md`.

## What's in this release (R1, R5–R10)

Recent additions on top of the base pipeline, roughly in dependency order:

- **R10a — Canonical-modal hashtag prune (Step 3.1, in `cross_reference_personas`):** after merging atomics with identical `persona_item` text into a canonical, prune outlier atomics whose `source_hashtags` don't overlap the canonical's modal hashtag set (top-`CANONICAL_MODAL_TOP_K = 5` most-frequent hashtags across the cohort, computed by row-frequency so a single hallucination can't dominate). Only applied when `cohort_size ≥ CANONICAL_MODAL_MIN_COHORT = 3` — singletons / pairs are kept verbatim. Fixes a class of LLM-hallucination bug where a per-row inference call returned a `persona_item` topically unrelated to the row's hashtags but happened to lexically collide with a real canonical from another row, inflating `confidence_cross_referenced` and fanning out to topically-unrelated events at `save_to_backend` (e.g., a `#smokedhog` BBQ event carrying a "sour and gummy candy" preference). A post-hoc cleanup script (`scripts/clean_existing_personas.py`) applies the same gate to already-emitted `backend/{uid}/*.json` without a pipeline regen, with an LLM-judge tiebreaker (`pref_event_grounding_check_prompt`) for borderline pairs (lenient toward name/genre matches like `#kaicenat` for a "comedy" canonical, strict on clear semantic mismatches).
- **R10b — Per-family inferior-response generation (`evaluation/llm_postprocess.py`):** the paired `inferior_response` foil — used in eval tasks to test whether models can distinguish a personalized response from a plausibly-wrong-for-this-moment one — is now generated differently per task family: ranking tasks (`personalized_recommendation`, `personalized_feed_ranking`, `at_ai_directive_followup`, `short_vs_long_term_lifecycle`) use a deterministic ordering inversion via `_compute_ranking_inferior` (no LLM call); list/digest tasks replace one bullet with a disliked-topic alternative; voice tasks paraphrase the same factual content into a contrasting voice register (Jaccard < 0.6 target); freeform tasks write an independent rewrite that does NOT echo the gold's opening clause. After every LLM-rewrite generation, `_validate_inferior` rejects pairs that fail any of: prefix overlap, substring containment, opening-N-tokens overlap, token Jaccard out of bounds, or length-ratio > 0.5 — with up to 3 retries before dropping the foil. `_EXAMPLE_GEN_PROMPT` was tightened to forbid telegraph phrases ("as a fan of X", "since you love Y") that would mark a response as the personalized one. Fixes a near-universal failure mode where 94/96 example/inferior pairs in user 115's benchmark were either prefix-overlap (gold + appended clause) or minimal-edit (one word swapped). After the fix: 103/103 pairs pass the validator. See `scripts/regenerate_inferiors.py` for in-place re-emission of just the inferior fields without a full benchmark rebuild, and `evaluation/audit_example_inferior_pairs.py` for a checked-in audit tool.
- **R7 — Ad injection (Step 20):** ~6% of commerce-adjacent events become sponsored ads with ad-shaped content (`ad_metadata`). New `AD_ACTIONS` (`clicked_ad`, `hidden_ad`, `dismissed_ad`) on IG/FB/Threads; Chatbot never carries them. Invariant: `event.is_ad ⇔ action ∈ AD_ACTIONS`.
- **R6 — Time horizon + stop conditions (Step 4):** every surviving canonical carries `time_horizon ∈ {"short_term", "long_term"}`. Short-term (bounded intents — travel, event prep, purchase, how-to, medical) uses `XREF_THRESHOLD_SHORT_TERM = 3.0` instead of the 20/50 long-term bars. An LLM pass emits structured `stop_condition: {type, description, expected_stop_ts}`. LLM can demote short→long but not promote long→short.
- **R1 — Cross-polarity contradiction causality gate (Step 7):** the positive/negative cross-ref pipelines are now cross-checked. Pos/neg canonical pairs sharing ≥2 hashtags are LLM-confirmed as semantically opposite, then must pass a temporal-precedent rule: the later stance survives only with ≥ `MIN_STANCE_FLIP_PRIOR` same-polarity rows before the first opposing row. Failed gates drop the later canonical; `update_history` entries carry `resolution: "suppressed_insufficient_precedent"` or `"stance_shift_with_precedent"`. Fixes the 115-boxing bug (stance flip 1h apart with no prior evidence).
- **R5 — Per-session geolocation + calendar modification stream (Steps 15, 16):** each event carries `event_location` shared across all rows in its session. `backend/{uid}/calendar.json` holds an add/update/remove stream for synthetic calendar events. `BackendQuery.get_calendar_state(T)` folds modifications with `ts ≤ T` into the live calendar state at time T. Home-only + up to 2 travel cities cap for an 8-day window.
- **R8 — Drop split / over_personalization_irrelevant:** the data-gen output is pure history. Eval picks its own test moments dynamically by cutting the timeline — no pre-flagged train/test partition.
- **R9 — Benchmark CSV + contradiction-aware GT:** `build_benchmark` additionally emits `benchmark/{uid}/benchmark.csv` for HuggingFace publication. `BackendQuery.get_preferences(..., include_superseded=False)` filters canonicals superseded at T (via the `"stance_shift_with_precedent"` update_history entry).

See EVAL.md for the corresponding E2/E3/E4/E5 evaluation tasks that consume these signals.

## Core Research Questions

1. How can we simulate data reflecting real-world distributions/formats to create a personalization dataset mimicking all-day digital behavior?
2. How well can LLMs understand user personas from noisy cross-platform contexts to provide proactive personalization?

---

## 1. Conceptual Framework

Three-stage generative process:

```
Persona Generation  -->  Preference Generation  -->  Omni-Platform Interaction History
```

- **Stage 1 — Persona Generation:** Infer rich user persona (demographics, personality, career, education, bio) from raw hashtag interaction logs. This "skeleton" is the shared ground truth all downstream artifacts trace to.
- **Stage 2 — Preference Generation:** Extract, filter, and cross-reference atomic preference statements. Score, deduplicate, validate to form a high-confidence preference set anchored to the skeleton.
- **Stage 3 — Omni-Platform History:** Distribute preferences across four platforms (Instagram, Facebook, Threads, AI Chatbot) producing realistic, timestamped interaction events. Noise is injected *after* skeleton establishment.

Three interaction pillars:

| Pillar | What it models | Platform(s) |
|--------|---------------|-------------|
| Social Media Engagement | Feed browsing: likes, saves, shares, skips, @ai comments | Instagram, Facebook, Threads |
| Human-LLM Chat | Conversational queries, ask-to-forget, corrections | AI Chatbot |
| Multi-Platform Interactions | Cross-app routing, session-based browsing, per-app personas | All four |

**Single Ground-Truth Principle:** Every event on every platform traces to one shared preference skeleton, established in Steps 1-2 and locked before platform-specific generation. Preferences appearing on multiple platforms are the same canonical preference. Per-app sub-personas describe *how the user presents*, not *what they like*. Noise (8% app reassignment, action perturbation) applies after skeleton finalization.

**Core Tensions:**

| Tension | Choice |
|---------|--------|
| Signal fidelity vs. coverage | Strict filtering (init >= 0.75, 7-day recency gate on corroboration, bottom-20% removal) — choose fidelity |
| Realism vs. tractability | Approximate with session routing, per-user action distributions, temporal evolution |
| Fairness vs. accuracy | Diversify demographics beyond census; avoid stereotypical combos — choose fairness |

---

## 2. Pipeline Overview

```
Input CSV (hashtag interactions per user)
  |
  +- Step 1:  Infer atomic personas             [LLM]      -- 1-3 per row, init 0.0-1.0
  +- Step 2:  Promote implicit negatives        [Algo+LLM] -- net-sentiment gate
  +- Step 3:  Cross-reference & filter          [Algo+LLM] -- cross_ref scores (uncapped)
  +- Step 4:  Classify horizons + stops         [LLM]      -- short_term confirmation + stop_condition
  +- Step 5:  Temporal contradiction graph      [LLM]      -- timeline grouping
  +- Step 6:  Build update histories            [Algo+LLM] -- reinforced/faded/evolved
  +- Step 7:  Resolve cross-polarity contradicts[Algo+LLM] -- temporal-precedent stance-flip gate
  +- Step 8:  Generate user profile             [LLM]      -- demographics + Big Five
  +- Step 9:  Infer hidden personas             [Algo+LLM] -- cross-row hashtag clustering
  +- Step 10: Infer MBTI                        [LLM]      -- type + per-dimension probabilities
  +- Step 11: Generate per-app sub-personas     [LLM]      -- 4 AppPersonas
  +- Step 12: Build sessions                    [Algo]     -- temporal grouping (5-sec gap)
  +- Step 13: Route preferences to apps         [LLM+Algo] -- mini-tier; ~40/20/20/20 distribution
  +- Step 14: Assign rows to apps               [Algo]     -- session majority vote + 8% noise
  +- Step 15: Assign session locations          [LLM]      -- mini-tier; home + up to 2 travel cities
  +- Step 16: Generate calendar modifications   [LLM]      -- mini-tier; scattered add/update/remove
  +- Step 17: Generate interaction formats      [Algo+LLM] -- per-user perturbed weights
  +- Step 18: Generate chatbot conversations    [LLM]      -- multi-turn, ask-to-forget
  +- Step 19: Generate synthetic content        [LLM]      -- text / image / short_video per event
  +- Step 20: Inject ad events                  [LLM]      -- ~6% of commerce-adjacent events become ads
  +- Step 21: Annotate stereotype marks         [LLM]      -- demographics-only
  +- Step 22: Save to backend                   [Algo]     -- 5 JSON files per user + calendar.json
```

**Model tiers:** the pipeline uses two LLM clients. The **flagship** model (`gpt-5-chat`) handles reasoning-heavy steps — 1 (atomic persona), 3/5/6 (cross-ref, temporal, histories), 7 (cross-polarity gate), 8 (profile), 9 (hidden personas), 10 (MBTI), 11 (app personas), 18 (chatbot conversations). The **mini** model (`gpt-5.4-mini`, configurable via `--mini_model`) handles mechanical and stylistic steps — 4 (horizon refinement), 9a (intimate-hashtag detection), 13 (app routing), 15 (geolocation), 16 (calendar modifications), 17 (interaction formats), 19 (synthetic content), 20 (ad content), 21 (stereotype marks). Mini falls back to flagship when no mini client is configured.

---

## 3. Input and Output

### Input

CSV of anonymized social media interactions (one row = one user engaging with one piece of content):

| Column | Type | Description |
|--------|------|-------------|
| `interaction_type` | string | `explicit_positive`, `implicit_positive`, `explicit_negative`, `implicit_negative` |
| `user_id` | string | Anonymized user identifier |
| `object_id` | string | Anonymized content identifier |
| `interaction_time` | int | Unix timestamp |
| `object_text` | string | Space-separated hashtags (e.g., `#CrossFit #MorningRoutine`) |

Interaction types: `explicit_positive` (liked/saved/shared), `implicit_positive` (lingered/watched 75%+), `explicit_negative` (hid/muted/unfollowed), `implicit_negative` (scrolled past).

### Output

Per-user directory at `backend/{user_id}/`: `profile.json` (profile + 4 AppPersonas + flat preference list), plus `instagram.json`, `facebook.json`, `threads.json`, `chatbot.json` (interaction events sorted by timestamp).

`profile.json["preferences"]` is a list of strings formatted as `"{latest_timestamp} : {persona_item}"`, sorted by latest timestamp descending (most recent first). The timestamp is the maximum `source_timestamp` across the canonical's supporting atoms.

Each app JSON is an array of interaction events. Each event = one source CSV row with nested `preferences[]`:

```
Event:
  +- source_object_id, source_timestamp, source_hashtags, source_interaction_type
  +- interaction_format: { app, action, action_label, user_message }
  +- [Instagram/Facebook/Threads non-stub only] content_type, content
  +- preferences[]:
  |     +- persona_item, category
  |     +- confidence_score_init, confidence_cross_referenced
  |     +- stereotype_mark, split ("test" on held-out items only; absent otherwise)
  |     +- update_history[], hidden_persona_labels
  |     +- (R8: `split` and `over_personalization_irrelevant` are no longer
  |        emitted by data-gen; the eval harness picks test moments and
  |        builds distractor pools from the full timeline at build time.
  |        See `EVAL.md` for the stratified-Jaccard distractor scheme used
  |        by `over_personalization_distractor_reject`.)
  +- [Chatbot only] conversation[], conversation_type, ask_to_forget
```

Same canonical preference text appears across multiple events (intentional real-world repetition). Per-app files mirror real data silos. Events with zero surviving preferences are omitted.

---

## 4. Step 1 — Atomic Persona Inference

For each interaction row (except `implicit_negative`), the LLM infers **1 to 3** atomic persona traits, treating the full hashtag set as one coherent signal. Each trait has: `persona_item` (specific testable statement), `category` (topical label), `confidence_score_init` (LLM confidence), `source_hashtags`. Keeping the count low reduces downstream scale and forces the LLM to pick only its strongest, most defensible inferences.

### Confidence Scale

**Positive interactions** (full 0.0-1.0):

| Range | Meaning |
|-------|---------|
| 0.80-1.00 | Near-certain, explicitly stated |
| 0.60-0.80 | Direct topic match |
| 0.40-0.60 | Reasonable deduction |
| 0.15-0.40 | Broader inference |
| 0.00-0.15 | Speculative |

**Negative interactions** (compressed range): 0.55-0.75 (direct dislike), 0.35-0.55 (reasonable deduction), 0.15-0.35 (broader inference), 0.00-0.15 (speculative). Always phrased negatively ("Dislikes X", "Avoids X").

Scores must use two decimal places and be spread across the full range to prevent LLM clustering at high values.

---

## 5. Step 2 — Implicit Negative Promotion (user-adaptive)

`implicit_negative` rows are skipped in Step 1 (a single scroll-past is too weak). Instead, aggregated via net-sentiment filtering to distinguish genuine dislike from baseline scrolling. A user skipping 30 boxing posts in one bad-mood evening doesn't *durably* dislike boxing; a user skipping 2-3 boxing posts per day across a week probably does.

**Daily-capped net-sentiment formula** per hashtag:

```
capped_neg = sum(min(day_count, IMPL_NEG_DAILY_CAP) for day_count in negs_by_day)
net = (capped_neg × 1.0) - (explicit_pos_count × 2.0) - (implicit_pos_count × 1.0)
```

The per-day cap (`IMPL_NEG_DAILY_CAP = 5`) prevents a single mood-driven skipping burst from driving promotion. Positive counter-signals (`explicit_pos_count`, `implicit_pos_count`) are NOT capped — they represent deliberate engagement that should always weigh against the negative signal.

**User-adaptive promotion gate** — different users have very different scroll-and-skip volumes. A user generating 3000 implicit_negative rows in a window has a much higher "noise floor" than one generating 500. A single global threshold over-promotes the heavy skipper and under-promotes the light one, so the gate scales with each user's volume:

```
user_threshold = NEG_PROMOTION_RATIO × user_total_impl_neg   # 0.008 × N
```

All three must hold:
1. `net >= user_threshold` — a durable dislike must carry ≥ 0.8% of this user's total skip volume as net-negative signal on one tag.
2. Negative rows span `>= 3` distinct calendar days (`MIN_TEMPORAL_DAYS`).
3. After the per-day cap (`IMPL_NEG_DAILY_CAP = 5`).

Examples from the 10-user gistbench sample (cap=5, days=3):

| User | user_impl_neg | threshold | promoted hashtags |
|---|---:|---:|---:|
| 115 | 872 | 7.0 | 3 (#fightfans + 2 others) |
| 755 | 1909 | 15.3 | 10 (#parenting, #spiritualgrowth, #faith, …) |
| 655 | 2617 | 20.9 | dominated by positive counter-signal → 0 |
| 760 | 3155 | 25.2 | dominated by positive counter-signal → 0 |
| 251 | 2 | 0.02 | 0 (fails MIN_TEMPORAL_DAYS floor) |

`MIN_IMPLICIT_NEGATIVE_REPETITION = 15` is retained for a **different** gate — the cross-ref init filter in Step 3, which requires an implicit-only negative *canonical* to have ≥ 15 distinct source rows to survive. It is NOT the promotion threshold.

**Processing:** One LLM call per "hot" hashtag on a representative row (single hashtag only). Inferred preferences must be independently produced by >= 2 different hot-hashtag LLM calls (`MIN_PREF_CORROBORATION = 2`). Promoted rows become `explicit_negative` at BOTH the atomic level (`source_interaction_type = "explicit_negative"` on every promoted atomic) and the event level. Non-promoted implicit_negative rows remain as stub events with empty `preferences: []` (rendered greyscale in HTML).

---

## 6. Step 3 — Cross-Referencing

Seven sub-stages transform raw inferences into the validated preference skeleton:

1. **Merge Duplicates:** Normalize (lowercase, whitespace-collapsed) and group by exact string match. No semantic dedup — handled later by LLM relationship discovery.

2. **Init Filter:** Drop canonicals with `max(init) < 0.75` (`MIN_PERSONA_INIT_CONFIDENCE`). No exploratory retention — strict floor.

3. **Weighted Corroboration (recency-gated):** Per canonical, count distinct source rows: +1.0 per explicit row (init >= 0.75), +0.5 per implicit row (init >= 0.75). **Only rows whose `source_timestamp` falls within the user's trailing 7-day window (`RECENCY_WINDOW_SECONDS`, anchored on the user's latest interaction) contribute to the score and to the `n_explicit_rows` / `n_implicit_rows` mix.** Older rows still pass the init filter but don't count here — recency is the strictness mechanism, so canonicals supported only by stale evidence fail the survival threshold in Step 7. Score is intentionally uncapped — magnitude is meaningful.

4. **LLM Relationship Discovery:** Per-category LLM calls identify `similar` and `contradictory` relationships. LLM does not alter scores. Categories with one canonical are skipped.

5. **Union-Find Clustering:** Similar preferences merged; cluster representative = highest init. Cross-ref scores summed across cluster. Contradictory relationships preserved.

6. **Contradiction Penalty:** Subtract contradicting canonical's cross-ref score. Floor at 0.0.

7. **Bottom-20% Filter + Per-canonical Survival Threshold:** First remove the bottom 20% by xref (no exemption). Then apply an **evidence-mix-dependent threshold** — a canonical survives iff its `cross_ref` exceeds `canonical_xref_threshold(n_explicit_rows, n_implicit_rows)`, which interpolates linearly between `XREF_THRESHOLD_EXPLICIT = 20.0` (pure-explicit support) and `XREF_THRESHOLD_IMPLICIT = 50.0` (pure-implicit support). Canonicals backed mostly by implicit positives thus face a substantially higher bar to survive.

**Negative cross-referencing** runs the same pipeline independently (within negatives only). Differences: canonicals with only implicit evidence need >= 10 distinct source rows to survive; the bottom-20% step is skipped; the per-canonical xref threshold in step 7 still applies (same recency window as positives).

### Step 4 — Time Horizon + Stop Conditions

With the observation window being short (~8 days), time horizons must be inferred from category + span fraction + row count rather than raw span in days.

**Rule-based pre-label (runs INSIDE Step 3, before the survival filter):** a canonical is eligible for `short_term` iff `(span_days / obs_window_days) ≤ SHORT_TERM_MAX_SPAN_FRAC` (0.35) AND `n_rows < SHORT_TERM_MAX_ROWS` (8) AND `category ∈ SHORT_TERM_ALLOWED_CATEGORIES` (travel, event_prep, purchase_intent, how_to, medical_consultation, trip). Everything else defaults to `long_term`. The allow-list is the anti-loophole — a canonical cannot claim short-term just by having a sparse tail; it must be in a bounded-intent category.

Short-term canonicals use `XREF_THRESHOLD_SHORT_TERM = 3.0` instead of the long-term 20/50 interpolation, letting legitimate one-off intents (hotel recon, how-to search, upcoming event prep) survive despite little corroborating evidence.

**LLM refinement (Step 4 — new step):** one batched mini-tier call per ~20 rule-labeled short-term candidates. The LLM may:
- **Confirm** `short_term` and emit a structured `stop_condition`:
  ```json
  {"type": "event"|"date"|"mastery"|"relocation",
   "description": "...",
   "expected_stop_ts": <unix int | null>}
  ```
- **Demote** to `long_term` when the item is actually an enduring trait sampled sparsely.

The LLM **cannot promote** `long_term` → `short_term` (only rule-labeled candidates are in the prompt). This guards against weak long-term signals bypassing the short-term xref floor.

The prompt explicitly tells the LLM that in an 8-day window, "long_term" means "an enduring identity trait inferable from this window" — not literal year-scale persistence. `expected_stop_ts` may fall outside the observed window; eval tasks handle this by clamping to a synthetic post-window moment.

---

## 7. Steps 4-5 — Temporal Evolution

**Step 4 — Contradiction Graph:** Contradictory preferences grouped by topic with chronological timelines showing stance shifts.

**Step 7 — Cross-Polarity Contradiction Gate (R1 fix):** Positive and negative canonicals can survive their independent cross-ref pipelines with no awareness of each other. Step 7 walks the Cartesian product of surviving (positive, negative) canonicals, filtering to pairs sharing ≥ `HASHTAG_OVERLAP_MIN = 2` source hashtags. A mini-tier LLM call classifies each candidate pair into ONE of:

- **`contradiction`** — same topic AND same granularity, opposite stance (e.g., "Interested in NFL football content" vs "Not interested in NFL football content"). Gets the full dominance + precedent + ambivalence resolution pipeline below.
- **`ambivalence`** — same topic but DIFFERENT granularities (e.g., "Interested in NFL football content" vs "Not interested in NFL training-camp and team-specific football content"). Both sides are real user stances at different levels of specificity. BOTH survive, marked `update_type: "ambivalent"` with `resolution: "different_granularity"`. No dominance check, no precedent check — they're legitimate coexistence, not rivals.
- **`unrelated`** — no opposing stance relationship; skipped.

Each **contradiction** pair is resolved in three further stages:

1. **Dominance check.** If `stronger_rows / weaker_rows >= DOMINANCE_DROP_RATIO` (2.5), the weaker canonical is dropped as noise regardless of temporal order. The survivor's `update_history` gets an entry with `resolution: "suppressed_weak_minority"` + the ratio.
2. **Temporal-precedent rule.** The LATER-emerging stance is kept only if `same_polarity_rows_before_opposite_first_row >= MIN_STANCE_FLIP_PRIOR` (5 for long_term, `MIN_STANCE_FLIP_PRIOR_SHORT = 1` for short_term). When the gate FAILS, the later canonical is demoted with `resolution: "suppressed_insufficient_precedent"`.
3. **Concurrent-ambivalence detection** (when both previous stages pass). If the earlier side still has ≥ `MIN_EARLIER_POST_FLIP_FOR_CONCURRENT` (5) rows AFTER the later side's first row, both polarities are interleaved — not a clean temporal shift. Both survive with `resolution: "concurrent_ambivalence"`. Otherwise it's a clean shift and the entries carry `resolution: "stance_shift_with_precedent"`.

Dropped canonicals are stored in `self._suppressed_stance_flips` for audit. The visualizer distinguishes the resolution labels: `stance_shift_with_precedent` (red, emphatic), `concurrent_ambivalence` / `different_granularity` (amber, "mixed feelings"), `suppressed_weak_minority` / `suppressed_insufficient_precedent` (grey, strikethrough). `update_type` is `"contradicted"` for same-granularity pairs and `"ambivalent"` for different-granularity pairs — terminology difference makes the nature of the disagreement explicit in the history.

**History causality:** every cross-polarity entry now carries the `source_object_id` of the opposing canonical's first event. The `save_to_backend` causality filter then uses lexicographic `(ts, oid)` ordering so entries are emitted strictly before the current event in the HTML display order. Entries with no `source_object_id` fall back to strict `ts < event_ts` (drop same-timestamp).

**Step 5 — Update Histories:** Each preference gets a temporal `update_history[]` array with entries tagged by `update_type`:

| `update_type` | Definition |
|----------------|-----------|
| `new` | First appearance (filtered from serialization — redundant with event timestamp) |
| `reinforced` | Multiple distinct source rows; up to 5 samples, evenly spaced |
| `faded` | Inactive > 48h before user's last activity (`FADE_THRESHOLD_SECONDS = 172,800`) |
| `contradicted` | **Same-granularity** contradicting preference — "Interested in NFL" vs "Not interested in NFL" (detected by Step 7's `contradiction` classification) |
| `ambivalent` | **Different-granularity** coexisting preference — "Interested in NFL football" vs "Not interested in NFL training-camp" (Step 7's `ambivalence` classification). Both stances are real; neither is noise |
| `deepened` | General interest became more specific over time |
| `branched` | Interest expanded into new sub-direction |
| `shifted` | Focus moved within same domain |
| `intensified` | Engagement grew demonstrably stronger |
| `similar` | Semantically similar preference discovered in cross-ref |

Entry fields: `update_type` (all), `preference` (contradicted/ambivalent/deepened/branched/shifted/similar), `formatted_timestamp` (all), `source_object_id` (reinforced/contradicted/ambivalent, for lexicographic causality ordering), `source_app` (reinforced/deepened/branched/shifted/similar/contradicted/ambivalent), `occurrence`+`total_occurrences` (reinforced), `description` (deepened/branched/shifted/intensified), `resolution` (contradicted/ambivalent — see Step 7).

**Causality filter:** `save_to_backend` emits each entry only if its `(timestamp, source_object_id)` is strictly less than the current event's `(timestamp, source_object_id)` — i.e., the referenced event appears BEFORE the current event in the HTML display order. Entries without `source_object_id` fall back to strict `timestamp < event_timestamp` (same-timestamp entries are dropped as indeterminate).

---

## 8. Step 6 — Synthetic User Profile

Demographics sampled first; everything downstream (name, career, bio) must be consistent.

**Gender x Orientation** (21 entries, key ones): Cis female hetero 30%, Cis male hetero 32%, Cis male gay 5%, Cis female bi/lesbian 4% each, Non-binary queer 2%, Trans female/male hetero 2% each, remaining categories 0.5-1% each.

**Race/Ethnicity** (28 entries, key ones): White American 15%, Chinese 10%, Black/African American 8%, Indian 8%, Mexican American 8%, Filipino/Vietnamese/Korean 4% each, Japanese/MENA/Multiracial 3% each, remaining categories 1-2% each. Intentionally diversified beyond census.

**LLM-generated fields:** Name (culturally appropriate), Career (consistent with *some* preferences), Education, Big Five personality (each low/medium/high), Bio (3-5 sentences). LLM instructed to avoid stereotypical demographic-career-hobby combinations.

**Mobility class (v0 / e6 substrate):** Each user is assigned a `mobility_class ∈ {homebody, domestic, international, nomadic}` at profile generation time using an MD5-seeded per-user RNG (deterministic across regen runs). Distribution across the cohort: ~30% homebody, ~40% domestic, ~20% international, ~10% nomadic. Not every user moves around in an 8-day window — homebodies stay in their home city for the full window and are explicitly NOT forced into a trip arc. The class drives class-adaptive constraints in Step 15 (city count, home-share floor, trip-arc presence) and Step 16 (class-conditional transit entries).

---

## 9. Step 7 — Hidden Persona Inference

Infers deeper motivational layers (*why* a user engages, not just *what* they like) from cross-row hashtag patterns. Grounded in behavioral science.

### Three-Phase Algorithm

**Phase 1 — Hashtag Census (algo):** Count occurrences, per-type breakdown, distinct days for each hashtag. Filter to >= 3 occurrences (`HIDDEN_PERSONA_HASHTAG_MIN_FREQ`). Pass top ~200 (`HIDDEN_PERSONA_TOP_HASHTAGS`) to LLM.

**Phase 1b — Intimate + Medical Pre-Screen (LLM):** A single mini-tier call (`detect_intimate_or_medical_hashtags_prompt`) classifies positive-signal hashtags into two independent buckets: (1) adult/kink/sexually-suggestive, and (2) medical / aesthetic-medicine (specific medication, dermatology active, aesthetic procedure, weight-loss / hormone treatment, dental aesthetic, supplement, or chronic-condition management practice that implies the user is *applying / taking / preparing to take*, not just curious). No keyword list lives in code — substring heuristics produce too many false positives on both axes (`cummins` / `hotchicken` / `earthporn` for intimate; `MassageTherapy` / `CarSeatGapFiller` / `woodfiller` for medical). Flagged hashtags from either bucket are **force-included in the top-N table** even if their counts fall below `HIDDEN_PERSONA_HASHTAG_MIN_FREQ`, so a single high-stakes signal cannot be dropped.

**Phase 2 — LLM Clustering:** Groups hashtags into **at most 6** thematic clusters, actively using the user's profile (demographics, career, bio) to ground inference. The prompt flags `intimate_interest` and `covert_concern` as priority signals that must be surfaced whenever hashtag evidence supports them. Twelve types (11 discovered + 1 injected):

| Type | Captures | Basis |
|------|----------|-------|
| `personality_trait` | Core character attributes | Big Five; Dark Triad behavioral markers |
| `aspiration` | Dreams and goals | Maslow's esteem/self-actualization |
| `emotional_pattern` | Recurring emotional dynamics | Uses & Gratifications (affective) |
| `identity_anchor` | Cultural era, tribal belonging (overt + covert markers) | Identity Signaling Theory |
| `intimate_interest` | Body confidence, sensuality, attraction patterns | Self-presentation research |
| `intellectual_curiosity` | Hidden learning interests | Self-Determination Theory |
| `private_hobby` | Consumed but not publicly shared (high implicit ratio) | Uses & Gratifications (escapist) |
| `parasocial_attachment` | Intense bond with public figure (>= 15 rows) | Parasocial Relationship Theory |
| `compensatory_need` | Unmet needs via private consumption (privacy_ratio > 0.7) | Compensatory Internet Use Theory |
| `covert_concern` | Specific worries / fears / pressures the user privately dwells on (health anxiety, financial stress, parenting worry, relationship insecurity, body-image pressure) | Uses & Gratifications (reassurance seeking) |
| `medical_aesthetic_concern` | Active engagement with a specific medication, dermatology active, aesthetic procedure, GLP-1 / hormone treatment, hair-loss treatment, dental aesthetic, supplement, or chronic-condition practice (regimen comparisons, side-effect threads, before/after, "is it safe to combine" questions) where the engagement implies the user is *applying / taking*, not just curious. Distinct from `covert_concern` (anxiety-driven) and `private_hobby` (passive consumption) by signalling a downstream **interaction surface** — drug-drug, drug-procedure, product-sun, post-procedure aftercare — that future personalization must respect | Compensatory Internet Use (active management) + safety-relevant exposure tracking |
| `sensitive_life_event` *(synthetic — Phase 5)* | Discrete, time-bounded personal episodes the user is actively navigating (divorce, surgery, breakup, gender exploration, parent conflict, etc.) | Distinct from `covert_concern` (ongoing worry) by being *episodic* with start + end |

**Phase 3 — Validation:** Each cluster needs >= 40 distinct rows (`MIN_HIDDEN_PERSONA_ROWS`) and >= 3 distinct days (`MIN_HIDDEN_PERSONA_DAYS`). Privacy ratio reported (> 0.7 required for `compensatory_need`). `first_seen_ts` / `last_seen_ts` are derived from the cluster's evidence rows (min/max `interaction_time`). **Exemptions:** `intimate_interest` clusters whose evidence overlaps the Phase-1b intimate pre-screen skip both gates entirely (one positive signal is enough). `medical_aesthetic_concern` clusters whose evidence overlaps the Phase-1b medical pre-screen drop the row floor to `MIN_HIDDEN_PERSONA_ROWS_MEDICAL` (15) and the day floor to `MIN_HIDDEN_PERSONA_DAYS_MEDICAL` (2) — a steady GLP-1 / retinoid / hormone regimen produces only a handful of weekly engagements but is high-stakes for downstream personalization.

**Phase 4 — Deduplication:** Merge hidden personas with Jaccard >= 0.5 on evidence hashtags. Persona with more evidence_rows becomes base; hashtags and surface_connections unioned; metrics + first/last_seen recomputed. Repeats until no merges.

**Phase 5 — Sensitive-Life-Event Injection (LLM-personalized):** Every user gets exactly one synthetic `sensitive_life_event` cluster bundling **1–3 episodes** drawn from `SENSITIVE_LIFE_EVENT_TOPIC_MENU` (15 topics — divorce, breakup, surgery, gender/sexuality exploration, parent conflict, miscarriage, job loss, addiction recovery, mental health diagnosis, custody dispute, fertility struggle, death in family, chronic illness diagnosis, abuse recovery, financial collapse). A mini-tier LLM call (`personalize_sensitive_life_event_prompt`) picks topics that **fit this user's profile + hidden personas + top hashtags**, demands diversity across themes, and writes ALL user-facing text from scratch — `label_fragment`, `specific_situation` (1–2 sentences of grounded detail), `evidence_hashtags` (4–6, lowercase), and 3 `exemplar_persona_items` (≤ 10 words each, tied to the situation). **No template fallback** — if the LLM call fails the user simply gets no `sensitive_life_event` persona. Each event carries `[first_seen_ts, last_seen_ts]` placed at random points in the user's observation window plus an `active_window_end = last_seen_ts + 14 days` ("still raw" buffer). The cluster is marked `is_synthetic=True` with `privacy_ratio=1.0`. The cluster's `evidence_oids` list is empty (no real backing rows) — `hidden_persona_labels` will not link real preferences to this cluster, which is by design: the eval grades against the cluster's own `events[].exemplar_persona_items` directly. The Step 21 audit (`audit_persona_safety`) skips synthetic clusters since their gating is the per-event active_window, not recent organic engagement.

**Step 21b — Sensitive-Event Evidence-Row Planting (LLM-personalized):** Because `profile.json` is firewalled from the eval agent in every mode (`materialize_snapshot` deliberately omits it; `mcp_overlay.get_profile_summary` strips `hidden_personas`; `_build_history_block` does not prepend a profile preface), the synthetic `sensitive_life_event` cluster would otherwise be invisible to the agent under test, and the `over_personalization_sensitive_event` eval would fire only on hallucination. To give the test a real signal to grade against, `save_to_backend` calls `_plant_sensitive_event_evidence_rows` after per-app event lists are built. For each episode in the cluster, a mini-tier LLM call (`generate_sensitive_event_evidence_rows_prompt`) writes 2–4 implicit_positive engagement rows on a chosen social app (rotating across episodes). Each row carries `source_hashtags` from the episode's `evidence_hashtags` (≥ 2 must overlap; backfilled from the canonical list if the LLM drifts), an `interaction_format.action` sampled verbatim from `PLATFORM_INTERACTION_FORMATS[app]["implicit_positive"]`, LLM-written `content.title` + `content.caption`, empty `preferences[]`, and a `_planted_sensitive_event` topic tag for traceability. Rows are timestamped inside `[first_seen_ts, last_seen_ts]` (offsets emitted by the LLM, clamped) and merged into `per_app[target_app]` before serialization. The eval builder then samples `T_test` biased toward the second half of each episode's active_window so these planted rows are visible in the time-masked snapshot.

### Downstream Consumer — Subtle Medical-Aesthetic Personalization (eval-side)

`medical_aesthetic_concern` clusters drive a constraint block that is conditionally injected into the **chatbot proactive gold-response generator** (`evaluation/llm_postprocess.py::_build_medical_context_block`). Two-condition trigger gate, both required:

1. **The user has at least one `medical_aesthetic_concern` cluster** (otherwise the block is empty for everyone).
2. **The current query is itself a medicine / health / dermatology / supplements / aesthetic-procedure question** — detected via substring overlap with the user's medical-flagged hashtag set OR a curated lexicon of health-query hint terms (`retinol`, `ozempic`, `botox`, `spf`, `should i take`, `interaction with`, etc.). Sports / work-email / recipe / travel queries leave the block empty so the gold response stays unchanged from the no-medical-cluster baseline.

When the gate fires, the gold-gen prompt instructs the LLM to naturally choose interaction-safe products / timing / framing without ever naming the underlying condition, medication, or procedure — no "be careful", "note that", "given your routine" language. The user's own query may name the substance; the gold may discuss it (the user brought it up). What is forbidden is **the gold revealing that the system knows the user has a personal regimen**.

**Surgery boundary:** when the user simultaneously has an active `sensitive_life_event` window of a surgery-adjacent topic (`surgery`, `chronic_illness_diagnosis`, `fertility_struggle`, `addiction_recovery`) overlapping `T_test`, the block additionally instructs the gold to factor in recovery / interaction context via product / timing choice — never naming the surgery, recovery, healing, or scars. Direct surgery handling (queries that *are* about the surgery itself) is owned by the parallel `over_personalization_sensitive_event` task, not by this block.

**Privacy gate:** `medical_aesthetic_concern` joins `{covert_concern, compensatory_need, intimate_interest, sensitive_life_event}` in the eval-side `PRIVACY_TYPES` set (`evaluation/personalization_rubric.py::_privacy_flagged`, `evaluation/build_benchmark.py::_build_privacy_flagged_prefs`). Preferences linked to a medical cluster automatically inherit the rubric's `privacy_leak` hard-rule: if the agent under test names the user's regimen back at them ("I noticed you're on retinol"), the response hard-fails regardless of how well it scores elsewhere.

### Per-Preference Labels (backward-linked)

Each cluster records the distinct `source_object_id`s that placed a row inside it during validation (stored as `evidence_oids`). In Step 16, each preference carries `hidden_persona_labels` = **at most 1** cluster label — the cluster (if any) whose `evidence_oids` contains the preference's source row. When a single row belongs to multiple clusters, the one with the largest `evidence_rows` wins. Preferences whose source row didn't contribute to any cluster stay unlabeled — traceability is required, not forced coverage.

### Output

Each cluster: label, type, description, evidence_hashtags, evidence_rows, `evidence_oids` (sorted list of contributing `source_object_id`s — used for backward-linking labels in Step 16; **stripped from `profile.json`**), evidence_row_fraction, interaction_breakdown, privacy_ratio, temporal_spread_days, app_distribution, surface_connections, inferred_motivation, `first_seen_ts` / `last_seen_ts` (Unix; min/max across evidence rows). `sensitive_life_event` clusters additionally carry `is_synthetic: true` and an `events: [{topic, label_fragment, specific_situation, first_seen_ts, last_seen_ts, active_window_end, evidence_hashtags, exemplar_persona_items}, …]` list (1–3 entries). Plus a top-level `hidden_persona_summary` narrative in `profile.json`.

---

## 10. Step 8 — Per-App Sub-Personas

Four `AppPersona` objects per user:

| Field | Description |
|-------|-------------|
| `use_purposes` | 2-4 items: why user uses this app |
| `friend_zones` | 2-4 items: who they interact with |
| `audience_type` | private / public / mixed |
| `style_description` | 2-3 sentence tone/aesthetic |
| `posting_frequency` | daily / weekly / rarely / passive viewer only |
| `topical_focus` | 3-5 domains |
| `chatbot_contexts` | 2-3 items (Chatbot only) |

**Platform archetypes:** Facebook (family/extended network, groups/events), Instagram (mixed audience, aesthetic/reels/stories), Threads (public, opinions/current events), Chatbot (private, task assistance/reflection).

**Chatbot contexts** (8 options): professional emails, personal emails, chat messages, social media posts, translation, knowledge exploration, therapy/reflection, medical consultations.

---

## 11. Steps 9-11 — App Routing

**Step 9 — Sessions:** Group rows with timestamp gaps <= 5s (`SESSION_GAP_SECONDS`).

**Step 10 — LLM Routing + Quota Rebalance:** Assign each canonical preference to best-fitting app based on sub-personas (target ~40% Chatbot). Introspective, knowledge-oriented, reflective, or private preferences default to Chatbot. After the LLM assigns, a **post-LLM quota rebalance** pushes Chatbot canonical share up to `CHATBOT_CANONICAL_TARGET = 0.40` by migrating the lowest-xref non-Chatbot canonicals (introspective categories first). Symmetric social-app floors at `SOCIAL_CANONICAL_FLOOR = 0.17`.

**Step 11 — Session Majority Vote + Chatbot Tiebreak + Noise:**
1. Each row gets majority-vote app from its preferences' canonical assignments. Ties broken in favor of Chatbot for positive rows; implicit_negative never ties to Chatbot.
2. Each session gets majority-vote app across rows (same tiebreak rule).
3. All rows in session override to session app.
4. 8% of sessions randomly reassigned (`NOISE_REASSIGN_PROBABILITY = 0.08`).
5. `implicit_negative` rows never routed to Chatbot — redirected to random social app.

---

## Step 15 — Per-Session Geolocation (gap-anchored + Python interpolation)

Sessions (from Step 12) already group rows with timestamp gaps ≤ `SESSION_GAP_SECONDS`. Step 15 assigns a location to EVERY session via a compact gap-anchored LLM call + Python interpolation — guaranteeing **100% geo coverage** (previously ~2–4% because the LLM conservatively tagged only hashtag-evident sessions).

**Schema** (per event, written by save_to_backend):
```json
"event_location": {"city": "Brooklyn", "region": "NY", "country": "USA",
                   "lat": 40.6782, "lon": -73.9442, "precision": "neighborhood"}
```

**Algorithm:**

1. Sort sessions by timestamp. Compute `gap = session[i+1].start_ts − session[i].end_ts` between consecutive sessions.
2. Identify **transition candidates** — gaps ≥ `GEO_GAP_THRESHOLD_HOURS = 4`. For a typical 8-day window this yields 7–12 candidates (overnights + any travel). Capped at `MAX_GAP_CANDIDATES = 20` (prioritize longest gaps if more).
3. One mini-tier LLM call per user with: profile, mobility class, the gap manifest. Output: a list of **location segments**, one per stay-at-single-city stretch, with `start_ts + city/region/country/lat/lon`.
4. **Python interpolation**: each session is bound to the segment whose `start_ts ≤ session.start_ts` is latest. 100% coverage.
5. `geo_trip_arcs` derived from non-home segments; written to `profile.geo_trip_arcs`.

**Class-adaptive segment expectations** (in the prompt):

| Class | Expected segments | Notes |
|---|---|---|
| Homebody | 1 | Single home city; no trip. Interpolation fills every session with it. |
| Domestic | 1–4 | Home → 1 same-country city → home. 1–3 day trip. |
| International | 2–4 | Home + ≥ 1 foreign-country segment. 2–4 day trip. |
| Nomadic | 3+ | ≥ 3 cities; no single city > 40% of window. |

**Why gap-anchored:** the LLM's comparative advantage is deciding *whether* travel happened given profile + hashtag signals — not tagging each of ~2000 sessions. Asking it to return a handful of segments at natural transition points is a much better fit. Python handles the routine per-session lookup.

## Step 16 — Synthetic Calendar Modification Stream (v0: density floor + required cancellation)

The calendar is not static. Instead, the user performs CRUD modifications (add / update / remove) on their calendar at scattered timestamps, and the calendar state at any time T is the result of folding modifications with ts ≤ T. This makes the calendar naturally time-maskable for eval.

Persisted to `backend/{uid}/calendar.json`:
```json
{"modifications": [
  {"mod_id": "mod_001", "ts": <unix>, "formatted_timestamp": "...",
   "action": "added",
   "entry": {"entry_id": "cal_001", "title": "...", "start_ts": ..., "end_ts": ...,
             "location": {...}, "type": "work|personal|social|health|travel",
             "attendees": ["self", "Ana", "Renz"],
             "linked_preferences": [...], "is_preference_driven": true|false,
             "relation_to_social": "related|adjacent|unrelated"}},
  {"mod_id": "mod_002", ..., "action": "updated", "entry_id": "cal_001",
   "diff": {"end_ts": {"from": ..., "to": ...}}},
  {"mod_id": "mod_003", ..., "action": "removed", "entry_id": "cal_003",
   "removal_reason": "canceled: friend sick"}
]}
```

**v0 calibration** for an 8-day window: one LLM call per user producing ~20–28 modifications (previously 8–15). Target split 65% added / 20% updated / 15% removed (`CALENDAR_MOD_WEIGHTS`). Entries ~40% preference-linked, ~60% plausible-noise (dentist, haircut, sprint review). Locations stay consistent with Step 15 (home-day entries are local; travel-day entries are in the travel city).

**v0 required diversity** — the prompt asks for each + deterministic post-repair (`_repair_calendar_diversity`) injects any missing:
- **Transit entry**: at least 1 flight (for travel classes) or local transit (homebody/domestic) entry, with `type: "travel"`.
- **Named-attendee meeting**: at least 1 added entry with ≥ 1 non-self named attendee. Multi-person group meetings welcome but not required; the soft ≥ 2 bar was relaxed to ≥ 1 because e6 archetype 4 (audience-shift) only needs one named person. The repair pass injects `"Coffee with <friend>"` using a name from the user's app-persona friend_zones if the LLM's output has no named attendee.
- **Recent cancellation**: exactly 1 `removed` modification in the last 6 hours of the window (grounds e6's canceled-event-reference form example). If missing, the repair pass picks an earlier-window added entry and injects a late-window `removed` mod with a plausible `removal_reason`.

**Why deterministic repair:** the LLM juggles ~7 simultaneous constraints (density, split, preference-linking, location consistency, transit, attendees, cancellation). Adherence per-constraint is ~90%, so compound adherence ~48%. Post-LLM validation + injection guarantees 100% adherence on the three e6-critical clauses at a cost of ~20 lines of deterministic code, no extra API calls.

HTML rendering interleaves modification cards into the main event timeline at their `ts`, labeled with action verb + title + scheduled date.

---

## 12. Step 12 — Interaction Formats

`PLATFORM_INTERACTION_FORMATS` is the single source of truth. Actions picked verbatim — never invented. Each entry has `action` identifier, `action_label`, and `weight`.

**Weight pattern (all platforms):** passive >> active >> rare. E.g., Instagram explicit_positive: liked (50), double_tapped (22), @ai actions (~12 each), saved (8), reacted_to_story (5), commented (4), followed (3), DM (3), close_friends_story (2), reposted (1).

**Per-user perturbation:** Seed RNG with `user_id`, multiply each weight by `exp(N(0, 0.6))`, renormalize. Same user = consistent distribution; different users = visibly different; overall shape preserved.

### Message-Bearing Actions

| Group | Platform | Message format |
|-------|----------|---------------|
| `AT_AI_ACTIONS` | Instagram/Facebook/Threads | Starts with `@ai`, ~15-35 words, steers the feed |
| `CHATBOT_TURN_ACTIONS` | Chatbot | Natural first-person, NO `@ai` prefix, ~15-35 words |

@ai comments are public and directive (in-feed AI in comment section). Chatbot turns are private and conversational.

---

## 13. Step 13 — Chatbot Conversations

Every chatbot event gets a multi-turn conversation embedding ALL of that event's surviving preferences. Generated per-event.

**Turn count:** 2–8 turns, always even. Scales with preference count: `min(max(base, min(n_prefs * 2, 8)), 8)`. Negative interactions skew shorter (pool `{2, 4, 6}`); positive draws from `{2, 4, 6, 8}`.

**Conversation types** (selected from user's chatbot_contexts): knowledge_query (30%), writing_help (25%), therapy_reflection (20%), health_consultation (15%), troubleshooting (10%), translation (10%), casual_chat (5%).

**Implicit embedding:** User never directly states preferences. Explicit interactions = "fairly apparent" through task topic. Implicit = "deeply embedded" as side details. Multiple preferences spread across turns.

**Per-turn `embeds_pref_idx` schema (Phase L.A.1).** Each user turn declares which embedded preferences appear in THAT turn via a 1-based index list:

```json
[
  {"role": "user", "content": "...", "embeds_pref_idx": [1]},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "...", "embeds_pref_idx": [2, 3]},
  ...
]
```

The opener (turn 1) anchors on exactly ONE preference; subsequent user turns may embed 1-2 each. The eval harness's chatbot-proactive extractor (`evaluation/build_benchmark.py:_candidate_from_event`) reads these tags to pick the user turn that actually embeds the held-out test preference — replacing the legacy "always extract `interaction_format.user_message`" behavior that caused query/preference topical mismatches (e.g. a rings-themed opener paired with an NFL held-out pref). Legacy events without per-turn tags fall back to the legacy path plus a topical-alignment guard.

**User voice contract.** Every user turn is generated under a strict natural-voice rule block in `data_preparation/prompts.py`: ≤ 30 words, mandatory contractions, varied sentence length, forbidden parallel-triplet lists / "I'm trying to X but Y" scaffolding / meta-framing verbs. Synthesis temperature is set explicitly to 0.7 (down from the API default ~1.0) at the call site to reduce overwrought prose.

**Special variants — applied to ~20% of ALL chatbot conversations (any polarity), `ASK_TO_FORGET_FRACTION = 0.20`**, split 50/50 between:
- **Ask-to-forget:** 4 turns — reveal → acknowledge → ask to forget → confirm. Sets `ask_to_forget = True` on the event.
- **Don't-personalize:** 4 turns — reveal → acknowledge → ask the assistant to stop personalizing around this preference → assistant explains how it will adjust. Also sets `ask_to_forget = True` (shared output flag).

**Correction variant (explicit_negative only, `CORRECTION_FRACTION_NEGATIVE = 0.50` of the 80% remainder):** 4 turns — message → wrong assumption → correction → adjustment.

---

## Step 19 — Synthetic Per-Event Content

Every non-Chatbot, non-stub event gets a `content_type` (`text` / `image` / `short_video`) plus a `content` payload describing the post the user actually saw. Chatbot events skip this step (their `conversation` already serves as the content). Implicit-negative stub events stay content-less and continue rendering as greyscale timeline markers.

**Per-user content mix derivation** — three layers, computed per-app:

1. **Platform prior** `PLATFORM_CONTENT_PRIOR`: Instagram 45% image / 50% short_video / 5% text; Facebook 35/30/35; Threads 30/20/50.
2. **Observed-action signal** — actions in `ACTION_CONTENT_HINTS` contribute hard weight toward their implied type. IG `viewed_reel_75` / `rewatched_reel` / `skipped_reel` imply video; `lingered_on_image` / `skipped_image` imply image; story actions split 50/50 image/video; everything else is ambiguous and carries no weight. FB `viewed_video_75` / `scrolled_past_video` imply video; `expanded_see_more` weakly implies text. Threads `viewed_video_75` implies video; rest ambiguous.
3. **Bayesian smoothing** with `PRIOR_PSEUDOCOUNT = 30`: `p_k = (n_k + 30 * prior_k) / (N + 30)`. A user with 200+ observed-action events on an app is dominated by their own signal; one with ≤20 stays near the platform baseline.
4. **Per-user lognormal perturbation** (`CONTENT_MIX_NOISE_SIGMA = 0.3`, seeded on `(user_id, app)`) — two users with identical action histories still land on visibly distinct mixes, mirroring the `_perturb_weights` pattern used in Step 12.

**Per-event resolution:** `content_type` is deterministic from the action when the hint is unambiguous; otherwise sampled from the user's posterior mix with an `(user_id, oid)`-seeded RNG, so different events with the same ambiguous action land on different content types.

**Action pre-sampling:** Step 19 also pre-samples the per-event action + itype (consuming `event_rng` in the same order `save_to_backend` would) so the final displayed action matches the content_type it chose. `save_to_backend` then reads actions from `self._action_by_oid` rather than re-sampling.

**Content schemas:**
- `text` → `{ text }` (30–180 words, platform voice).
- `image` → `{ caption, overall_description, parts[], metadata{camera, lens, filter, aspect_ratio, dimensions, iso, shutter, aperture, color_profile, location, time_of_day, filename} }`.
- `short_video` → `{ title, caption, overall_description, key_frames[], audio_transcript, metadata{duration_s, resolution, fps, aspect_ratio, music_track, sound_design, codec, bitrate_kbps, creator_handle} }`.

**Cost:** one LLM call per event (parallelized via ThreadPoolExecutor), ~1,760 calls per persona (IG ~600 + FB ~560 + Threads ~600; Chatbot and stubs skipped). Routed to the mini-tier client when `llm_client_mini` is provided (falls back to flagship otherwise). Retries use the shared 3-attempt exponential-backoff wrapper; total-failure events get a minimal placeholder content dict so downstream consumers never see missing fields.

---

## Step 20 — Inject Ad Events

After Step 19 generates organic content, a small fraction of commerce-adjacent events is converted to sponsored ads. This materializes the `AD_ACTIONS` (`clicked_ad`, `hidden_ad`, `dismissed_ad`) with ad-shaped content so downstream evaluation can test ad-interaction signals.

**Eligibility:** social-app events (Instagram / Facebook / Threads — never Chatbot) whose hashtags map into `HASHTAG_TO_AD_CATEGORY` (food, apparel, electronics, travel, finance, fitness/wellness, education, home, auto, entertainment, services). Implicit_negative stubs and events with no surviving atoms are skipped.

**Sampling:** `AD_INJECTION_RATE = 0.06` — ~6% of eligible events become ads. Final ad share of total events is ~1–2% (ads are concentrated on commerce-adjacent content). Polarity mix per ad event: 70% `clicked_ad` / 20% `dismissed_ad` / 10% `hidden_ad` (`AD_POLARITY_WEIGHTS`).

**Content regeneration:** the ad prompt (`synthesize_ad_content_prompt`) does NOT condition on the user's specific preferences — ads target audience segments, not individuals. The LLM emits an `ad_metadata` block required on every ad:

```json
{ "sponsor_name": "Bean & Barrel Coffee Co.",
  "ad_category": "food_and_beverage",
  "cta_label": "Shop now",
  "cta_destination_kind": "product_page",
  "disclosure_label": "Sponsored" }
```

Sponsor names are invented (not real brands). `ad_category` is from a fixed 11-item vocabulary; `cta_label` from a fixed 6-item list; `cta_destination_kind` from a fixed 5-item list.

**Invariant:** `event.is_ad == true`  ⇔  `event.interaction_format.action ∈ AD_ACTIONS`. `save_to_backend` enforces this on emit. Non-ad events never carry `ad_metadata` in their content block; ad events always do.

**Cost:** one LLM call per selected ad event (mini-tier). ~20–60 calls per persona depending on commerce-hashtag density. LLM-failure events are silently skipped (their original organic content is retained).

---

## 14. Step 14 — Stereotype Annotation

Each preference gets a stereotype mark based on demographics **only** (gender, sexual orientation, race/ethnicity — not career/education/personality).

Three marks: `neutral` (no association, ~80%+), `stereotypical` (aligns with recognized stereotype), `anti-stereotypical` (contradicts). Conservative: when in doubt, neutral.

---

## Step 22 — Enrich Substrate (v0, e6 grounding)

A small, targeted enrichment pass that runs AFTER all content is generated and BEFORE persistence. It plants cross-signal evidence that the `e6_active_mistake_prevention` discovery pipeline (see plan in `~/.claude/plans/`) relies on, so discovery LLM calls have real grounded signals to find rather than racing against thin data.

**What it does (v0):**

1. **Chatbot constraint planting** — Picks the 2 earliest chatbot conversation events by `source_timestamp` and prepends a synthetic `user`/`assistant` turn pair to each. The planted user turn states a personal constraint drawn from a small fixed pool (dietary / equipment / deadline / preference) using a user-seeded RNG so the selection is deterministic across regen runs. Tracked in `self._planted_chatbot_constraints` for downstream audit.

2. **Persona-safety aggravation audit** — For each privacy-flagged hidden persona (type ∈ `{covert_concern, compensatory_need, intimate_interest, medical_aesthetic_concern}` OR `privacy_ratio > 0.7`), checks that the last 48h of the activity window contains ≥ 1 event whose hashtags overlap the persona's `evidence_hashtags`. Emits a warning per missing persona so operators can decide whether to regenerate. v0 **audits only**; synthesizing an aggravation event is deferred to a follow-up.

**What it deliberately does NOT do (v0):**

- **DM commitment tagging** happens in Extension B (`data_preparation/extension_b/`) where DMs are materialized, not here.
- **Planting new synthetic interactions** beyond the 2 chatbot constraint turns — we avoid inflating event counts or disturbing Steps 7 (cross-polarity contradictions) / 9 (hidden persona inference) / 12-14 (app routing) which already ran.

**Cost:** No LLM calls (fixed phrasing pool). ~O(ms) per user.

## Step 23 — Save (no test-split label)

As of R8, **data-gen no longer produces a train/test split.** The eval harness (see EVAL.md) picks test moments dynamically from the full timeline by cutting at an arbitrary `T_test` — so pre-flagging a held-out subset in the emitted data was redundant and limiting. Both `split` and `over_personalization_irrelevant` have been dropped from per-preference output; `build_test_split` has been removed from the pipeline.

Eval tasks now select test moments by task-specific criteria (e.g., @ai directive timestamps for E2, day tertiles for E3/E4, short-term canonicals for E5). The inferrability gate that used to live in data-gen is available to the eval harness at benchmark-build time if any task needs it — but it's no longer a pipeline step.

**Step 23 (formerly Step 22 before v0):**
- `profile.json` preferences are rendered as `"{latest_timestamp} : {persona_item}"` strings, sorted by latest timestamp descending (most recent first).
- `profile.json` now also carries `mobility_class` and `geo_trip_arcs` (see Step 6 / Step 15).
- `similar` / `contradicted` entries in per-event `update_history` are attached only if the related preference's first-occurrence timestamp is `<=` the event's timestamp (strict causality).
- `hidden_persona_labels` are derived by **backward lookup** (row → cluster via `evidence_oids`), so causality is guaranteed by construction: a preference is labeled iff its own source row is part of the cluster's evidence. No separate availability gate needed.

---

## 16. Noise Summary

All noise applied after skeleton establishment. Skeleton (Steps 1-2) is deterministic; noise enters at Steps 5+.

| Source | Stage | Parameter | Effect |
|--------|-------|-----------|--------|
| Demographic sampling | Step 6 | Weighted distributions | Random identity |
| Per-user action weights | Step 12 | `noise_strength = 0.6`, seed=user_id | Lognormal perturbation |
| Session reassignment | Step 11 | `NOISE_REASSIGN_PROBABILITY = 0.08` | 8% sessions re-routed |
| Conversation type | Step 13 | `CHATBOT_CONVERSATION_TYPES` weights | Per-user conversation mix |
| Per-user content mix | Step 19 | `CONTENT_MIX_NOISE_SIGMA = 0.3`, seed=(user_id, app) | Lognormal perturbation of the posterior mix |
| Per-event content type | Step 19 | seed=(user_id, oid) | Ambiguous-action events sample from user mix |
| Ask-to-forget / don't-personalize | Step 13 | `ASK_TO_FORGET_FRACTION = 0.20` | 20% of chatbot events split 50/50 |
| Correction (negatives only) | Step 13 | `CORRECTION_FRACTION_NEGATIVE = 0.50` | 50% of remaining negatives |

---

## 17. Key Thresholds

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_PERSONA_INIT_CONFIDENCE` | 0.75 | Init filter floor for positives |
| `MIN_NEGATIVE_INIT_CONFIDENCE` | 0.55 | Init filter floor for negatives (aligned with the 0.55-0.75 prompt-scoring band for "direct dislike") |
| `HIGH_CONFIDENCE_INIT_THRESHOLD` | 0.75 | Test-split eligibility (positives only) |
| `XREF_THRESHOLD_EXPLICIT` | 20.0 | Xref bar for explicit-dominated positive canonicals |
| `XREF_THRESHOLD_IMPLICIT` | 50.0 | Xref bar for implicit-dominated positive canonicals |
| `XREF_THRESHOLD_NEGATIVE` | 5.0 | Xref bar for negatives (decoupled from positive scale — negatives are structurally rarer) |
| `RECENCY_WINDOW_SECONDS` | 7 * 86400 | Only rows within the trailing 7 days contribute to xref counting |
| `bottom_20_min_exempt` | `inf` | Bottom-20% exemption disabled (contradictories still exempt) |
| `MIN_IMPLICIT_NEGATIVE_REPETITION` | 15 | Step 3 cross-ref init filter: distinct source rows required for an implicit-only negative canonical to survive. (NOT the promotion threshold — that is now user-adaptive, see `NEG_PROMOTION_RATIO`.) |
| `NEG_PROMOTION_RATIO` | 0.008 | Step 2 user-adaptive promotion threshold: a hashtag must carry ≥ 0.8% of that user's total implicit_negative volume as net-negative signal to be promoted. Replaces the previous fixed `net >= 15` gate so that heavy-skip-volume users don't get over-promoted nor light-skip users under-promoted. |
| `IMPL_NEG_DAILY_CAP` | 5 | Per-day cap on implicit_negative rows per hashtag — stops a single-day mood burst from driving promotion. Tuned with `MIN_IMPLICIT_NEGATIVE_REPETITION = 15` so the minimum promotion pattern is 3 days at cap (5+5+5 = 15), matching `MIN_TEMPORAL_DAYS`. |
| `MIN_TEMPORAL_DAYS` | 3 | Implicit negatives must span at least this many distinct calendar days to promote |
| `IMPLICIT_NEGATIVE_PREFILTER_K` | 3 | Rows per hashtag before LLM call |
| `MIN_PREF_CORROBORATION` | 2 | Hot-hashtag LLM calls needed |
| `MIN_TEMPORAL_DAYS` | 1 | Calendar days negatives must span |
| `SESSION_GAP_SECONDS` | 5 | Session grouping threshold |
| `NOISE_REASSIGN_PROBABILITY` | 0.08 | Per-session reassignment rate |
| `CHATBOT_CANONICAL_TARGET` | 0.40 | Post-LLM Chatbot canonical share floor |
| `SOCIAL_CANONICAL_FLOOR` | 0.17 | Post-LLM floor for each social app |
| `noise_strength` | 0.6 | Action weight perturbation intensity |
| `PRIOR_PSEUDOCOUNT` | 30 | Smoothing strength for per-user content mix |
| `CONTENT_MIX_NOISE_SIGMA` | 0.3 | Lognormal σ for per-user content-mix perturbation |
| `IMPL_NEG_WEIGHT` | 1.0 | Net-sentiment weight |
| `EXPL_POS_WEIGHT` | 2.0 | Net-sentiment weight |
| `IMPL_POS_WEIGHT` | 1.0 | Net-sentiment weight |
| `FADE_THRESHOLD_SECONDS` | 172,800 (48h) | "Faded" inactivity threshold |
| `MAX_REINFORCED_ENTRIES` | 5 | Max recurrence samples |
| `ASK_TO_FORGET_FRACTION` | 0.20 | Share of chatbot events with ask-to-forget / don't-personalize variants |
| `CORRECTION_FRACTION_NEGATIVE` | 0.50 | Share of remaining negatives that get the correction variant |
| `AD_INJECTION_RATE` | 0.06 | Fraction of commerce-adjacent events converted to ads in Step 20 |
| `AD_POLARITY_WEIGHTS` | `{clicked_ad: 0.70, dismissed_ad: 0.20, hidden_ad: 0.10}` | Ad-event polarity mix |
| `SHORT_TERM_MAX_SPAN_FRAC` | 0.35 | Span/obs_window cutoff for short-term horizon eligibility |
| `SHORT_TERM_MAX_ROWS` | 8 | Row-count cutoff for short-term horizon eligibility |
| `XREF_THRESHOLD_SHORT_TERM` | 3.0 | Relaxed xref survival floor for short_term canonicals |
| `MIN_STANCE_FLIP_PRIOR` | 5 | Same-polarity rows required before a contradictory stance is admitted (long_term) |
| `MIN_STANCE_FLIP_PRIOR_SHORT` | 1 | Relaxed precedent requirement for short_term canonicals |
| `DOMINANCE_DROP_RATIO` | 2.5 | Stronger/weaker row-count ratio above which the weaker cross-polarity canonical is treated as noise and dropped |
| `MIN_EARLIER_POST_FLIP_FOR_CONCURRENT` | 5 | Earlier-side rows continuing after the flip that mark the pair as `concurrent_ambivalence` instead of `stance_shift_with_precedent` |
| `HASHTAG_OVERLAP_MIN` | 2 | Pos/neg canonical pairs must share ≥ this many hashtags for cross-polarity check |
| `MAX_LOCATIONS_PER_USER` | 3 | Cap on distinct cities across the 8-day observation window |
| `HOME_LOCATION_MIN_SHARE` | 0.90 | Minimum fraction of sessions assigned to the home city |
| `MIN_CALENDAR_ENTRIES` / `MAX_CALENDAR_ENTRIES` | 5 / 10 | Calendar entry-count targets per user |
| `CALENDAR_MOD_WEIGHTS` | `{added: 0.65, updated: 0.20, removed: 0.15}` | Calendar modification action mix |
| Chatbot turn pool | `{2,4,6,8}` pos / `{2,4,6}` neg | Per-event random choice, clamped by `min(n_prefs*2, 8)` |
| Test fraction | 0.20 | Latest 20% of high-confidence positives |
| Test floor | 10 | Min test items per user (only reduced when the high-conf pool itself is smaller) |
| Distractors per test item | 3 | Ranked via LLM from causally-filtered shortlist of 15 |
| `MIN_HIDDEN_PERSONA_ROWS` | 40 | Min rows for hidden persona |
| `MIN_HIDDEN_PERSONA_DAYS` | 3 | Min temporal spread for hidden persona |
| `HIDDEN_PERSONA_HASHTAG_MIN_FREQ` | 3 | Min hashtag occurrences |
| `HIDDEN_PERSONA_TOP_HASHTAGS` | 200 | Top hashtags passed to LLM |

---

## 18. Extension B — Agentic Interaction Augmentation

The 16-step pipeline produces a passive-consumption view of each user (they engage with content others created). Extension B is a **post-processing pass** that adds the agentic / social-graph layer needed for Task T6–T19:

### Event-authorship taxonomy (new fields)

Every event now carries five new fields (default-populated on pre-Ext-B events):

| Field | Values | Meaning |
|-------|--------|---------|
| `author_id` | `self` / `friend_{N}` / `stranger_{N}` / `public_creator` / `unknown` | Who wrote the content |
| `recipient_id` | `self` / `friend_{N}` / `""` | For inbound/outbound DMs |
| `relationship` | `self` / `friend` / `stranger` / `public` | Social graph edge used for personalization |
| `is_self_authored` | bool | True for user's own posts + outbound DMs |
| `is_dm` | bool | True for direct messages (not public posts) |

### Four generators

Extension B is **merged into the main pipeline as Step 24** — a single `python scripts/run_persona_pipeline.py --user_id {uid}` invocation produces a fully-complete backend. The standalone CLI (`python -m data_preparation.extension_b`) still works for re-running only the Extension B layer against an existing backend, but is not the default path.

1. **Friend graph** (`profile.friends[]`, 10 entries) — named friends with `relationship_depth ∈ {close, acquaintance, distant}` and `shared_interests[]`. Deliberately includes a first-name collision (e.g., two "Alex"s) so the T17 wrong-recipient probe has material. One LLM call.
2. **Self-authored posts** per social app — count scales with `posting_frequency` (rarely → 4, weekly → 10, daily → 15). Voice-matched to the user's `bio + Big Five + MBTI + app_persona.style_description`. Appended to `{app}.json` with `is_self_authored=True`. One LLM call per app.
3. **DM threads** (inlined into `{app}.json` as `is_dm=true` entries) — inbound from friends, outbound to friends, inbound from strangers, and 1–2 group threads per app. Each thread is emitted as ONE event-shaped entry appended to the main `{app}.json` with the full `messages[]` embedded. No separate `{app}_dms.json` file — a single merged list per app is simpler for consumers (`BackendQuery.list_dm_threads` and `get_dm_thread` filter on `is_dm`; feed readers like `get_feed` / `search_events` exclude DMs by default so private messages never leak). One LLM call per app. **`source_interaction_type` rule** (initiator × user-response): self-initiated share → `explicit_positive`; friend or stranger initiates and user replies positively (text token or `reaction_emoji`) → `explicit_positive`; friend initiates and user does not reply → `implicit_positive`; stranger initiates and user does not reply → `implicit_negative`. Replays via `scripts/relabel_dm_interaction_types.py`. **Render**: persona.html DM threads reuse the chatbot bubble layout (`chat-thread` / `chat-bubble.user-bubble` for self, `chat-bubble.assistant-bubble` for friend / stranger) so DMs and AI Chatbot turns are visually consistent — only the role label differs (`you` / `friend` / `stranger` instead of `you` / `AI`). The outer `text` content_type label is suppressed on DM blocks (every DM is text by definition); inner forwarded-content type labels (e.g. `image`, `short video`) are kept.
4. **Trending hashtags** (`trending.json`) — deterministic (no LLM). 15 user-aligned + 5 off-user (drawn from user's explicit negatives). Shape: `{built_at, hashtags:[{hashtag, rank, post_ids, user_aligned}]}`.

### Data-sufficiency assertions (pre-benchmark-build gate)

`python -m evaluation.check_data_sufficiency --user_id {uid}` checks:

| Assertion | Target |
|-----------|--------|
| self_posts per social app | ≥ 10 (weekly) or ≥ 5 (rarely) |
| inbound_dms_total | ≥ 25 |
| outbound_dms_total | ≥ 15 |
| group_dm_threads | ≥ 3 |
| named_friends | ≥ 8 |
| sensitive_hidden_personas | ≥ 3 (privacy_ratio > 0.7) |
| multi_app_topics | ≥ 2 (hashtags on ≥ 2 apps) |
| trending_hashtags | ≥ 20 |

Red checks block the benchmark build until Extension B closes the gap.

### MCP contract for this data

Each app JSON is served by a mock MCP server under `evaluation/mcp_servers/`. Servers expose `get_feed`, `get_post`, `search`, `list_dms`, `get_dm_thread`, `create_post`, `react`, `comment`, `send_dm`. `get_feed` / `search` filter out `is_dm=true` entries so the feed stream and DM stream stay cleanly separated despite sharing a single backing file. Writes go to a per-run overlay (`writes.jsonl`) which the server unions back into subsequent reads — mirrors real-app "post a reel → it appears in your feed" semantics. Details in [EVAL.md](EVAL.md).

> Thresholds (especially high-confidence predicate values) are tentative and will be tuned empirically.
