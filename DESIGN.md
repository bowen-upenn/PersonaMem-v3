# PersonaMem-v3: Design Document

*Towards All-Day-Long Omni-Platform Personal Intelligence*

> Design rationale for the PersonaMem-v3 data generation pipeline. For implementation, see `persona_agent.py`, `prompts.py`, and `skill.md`.

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
  +- Step 1:  Infer atomic personas           [LLM]      -- 1-3 per row, init 0.0-1.0
  +- Step 2:  Promote implicit negatives       [Algo+LLM] -- net-sentiment gate
  +- Step 3:  Cross-reference & filter         [Algo+LLM] -- cross_ref scores (uncapped)
  +- Step 4:  Temporal contradiction graph     [LLM]      -- timeline grouping
  +- Step 5:  Build update histories           [Algo+LLM] -- reinforced/faded/evolved
  +- Step 6:  Generate user profile            [LLM]      -- demographics + Big Five
  +- Step 7:  Infer hidden personas            [Algo+LLM] -- cross-row hashtag clustering
  +- Step 7b: Infer MBTI                       [LLM]      -- type + per-dimension probabilities
  +- Step 8:  Generate per-app sub-personas    [LLM]      -- 4 AppPersonas
  +- Step 9:  Build sessions                   [Algo]     -- temporal grouping
  +- Step 10: Route preferences to apps        [LLM+Algo] -- ~40/20/20/20 distribution
  +- Step 11: Assign rows to apps              [Algo]     -- session majority vote + 8% noise
  +- Step 12: Generate interaction formats     [Algo+LLM] -- per-user perturbed weights
  +- Step 13: Generate chatbot conversations   [LLM]      -- multi-turn, ask-to-forget
  +- Step 13b: Generate synthetic content      [LLM]      -- text / image / short_video per event
  +- Step 14: Annotate stereotype marks        [LLM]      -- demographics-only
  +- Step 15: Build test split                 [LLM+Algo] -- newest-first ≥10, inferrability-labelled
  +- Step 16: Save to backend                  [Algo]     -- 5 JSON files per user
```

**Model tiers:** the pipeline uses two LLM clients. The **flagship** model (`gpt-5-chat`) handles reasoning-heavy steps — 1 (atomic persona), 3-6 (cross-ref, temporal, histories, profile), 7b/7c (hidden personas + summary), 7b-MBTI, 8 (app personas), 13 (chatbot conversations), 15 (train/test split). The **mini** model (`gpt-5.4-mini`, configurable via `--mini_model`) handles mechanical steps — 7a (intimate-hashtag detection), 10 (app routing), 12 (interaction formats), 13b (synthetic content), 14 (stereotype marks). Mini falls back to flagship when no mini client is configured.

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
  |     +- over_personalization_irrelevant (test items only — list of 3 {persona_item, category} distractors, ranked most- to least-jarring)
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

## 5. Step 2 — Implicit Negative Promotion

`implicit_negative` rows are skipped in Step 1 (a single scroll-past is too weak). Instead, aggregated via net-sentiment filtering to distinguish genuine dislike from baseline scrolling.

**Net-sentiment formula** per hashtag:
```
net = (implicit_neg_count x 1.0) - (explicit_pos_count x 2.0) - (implicit_pos_count x 1.0)
```

**Promotion gate** — both must hold:
1. `net >= 5` (`MIN_IMPLICIT_NEGATIVE_REPETITION`)
2. Negative rows span >= 1 distinct calendar day (`MIN_TEMPORAL_DAYS`)

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

---

## 7. Steps 4-5 — Temporal Evolution

**Step 4 — Contradiction Graph:** Contradictory preferences grouped by topic with chronological timelines showing stance shifts.

**Step 5 — Update Histories:** Each preference gets a temporal `update_history[]` array with entries tagged by `update_type`:

| `update_type` | Definition |
|----------------|-----------|
| `new` | First appearance (filtered from serialization — redundant with event timestamp) |
| `reinforced` | Multiple distinct source rows; up to 5 samples, evenly spaced |
| `faded` | Inactive > 48h before user's last activity (`FADE_THRESHOLD_SECONDS = 172,800`) |
| `contradicted` | Contradicting preference discovered in cross-ref |
| `deepened` | General interest became more specific over time |
| `branched` | Interest expanded into new sub-direction |
| `shifted` | Focus moved within same domain |
| `intensified` | Engagement grew demonstrably stronger |
| `similar` | Semantically similar preference discovered in cross-ref |

Entry fields: `update_type` (all), `preference` (contradicted/deepened/branched/shifted/similar), `formatted_timestamp` (all), `source_app` (reinforced/deepened/branched/shifted/similar), `occurrence`+`total_occurrences` (reinforced), `description` (deepened/branched/shifted/intensified).

**Causality filter:** only entries with `timestamp <= event timestamp` are included — no knowledge leakage.

---

## 8. Step 6 — Synthetic User Profile

Demographics sampled first; everything downstream (name, career, bio) must be consistent.

**Gender x Orientation** (21 entries, key ones): Cis female hetero 30%, Cis male hetero 32%, Cis male gay 5%, Cis female bi/lesbian 4% each, Non-binary queer 2%, Trans female/male hetero 2% each, remaining categories 0.5-1% each.

**Race/Ethnicity** (28 entries, key ones): White American 15%, Chinese 10%, Black/African American 8%, Indian 8%, Mexican American 8%, Filipino/Vietnamese/Korean 4% each, Japanese/MENA/Multiracial 3% each, remaining categories 1-2% each. Intentionally diversified beyond census.

**LLM-generated fields:** Name (culturally appropriate), Career (consistent with *some* preferences), Education, Big Five personality (each low/medium/high), Bio (3-5 sentences). LLM instructed to avoid stereotypical demographic-career-hobby combinations.

---

## 9. Step 7 — Hidden Persona Inference

Infers deeper motivational layers (*why* a user engages, not just *what* they like) from cross-row hashtag patterns. Grounded in behavioral science.

### Three-Phase Algorithm

**Phase 1 — Hashtag Census (algo):** Count occurrences, per-type breakdown, distinct days for each hashtag. Filter to >= 3 occurrences (`HIDDEN_PERSONA_HASHTAG_MIN_FREQ`). Pass top ~200 (`HIDDEN_PERSONA_TOP_HASHTAGS`) to LLM.

**Phase 1b — Intimate-Signal Pre-Screen (LLM):** Ask the LLM (via `detect_intimate_hashtags_prompt`) to flag adult/kink/sexually-suggestive hashtags among the user's positive-signal tags. No keyword list lives in code — substring heuristics produce too many false positives (e.g. `cummins`, `hotchicken`, `earthporn`, `nakedchef`, `cheatersexposed`). Flagged hashtags are **force-included in the top-N table** even if their counts fall below `HIDDEN_PERSONA_HASHTAG_MIN_FREQ`, so a single intimate signal cannot be dropped.

**Phase 2 — LLM Clustering:** Groups hashtags into **at most 6** thematic clusters, actively using the user's profile (demographics, career, bio) to ground inference. The prompt flags `intimate_interest` and `covert_concern` as priority signals that must be surfaced whenever hashtag evidence supports them. Ten types:

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

**Phase 3 — Validation:** Each cluster needs >= 40 distinct rows (`MIN_HIDDEN_PERSONA_ROWS`) and >= 3 distinct days (`MIN_HIDDEN_PERSONA_DAYS`). Privacy ratio reported (> 0.7 required for `compensatory_need`). **Exemption:** `intimate_interest` clusters whose evidence overlaps the Phase-1b pre-screened set skip both gates — one positive signal is enough to surface an intimate persona.

**Phase 4 — Deduplication:** Merge hidden personas with Jaccard >= 0.5 on evidence hashtags. Persona with more evidence_rows becomes base; hashtags and surface_connections unioned; metrics recomputed. Repeats until no merges.

### Per-Preference Labels (backward-linked)

Each cluster records the distinct `source_object_id`s that placed a row inside it during validation (stored as `evidence_oids`). In Step 16, each preference carries `hidden_persona_labels` = **at most 1** cluster label — the cluster (if any) whose `evidence_oids` contains the preference's source row. When a single row belongs to multiple clusters, the one with the largest `evidence_rows` wins. Preferences whose source row didn't contribute to any cluster stay unlabeled — traceability is required, not forced coverage.

### Output

Each cluster: label, type, description, evidence_hashtags, evidence_rows, `evidence_oids` (sorted list of contributing `source_object_id`s — used for backward-linking labels in Step 16), evidence_row_fraction, interaction_breakdown, privacy_ratio, temporal_spread_days, app_distribution, surface_connections, inferred_motivation. Plus a top-level `hidden_persona_summary` narrative in `profile.json`.

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

**Special variants — applied to ~20% of ALL chatbot conversations (any polarity), `ASK_TO_FORGET_FRACTION = 0.20`**, split 50/50 between:
- **Ask-to-forget:** 4 turns — reveal → acknowledge → ask to forget → confirm. Sets `ask_to_forget = True` on the event.
- **Don't-personalize:** 4 turns — reveal → acknowledge → ask the assistant to stop personalizing around this preference → assistant explains how it will adjust. Also sets `ask_to_forget = True` (shared output flag).

**Correction variant (explicit_negative only, `CORRECTION_FRACTION_NEGATIVE = 0.50` of the 80% remainder):** 4 turns — message → wrong assumption → correction → adjustment.

---

## 13b. Step 13b — Synthetic Per-Event Content

Every non-Chatbot, non-stub event gets a `content_type` (`text` / `image` / `short_video`) plus a `content` payload describing the post the user actually saw. Chatbot events skip this step (their `conversation` already serves as the content). Implicit-negative stub events stay content-less and continue rendering as greyscale timeline markers.

**Per-user content mix derivation** — three layers, computed per-app:

1. **Platform prior** `PLATFORM_CONTENT_PRIOR`: Instagram 45% image / 50% short_video / 5% text; Facebook 35/30/35; Threads 30/20/50.
2. **Observed-action signal** — actions in `ACTION_CONTENT_HINTS` contribute hard weight toward their implied type. IG `viewed_reel_75` / `rewatched_reel` / `skipped_reel` imply video; `lingered_on_image` / `skipped_image` imply image; story actions split 50/50 image/video; everything else is ambiguous and carries no weight. FB `viewed_video_75` / `scrolled_past_video` imply video; `expanded_see_more` weakly implies text. Threads `viewed_video_75` implies video; rest ambiguous.
3. **Bayesian smoothing** with `PRIOR_PSEUDOCOUNT = 30`: `p_k = (n_k + 30 * prior_k) / (N + 30)`. A user with 200+ observed-action events on an app is dominated by their own signal; one with ≤20 stays near the platform baseline.
4. **Per-user lognormal perturbation** (`CONTENT_MIX_NOISE_SIGMA = 0.3`, seeded on `(user_id, app)`) — two users with identical action histories still land on visibly distinct mixes, mirroring the `_perturb_weights` pattern used in Step 12.

**Per-event resolution:** `content_type` is deterministic from the action when the hint is unambiguous; otherwise sampled from the user's posterior mix with an `(user_id, oid)`-seeded RNG, so different events with the same ambiguous action land on different content types.

**Action pre-sampling:** Step 13b also pre-samples the per-event action + itype (consuming `event_rng` in the same order `save_to_backend` would) so the final displayed action matches the content_type it chose. `save_to_backend` then reads actions from `self._action_by_oid` rather than re-sampling.

**Content schemas:**
- `text` → `{ text }` (30–180 words, platform voice).
- `image` → `{ caption, overall_description, parts[], metadata{camera, lens, filter, aspect_ratio, dimensions, iso, shutter, aperture, color_profile, location, time_of_day, filename} }`.
- `short_video` → `{ title, caption, overall_description, key_frames[], audio_transcript, metadata{duration_s, resolution, fps, aspect_ratio, music_track, sound_design, codec, bitrate_kbps, creator_handle} }`.

**Cost:** one LLM call per event (parallelized via ThreadPoolExecutor), ~1,760 calls per persona (IG ~600 + FB ~560 + Threads ~600; Chatbot and stubs skipped). Routed to the mini-tier client when `llm_client_mini` is provided (falls back to flagship otherwise). Retries use the shared 3-attempt exponential-backoff wrapper; total-failure events get a minimal placeholder content dict so downstream consumers never see missing fields.

---

## 14. Step 14 — Stereotype Annotation

Each preference gets a stereotype mark based on demographics **only** (gender, sexual orientation, race/ethnicity — not career/education/personality).

Three marks: `neutral` (no association, ~80%+), `stereotypical` (aligns with recognized stereotype), `anti-stereotypical` (contradicts). Conservative: when in doubt, neutral.

---

## 15. Steps 15-16 — Test Split and Save

**Time-based, cross-app test selection (no "train" label):**
1. Sort high-confidence positives (`init >= 0.75 AND cross_ref > canonical_xref_threshold(...)`) by latest-occurrence timestamp, newest-first.
2. `n_test_target = min(pool_size, max(10, int(pool_size * 0.2)))`. Floor of 10 items per user when the pool is large enough, 20% otherwise.
3. Walk the pool newest-first in batches of `n_test_target`. For each batch, the inferrability gate runs against "all cross_referenced_personas minus this batch". Inferrable items become `test` in strict newest-first order until the target is hit or the pool is exhausted.
4. Items that fail the inferrability gate stay in `cross_referenced_personas` as interaction history — **never deleted**. Only "test" is written to the output; non-test preferences have no `split` field.

**Inferrability gate:** LLM evaluates each candidate — can it be predicted from the rest of the history? The gate is informational only; it never removes canonicals from the pipeline.

**Distractor pairing (3 per test item, causal):** For each test item, Python filters the non-test high-confidence pool to items whose first-occurrence timestamp `<=` the test item's last-occurrence timestamp (causality). LLM then ranks the top 3 most topically irrelevant / annoying items from a random shortlist of 15. Stored as a **list** of `{persona_item, category}` objects under `over_personalization_irrelevant`.

**Step 16:**
- `profile.json` preferences are rendered as `"{latest_timestamp} : {persona_item}"` strings, sorted by latest timestamp descending (most recent first).
- `similar` / `contradicted` entries in per-event `update_history` are attached only if the related preference's first-occurrence timestamp is `<=` the event's timestamp (strict causality).
- `hidden_persona_labels` are now derived by **backward lookup** (row → cluster via `evidence_oids`), so causality is guaranteed by construction: a preference is labeled iff its own source row is part of the cluster's evidence. No separate availability gate needed.

---

## 16. Noise Summary

All noise applied after skeleton establishment. Skeleton (Steps 1-2) is deterministic; noise enters at Steps 5+.

| Source | Stage | Parameter | Effect |
|--------|-------|-----------|--------|
| Demographic sampling | Step 6 | Weighted distributions | Random identity |
| Per-user action weights | Step 12 | `noise_strength = 0.6`, seed=user_id | Lognormal perturbation |
| Session reassignment | Step 11 | `NOISE_REASSIGN_PROBABILITY = 0.08` | 8% sessions re-routed |
| Conversation type | Step 13 | `CHATBOT_CONVERSATION_TYPES` weights | Per-user conversation mix |
| Per-user content mix | Step 13b | `CONTENT_MIX_NOISE_SIGMA = 0.3`, seed=(user_id, app) | Lognormal perturbation of the posterior mix |
| Per-event content type | Step 13b | seed=(user_id, oid) | Ambiguous-action events sample from user mix |
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
| `MIN_IMPLICIT_NEGATIVE_REPETITION` | 5 | Implicit-only negative survival threshold (distinct rows) and net-sentiment floor |
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
| Chatbot turn pool | `{2,4,6,8}` pos / `{2,4,6}` neg | Per-event random choice, clamped by `min(n_prefs*2, 8)` |
| Test fraction | 0.20 | Latest 20% of high-confidence positives |
| Test floor | 10 | Min test items per user (only reduced when the high-conf pool itself is smaller) |
| Distractors per test item | 3 | Ranked via LLM from causally-filtered shortlist of 15 |
| `MIN_HIDDEN_PERSONA_ROWS` | 40 | Min rows for hidden persona |
| `MIN_HIDDEN_PERSONA_DAYS` | 3 | Min temporal spread for hidden persona |
| `HIDDEN_PERSONA_HASHTAG_MIN_FREQ` | 3 | Min hashtag occurrences |
| `HIDDEN_PERSONA_TOP_HASHTAGS` | 200 | Top hashtags passed to LLM |

> Thresholds (especially high-confidence predicate values) are tentative and will be tuned empirically.
