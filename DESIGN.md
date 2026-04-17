# PersonaMem-v3: Design Document

*Towards All-Day-Long Omni-Platform Personal Intelligence*

> This document summarizes every design choice in the PersonaMem-v3 data generation pipeline. It is written for researchers and collaborators who need to understand the rationale — not the code. For implementation details, see `persona_agent.py`, `prompts.py`, and `skill.md`.

## Core Research Questions

1. How can we simulate data that accurately reflects real-world distributions and formats to create a personalization dataset that mimics a user's digital behaviors all day long?
2. How well can LLMs understand user personas and preferences from noisy contexts across multiple platforms to provide proactive personalization and next-step suggestions?

---

## 1. Conceptual Framework

The pipeline follows a three-stage generative process:

```
Persona Generation  ──>  Preference Generation  ──>  Omni-Platform Interaction History
```

**Stage 1 — Persona Generation.** From raw hashtag-based interaction logs, we infer a rich user persona: demographics, personality traits, career, education, and a bio. This persona is the "skeleton" — a single shared ground truth that every downstream artifact must trace back to.

**Stage 2 — Preference Generation.** From the same interaction logs, we extract, filter, and cross-reference atomic preference statements (e.g., "Enjoys home cooking", "Dislikes clickbait fitness content"). These preferences are scored, deduplicated, and validated against each other to form a high-confidence preference set anchored to the persona skeleton.

**Stage 3 — Omni-Platform Interaction History.** Preferences are distributed across four platforms — Instagram, Facebook, Threads, and an AI Chatbot — producing realistic, timestamped interaction events with platform-specific actions, conversations, and engagement patterns. Noise is injected at this stage, *after* the ground-truth skeleton is established.

The data captures three interaction pillars, spanning all-day digital behavior:

| Pillar | What it models | Platform(s) |
|--------|---------------|-------------|
| **Social Media Engagement** | Feed browsing: likes, saves, shares, skips, @ai comments | Instagram, Facebook, Threads |
| **Human-LLM Chat** | Conversational queries, ask-to-forget, corrections | AI Chatbot |
| **Multi-Platform Interactions** | Cross-app routing, session-based browsing, per-app personas | All four |

### The Single Ground-Truth Principle

Every interaction event, on every platform, traces back to **one shared preference skeleton**. The skeleton is established through Steps 1–2 (inference + cross-referencing) and locked in before any platform-specific generation begins. This means:

- A preference that appears on Instagram and again in a Chatbot conversation is the *same* canonical preference.
- The user profile (demographics, personality) is generated once and shared across all apps.
- Per-app sub-personas describe *how the user presents* on each platform, not *what they like* — the preference set is shared.
- Noise (8% app reassignment, per-user action perturbation, etc.) is applied *after* the skeleton is finalized, never to the skeleton itself.

### Three Core Tensions

| Tension | Trade-off |
|---------|-----------|
| **Signal fidelity vs. coverage** | Strict filtering (init ≥ 0.5, bottom-20% removal) produces a cleaner ground truth but drops weaker-but-valid preferences. We choose fidelity. |
| **Realism vs. tractability** | Perfectly realistic data would require modeling attention, mood, and social context. We approximate with session-based routing, per-user action distributions, and temporal evolution. |
| **Fairness vs. accuracy** | Demographic distributions are intentionally diversified beyond census proportions; the LLM is instructed to avoid stereotypical combinations even when they might be statistically likely. We choose fairness. |

---

## 2. Pipeline Overview

```
Input CSV (hashtag interactions per user)
  │
  ├─ Step 1:  Infer atomic personas           [LLM]      ── scores: init 0.0–1.0
  ├─ Step 2:  Promote implicit negatives     [Algo+LLM] ── net-sentiment gate
  ├─ Step 3:  Cross-reference & filter       [Algo+LLM] ── scores: cross_ref (uncapped)
  ├─ Step 4:  Temporal contradiction graph   [LLM]      ── timeline grouping
  ├─ Step 5:  Build update histories         [Algo+LLM] ── reinforced/faded/evolved
  ├─ Step 6:  Generate user profile          [LLM]      ── demographics + Big Five
  ├─ Step 7:  Infer hidden personas          [Algo+LLM] ── cross-row hashtag clustering
  ├─ Step 8:  Generate per-app sub-personas  [LLM]      ── 4 AppPersonas
  ├─ Step 9:  Build sessions                 [Algo]     ── temporal grouping
  ├─ Step 10: Route preferences to apps      [LLM+Algo] ── distribution: ~40/20/20/20
  ├─ Step 11: Assign rows to apps            [Algo]     ── session majority vote + 8% noise
  ├─ Step 12: Generate interaction formats   [Algo+LLM] ── per-user perturbed weights
  ├─ Step 13: Generate chatbot conversations [LLM]      ── multi-turn, ask-to-forget
  ├─ Step 14: Annotate stereotype marks      [LLM]      ── demographics-only
  ├─ Step 15: Build train/test split         [LLM+Algo] ── 80/20, inferrability gate
  └─ Step 16: Save to backend               [Algo]     ── 5 JSON files per user
```

| Step | Type | Key output |
|------|------|------------|
| 1 | LLM | ~10 atomic preferences per interaction, each with init confidence |
| 2 | Algo + LLM | Promoted implicit negatives via net-sentiment (≥5) + temporal spread (≥3 days) |
| 3 | Algo + LLM | Deduplicated, cross-referenced preference skeleton with cross_ref scores |
| 4 | LLM | Contradiction timelines grouped by topic |
| 5 | Algo + LLM | Update history per preference (reinforced, faded, evolved) |
| 6 | LLM | Synthetic user profile: name, demographics, Big Five, career, bio |
| 7 | Algo + LLM | Hidden personas: deeper motivational layers from cross-row hashtag patterns |
| 8 | LLM | Four distinct AppPersonas (one per platform) |
| 9 | Algo | Temporal sessions (rows within ≤5s grouped together) |
| 10 | LLM + Algo | Each preference routed to one primary app |
| 11 | Algo | Session majority vote + 8% noise for row-to-app assignment |
| 12 | Algo + LLM | Platform-specific action + optional user_message per preference |
| 13 | LLM | Multi-turn chatbot conversations with implicit preference embedding |
| 14 | LLM | Stereotype marks (neutral / stereotypical / anti-stereotypical) |
| 15 | LLM + Algo | Train/test labels + distractor pairing for test items |
| 16 | Algo | `profile.json` + 4 app JSONs per user |

---

## 3. Input and Output

### Input

A CSV of anonymized social media interactions. Each row is one user engaging with one piece of content.

| Column | Type | Description |
|--------|------|-------------|
| `interaction_type` | string | `explicit_positive`, `implicit_positive`, `explicit_negative`, `implicit_negative` |
| `user_id` | string | Anonymized user identifier |
| `object_id` | string | Anonymized content identifier |
| `interaction_time` | int | Unix timestamp |
| `object_text` | string | Space-separated hashtags (e.g., `#CrossFit #MorningRoutine #FitFam`) |

The four interaction types capture the full spectrum of engagement:

| Type | Signal strength | Example |
|------|----------------|---------|
| `explicit_positive` | Strong positive | User liked, saved, or shared the content |
| `implicit_positive` | Weak positive | User lingered, watched 75%+ of a video, or tapped through |
| `explicit_negative` | Strong negative | User hid, muted, unfollowed, or reported |
| `implicit_negative` | Weak negative | User scrolled past with no interaction |

### Output

Per-user directory at `backend/{user_id}/`:

| File | Contents |
|------|----------|
| `profile.json` | User profile + 4 AppPersonas + flat unique preference list |
| `instagram.json` | Interaction events routed to Instagram (time-sorted) |
| `facebook.json` | Interaction events routed to Facebook (time-sorted) |
| `threads.json` | Interaction events routed to Threads (time-sorted) |
| `chatbot.json` | Interaction events routed to Chatbot, with conversations (time-sorted) |

Each app JSON is an array of **interaction events**, sorted by timestamp. Each event represents one source CSV row and contains a nested `preferences` list:

```
Event (one per source row):
  ├─ source_object_id, source_timestamp, source_hashtags
  ├─ source_interaction_type
  ├─ interaction_format: { app, action, action_label, user_message }
  ├─ preferences[]:
  │     ├─ persona_item, category
  │     ├─ confidence_score_init, confidence_cross_referenced
  │     ├─ stereotype_mark, split (train/test)
  │     ├─ update_history[]
  │     └─ over_personalization_irrelevant (test items only)
  └─ [Chatbot only] conversation[], conversation_type, ask_to_forget
```

The same canonical preference text naturally appears across multiple events — this preserves real-world repetition and is intentional.

> **Realism.** Per-app files mirror real-world data silos (Meta's internal systems store Instagram and Facebook data separately). Event-level nesting preserves the temporal structure of browsing sessions. Chatbot-specific fields (multi-turn conversations, ask-to-forget flags) exist only on Chatbot events, just as they would in a real system.

---

## 4. Step 1 — Atomic Persona Inference

For each interaction row (except `implicit_negative`, handled separately in Step 1b), the LLM infers **~10 atomic persona traits** from the hashtags.

Each trait includes:
- `persona_item`: a specific, testable statement (e.g., "Enjoys cooking Italian food at home")
- `category`: a topical label (e.g., "Italian cooking")
- `confidence_score_init`: the LLM's confidence in this single inference
- `source_hashtags`: which hashtag(s) led to this inference

### Scores

The LLM follows a calibrated confidence scale. Scores must use two decimal places and be spread across the full range — not clustered at the top.

**Positive interactions** (full 0.0–1.0 range):

| Range | Meaning | Example |
|-------|---------|---------|
| 0.80–1.00 | Near-certain, explicitly stated | `#ILoveRunning` → "Enjoys running" (0.92) |
| 0.60–0.80 | Direct topic match | `#CrossFit` → "Interested in CrossFit" (0.71) |
| 0.40–0.60 | Reasonable deduction | `#MorningRoutine #FitFam` → "Values physical fitness as part of daily routine" (0.53) |
| 0.15–0.40 | Broader inference | `#FitFam` → "Likely health-conscious in dietary choices" (0.28) |
| 0.00–0.15 | Speculative | `#FitFam` → "May attend group fitness classes" (0.09) |

**Negative interactions** (compressed range reflecting inherent uncertainty):

| Range | Meaning |
|-------|---------|
| 0.55–0.75 | Direct dislike of the core topic |
| 0.35–0.55 | Reasonable deduction about what user dislikes |
| 0.15–0.35 | Broader inference about aversions |
| 0.00–0.15 | Speculative dislike |

Negative preferences are always phrased negatively: "Dislikes X", "Avoids X", "Not interested in X".

> **Realism.** The calibrated range prevents the common LLM failure mode of assigning 0.85+ to everything. By requiring scores spread across the full range, we get a realistic distribution where most inferences are moderate-confidence and only a few are near-certain. This mirrors real-world uncertainty: seeing `#CookingWithAPurpose` tells you the user is interested in cooking (high confidence) but only weakly suggests they value family meals (lower confidence).

---

## 5. Step 2 — Implicit Negative Promotion

`implicit_negative` rows (user scrolled past content with no interaction) are **skipped** in Step 1 — a single scroll-past is too weak a signal. Instead, we aggregate them:

### Problem

The base rate of scrolling past content is extremely high for *all* topics. Without filtering, every topic would be flagged as a dislike. We need a way to distinguish "user scrolled past because they dislike this" from "user scrolled past because everyone scrolls past most content."

### Net-Sentiment Formula

For each hashtag, we count its occurrences across all interaction types and compute:

```
net = (implicit_neg_count × 1.0) − (explicit_pos_count × 3.0) − (implicit_pos_count × 1.5)
```

| Weight | Interaction type | Rationale |
|--------|-----------------|-----------|
| ×1.0 | Each implicit_negative row | Baseline skip signal |
| ×3.0 | Each explicit_positive row | A single like is strong counter-evidence (cancels ~3 skips) |
| ×1.5 | Each implicit_positive row | A linger is moderate counter-evidence (cancels ~1.5 skips) |

**Example:** A hashtag with 8 skips, 1 like, and 1 linger scores `8 − 3 − 1.5 = 3.5` — below the threshold, NOT promoted. The user probably likes this topic but sometimes scrolls past it.

### Promotion Gate

A hashtag is "hot" only if **both** conditions hold:

1. `net_score ≥ 5` (the `MIN_IMPLICIT_NEGATIVE_REPETITION` threshold)
2. Negative rows span **≥ 3 distinct calendar days** (the `MIN_TEMPORAL_DAYS` threshold)

The temporal spread requirement prevents session-level noise (e.g., user scrolled fast during one sitting) from being misinterpreted as a persistent dislike.

### LLM Inference and Corroboration

- **One LLM call per hot hashtag** on a representative row, passing *only that single hashtag* (not the full row).
- Inferred preferences must be independently produced by **≥ 2 different hot-hashtag LLM calls** to survive (`MIN_PREF_CORROBORATION = 2`). This filters out one-off speculative inferences.
- Rows with ≥ 1 hot hashtag carrying surviving preferences are **promoted** to `explicit_negative` in the output.
- Non-promoted `implicit_negative` rows remain as **stub events** with empty `preferences: []` (rendered in greyscale in the HTML visualization).

> **Realism.** This multi-gate approach mirrors how real recommendation systems handle implicit negative signals: one skip means nothing, but consistent avoidance of a topic — while actively engaging with *other* content — is a meaningful signal. The net-sentiment weighting ensures that topics the user genuinely likes (but sometimes scrolls past) are not falsely flagged.

---

## 6. Step 3 — Cross-Referencing (The Core Engine)

This is the central pipeline stage that transforms raw atomic inferences into the validated preference skeleton. It runs seven sub-stages:

### 6.1 Merge Duplicates

All atomic persona items across all interaction rows are normalized (lowercase, whitespace-collapsed) and grouped by exact string match. Identical texts merge into one **canonical** preference.

No semantic deduplication is applied — this is intentional. "Enjoys cooking Italian food at home" and "Interested in Italian cuisine" remain separate. The corroboration count (next stage) captures frequency; semantic similarity is handled later via LLM relationship discovery.

### 6.2 Init Filter

Drop any canonical whose `max(confidence_score_init)` across all its instances is **< 0.5** (the `MIN_PERSONA_INIT_CONFIDENCE` threshold).

Additionally, 10% of below-threshold canonicals are randomly retained as an exploration mechanism, preventing the loss of valid-but-weakly-stated preferences entirely.

### 6.3 Weighted Corroboration

For each surviving canonical, count the distinct source rows (`source_object_id`) that independently produced it, weighted by interaction type:

```
confidence_cross_referenced = Σ over distinct source rows r:
    1.0  if r is explicit  AND  r.init ≥ 0.5
    0.5  if r is implicit  AND  r.init ≥ 0.5
```

This score is **intentionally uncapped**. A preference corroborated by 200 distinct rows scores ~200.0; one corroborated by 10 rows scores ~10.0. They must NOT both collapse to the same ceiling — the magnitude is a meaningful signal at scale.

### 6.4 LLM Relationship Discovery

Per-category LLM calls identify `similar` and `contradictory` relationships between distinct canonicals. The LLM does **not** alter any scores — it only discovers relationships for use in the next stages.

Categories with only one canonical are skipped (nothing to cross-reference against).

### 6.5 Union-Find Clustering

Similar preferences are merged via union-find. The cluster representative is the canonical with the highest `confidence_score_init`. Cross-ref scores are **summed** across cluster members. Contradictory relationships are preserved on the representative.

### 6.6 Contradiction Penalty

For each canonical with contradictory relationships, subtract the contradicting canonical's cross-ref score. Floor at 0.0.

**Example:** "Enjoys vegan cooking" (cross_ref = 15.0) is contradicted by "Loves BBQ ribs" (cross_ref = 8.0). After penalty: 15.0 − 8.0 = 7.0.

### 6.7 Bottom-20% Filter

Remove canonicals in the bottom 20% of cross-ref scores, **unless** their score exceeds 10.0 (the `bottom_20_min_exempt` threshold — high-count items are exempt). Then apply a hard floor: only canonicals with `cross_ref ≥ 10.0` (`HIGH_CONFIDENCE_CROSS_REF_THRESHOLD`) survive. Cross-ref starts at 1.0 (base) and accumulates 1.0 per distinct explicit row, 0.5 per distinct implicit row, so a score of 10.0 requires roughly 9 explicit corroborating rows beyond the initial occurrence.

### Summary

| Stage | Type | Effect on scores |
|-------|------|-----------------|
| Merge | Algorithmic | None (dedup only) |
| Init filter | Threshold | Drops canonicals with init < 0.5 |
| Corroboration | Algorithmic | Sets `confidence_cross_referenced` |
| LLM cross-ref | LLM | None (discovers relationships only) |
| Union-find | Algorithmic | Sums cross_ref across cluster |
| Contradiction penalty | Algorithmic | Subtracts from cross_ref |
| Bottom-20% filter | Statistical | Drops weak tail; floor at 10.0 |

### Negative Cross-Referencing

Negative preferences go through the **same pipeline independently** (within negatives only). Key differences:

- Canonicals supported **only** by implicit evidence must have **≥ 5 distinct source rows** to survive the init filter. Any explicit-negative evidence bypasses this gate.
- No bottom-20% filter is applied to negatives (they are already high-signal after promotion).

> **Realism.** The skeleton that emerges from this pipeline has survived multiple independent validation gates. A preference in the final set is one that (a) was inferred with ≥ 0.5 confidence, (b) appeared across multiple distinct interactions, (c) was not contradicted by stronger opposing evidence, and (d) was not in the weakest 20% of all preferences. This mirrors how real personalization systems require repeated, consistent signals before acting on a preference.

---

## 7. Steps 4–5 — Temporal Evolution

### Step 4: Temporal Contradiction Graph

Preferences marked as `contradictory` in Step 2 are grouped by topic/theme. For each group, a chronological timeline is built, showing how the user's stance shifted over time (e.g., "user initially preferred vegan options in January, then shifted to pescatarian by March").

### Step 5: Update Histories

Each surviving preference receives a temporal update history — an array of entries sorted by timestamp, each tagged with an `update_type`. There are **8 distinct values**:

| `update_type` | Definition | Example |
|----------------|-----------|---------|
| `new` | First appearance of the preference (the initial inference) | Filtered out during serialization — redundant with the event timestamp |
| `reinforced` | Preference appeared in multiple distinct source rows | Up to 5 recurrence timestamps, evenly sampled across the timeline |
| `faded` | Preference inactive for > 48 hours before the user's last activity | `(user_last_ts − pref_last_ts) ≥ FADE_THRESHOLD_SECONDS (172,800)` |
| `contradicted` | A contradicting preference was discovered during cross-referencing | "Enjoys vegan cooking" contradicted by "Loves BBQ ribs" |
| `deepened` | A general interest became more specific over time | "Likes cooking" → "Follows advanced baking techniques" |
| `branched` | Interest expanded into a new sub-direction | "Hair styling" → also "Hair product reviews" |
| `shifted` | Focus moved from one aspect to another within the same domain | "Comedy reels" → "Wholesome family humor" |
| `intensified` | Engagement grew demonstrably stronger over time | Casual mentions early → frequent engagement later |
| `similar` | A semantically similar preference was discovered during cross-referencing | "Enjoys Italian cooking" linked to "Interested in Mediterranean cuisine" |

### Update History Data Format

Each entry in the `update_history` array is a JSON object. Fields are ordered: `update_type` → `preference` → `formatted_timestamp` → extras. Not all fields appear on every type.

| Field | Type | Present on | Description |
|-------|------|-----------|-------------|
| `update_type` | string | All | One of the 8 values above |
| `preference` | string | `contradicted`, `deepened`, `branched`, `shifted`, `similar` | The related preference text (target of evolution, or the contradicting/similar preference) |
| `formatted_timestamp` | string | All | Human-readable timestamp (e.g., `"01:03, 04/03/2026"`) |
| `source_app` | string | `reinforced`, `deepened`, `branched`, `shifted`, `similar` | Which app the related row/preference was routed to |
| `occurrence` | int | `reinforced` | 1-indexed occurrence number (starts at 2, since 1 is the initial "new") |
| `total_occurrences` | int | `reinforced` | Total distinct source rows that produced this preference |
| `description` | string | `deepened`, `branched`, `shifted`, `intensified` | One-sentence LLM narrative describing the evolution pattern |

**Examples from output:**

```json
{"update_type": "reinforced", "formatted_timestamp": "01:03, 04/03/2026", "source_app": "Threads", "occurrence": 2, "total_occurrences": 58}
```
```json
{"update_type": "deepened", "preference": "Actively follows the Crawford vs. Canelo matchup.", "formatted_timestamp": "06:47, 04/03/2026", "source_app": "Threads", "description": "Athlete-specific boxing fandom evolved into focused engagement with marquee matchup narratives."}
```
```json
{"update_type": "faded", "formatted_timestamp": "22:56, 04/03/2026"}
```
```json
{"update_type": "contradicted", "preference": "Prefers bold, attention-grabbing fashion aesthetics.", "formatted_timestamp": "05:49, 04/04/2026"}
```
```json
{"update_type": "similar", "preference": "Interested in Mediterranean cuisine.", "formatted_timestamp": "01:15, 04/03/2026", "source_app": "Facebook"}
```

**Causality filter:** only entries with `timestamp ≤ event timestamp` are included — no knowledge leakage from future interactions.

A causality filter ensures only update entries with `timestamp ≤ event timestamp` are included — no knowledge leakage from future interactions.

> **Realism.** Preferences are not static. Real users' interests evolve: they discover new hobbies, lose interest in old ones, and shift positions on topics. The update history captures this temporal dimension, enabling downstream systems to reason about preference recency and trajectory.

---

## 8. Step 6 — Synthetic User Profile

Each user receives a synthetic profile grounded in their preference skeleton.

### Demographic Distributions

Demographics are **sampled first**, before any other profile field. Everything downstream (name, career, bio) must be consistent with the sampled demographics.

**Gender x Orientation** (18 entries):

| Category | Probability |
|----------|------------|
| Cisgender female, heterosexual | 30% |
| Cisgender male, heterosexual | 32% |
| Cisgender male, gay | 5% |
| Cisgender female, bisexual | 4% |
| Cisgender female, lesbian | 4% |
| Non-binary, queer | 2% |
| Cisgender male, bisexual | 2% |
| Transgender female, heterosexual | 2% |
| Transgender male, heterosexual | 2% |
| Cisgender female, queer | 1% |
| Cisgender male, queer | 1% |
| Transgender female, lesbian | 1% |
| Transgender female, bisexual | 1% |
| Transgender male, gay | 1% |
| Transgender male, bisexual | 1% |
| Non-binary, bisexual | 1% |
| Non-binary, pansexual | 1% |
| Non-binary, asexual | 0.5% |
| Genderfluid, queer | 0.5% |
| Genderfluid, pansexual | 0.5% |
| Agender, asexual | 0.5% |

**Race / Ethnicity** (28 entries):

| Category | Probability |
|----------|------------|
| White American | 15% |
| Black or African American | 8% |
| Chinese | 10% |
| Indian | 8% |
| Mexican American | 8% |
| Filipino | 4% |
| Vietnamese | 4% |
| Korean | 4% |
| Japanese | 3% |
| Middle Eastern or North African | 3% |
| Multiracial (other) | 3% |
| White European immigrant | 2% |
| Russian or Eastern European | 2% |
| Jewish American | 2% |
| African immigrant | 2% |
| Afro-Caribbean | 2% |
| Puerto Rican | 2% |
| Central American | 2% |
| South American | 2% |
| Pakistani or Bangladeshi | 2% |
| Southeast Asian | 2% |
| Multiracial (Black and White) | 2% |
| Multiracial (Asian and White) | 2% |
| Multiracial (Hispanic and White) | 2% |
| Cuban American | 1% |
| Central Asian | 1% |
| Native Hawaiian or Pacific Islander | 1% |
| American Indian or Alaska Native | 1% |

These distributions are intentionally diversified beyond US census proportions to ensure adequate representation of minority groups in the dataset.

### LLM-Generated Fields

Given the sampled demographics and the preference skeleton, the LLM generates:

| Field | Details |
|-------|---------|
| **Name** | Culturally appropriate for the sampled race/ethnicity |
| **Career** | Consistent with some preferences but not all |
| **Education** | Level and field |
| **Big Five** | Openness, conscientiousness, extraversion, agreeableness, neuroticism — each rated low/medium/high |
| **Bio** | 3–5 sentence description grounding the user in a realistic life context |

The prompt explicitly instructs the LLM to: be consistent with *some but not all* preferences (a real person's profile doesn't explain every interest), and actively avoid stereotypical demographic-career-hobby combinations.

> **Realism.** Demographics are sampled before generation to prevent the LLM from defaulting to a "generic" profile. The anti-stereotype instruction forces diversity: a Chinese female user might be a firefighter, a Black male user might be into figure skating. Surprising combinations are explicitly encouraged because real people are not their stereotypes.

---

## 9. Step 7 — Hidden Persona Inference

<!-- Theoretical foundations:
  - Katz, Blumler & Gurevitch (1973): Uses and Gratifications Theory — people actively select media to satisfy cognitive, affective, integrative, and escapist needs
  - Deci & Ryan (1985, 2000): Self-Determination Theory — autonomy, competence, and relatedness as basic psychological needs driving behavior
  - Kardefelt-Winther (2014): Compensatory Internet Use — social media compensates for unmet real-world needs; high private consumption signals compensation
  - Berger & Heath (2007); Smaldino et al. (2022): Identity Signaling Theory — consumption patterns signal group membership through overt and covert markers
  - Horton & Wohl (1956); Dibble et al. (2016): Parasocial Relationship Theory — one-sided emotional bonds with media figures detectable through concentrated engagement
  - Paulhus & Williams (2002): Dark Triad — narcissism, Machiavellianism, psychopathy manifest as behavioral styles in social media engagement (we use behavioral descriptions only, never clinical labels)
  - Maslow (1943): Hierarchy of Needs — users operate on multiple need levels simultaneously (safety, belonging, esteem, self-actualization)
-->

After the profile is generated, the pipeline analyzes **cross-row hashtag patterns** to infer deeper motivational layers not captured by individual-row inference. The type taxonomy is grounded in established behavioral science frameworks.

### Why Hidden Personas?

The per-row inference (Step 1) captures *what* a user likes. Hidden personas capture *why* — the underlying personality traits, aspirations, emotional patterns, identity anchors, intimate interests, and private hobbies that explain observable engagement. For example, a user who consumes both `#RelationshipGoals` (69x) and `#DivorceCourt` (5x) content has a surface preference for "romantic content," but the hidden persona is "romantic vulnerability and yearning" — a desire for emotional stability mixed with relationship anxiety.

### Three-Phase Algorithm

**Phase 1 — Hashtag Frequency Census (algorithmic).** Scan all interaction rows. For each hashtag, count total occurrences, per-interaction-type breakdown, and distinct calendar days. Filter to hashtags with ≥ 3 total occurrences. Pass the top ~200 to the LLM.

**Phase 2 — LLM Thematic Clustering.** The LLM receives the hashtag frequency table, the user's demographics, and the surviving preference skeleton. It groups hashtags into 8–15 thematic clusters, each representing a hidden persona. Types:

<!-- Uses & Gratifications → personality_trait, aspiration, emotional_pattern, intellectual_curiosity, private_hobby -->
<!-- Self-Determination Theory → autonomy needs surface in personality_trait; competence in intellectual_curiosity + aspiration; relatedness in emotional_pattern -->
<!-- Identity Signaling Theory → identity_anchor (overt tribal + covert aesthetic markers) -->
<!-- Compensatory Use Theory → compensatory_need (high privacy_ratio = unmet real-world need) -->
<!-- Parasocial Relationship Theory → parasocial_attachment (concentrated engagement with one figure) -->

| Type | What it captures | Theoretical basis |
|------|-----------------|-------------------|
| `personality_trait` | Core character attributes (nostalgic, risk-averse, drawn to transgressive humor) | Big Five; Dark Triad behavioral markers |
| `aspiration` | Dreams and goals (entrepreneurial ambitions, financial freedom) | Maslow's esteem/self-actualization |
| `emotional_pattern` | Recurring emotional dynamics (romantic yearning + anxiety, need for validation) | Uses & Gratifications (affective needs) |
| `identity_anchor` | Cultural era, community, tribal belonging — both overt markers (community names) AND covert signals (niche aesthetics like #hopelesscore, #cottagecore) | Identity Signaling Theory |
| `intimate_interest` | Body confidence, sensuality, attraction patterns — specific objects, clothing, body areas, dynamics | Self-presentation; body image research |
| `intellectual_curiosity` | Hidden learning interests (ancient history, science, paranormal) | Self-Determination Theory (competence need) |
| `private_hobby` | Interests consumed but not publicly shared (high implicit ratio) | Uses & Gratifications (escapist needs) |
| `parasocial_attachment` | Intense one-sided bond with a specific public figure, detected by ≥15 rows mentioning one figure | Parasocial Relationship Theory |
| `compensatory_need` | Unmet real-world needs satisfied through private media consumption (privacy_ratio >0.7) | Compensatory Internet Use Theory |

**Phase 3 — Algorithmic Validation.** Each cluster is validated against raw data:

| Metric | Threshold |
|--------|-----------|
| Distinct source rows | ≥ 20 (`MIN_HIDDEN_PERSONA_ROWS`) |
| Temporal spread (distinct days) | ≥ 3 (`MIN_HIDDEN_PERSONA_DAYS`) |
| Privacy ratio (`impl_pos / (impl_pos + expl_pos)`) | Reported, not gated (>0.7 required for `compensatory_need`) |
| App distribution | Computed retroactively after routing |

**Phase 3.5 — Hashtag-Overlap Deduplication.** After validation, a deduplication pass merges hidden personas whose evidence hashtags overlap significantly. Pairwise Jaccard similarity is computed on lowercase evidence_hashtag sets. If Jaccard ≥ 0.5 (half the hashtags overlap), the two personas are merged: the one with more evidence_rows becomes the base, evidence_hashtags are unioned, surface_connections are unioned, and evidence metrics (evidence_rows, evidence_row_fraction, interaction_breakdown, privacy_ratio, temporal_spread_days) are recomputed from raw data. The pass repeats iteratively until no more merges occur. The LLM prompt also instructs against producing near-duplicate clusters.

### Dual Personalities

<!-- Approach-avoidance conflict: Lewin (1935); Miller (1944) — individuals simultaneously attracted to and repelled by the same goal or competing goals -->

After validation, the LLM identifies **dual personality tensions** — pairs of hidden personas that coexist in contradiction. These represent psychologically well-established internal conflicts (approach-avoidance, public-vs-private self, contradictory needs). Both halves must independently pass validation gates. Examples: public confidence + private vulnerability, aspirational luxury + minimalist escape.

Stored as `dual_personalities` in `profile.json`: `[{"persona_a": "...", "persona_b": "...", "tension": "..."}]`.

### Per-Preference Hidden Persona Labels

Each preference in the saved app JSONs carries a `hidden_persona_labels` field linking it to the hidden persona(s) it provides evidence for. The match is computed by checking whether the preference's source hashtags overlap with a hidden persona's `evidence_hashtags`. Preferences with no overlap get an empty list — labels are never forced.

### Output

Each validated cluster becomes a `HiddenPersona` with: label, type, description, evidence_hashtags, evidence_rows, evidence_row_fraction, interaction_breakdown, privacy_ratio, temporal_spread_days, app_distribution, surface_connections, inferred_motivation.

A second LLM call generates a `hidden_persona_summary` — a cohesive narrative paragraph linking all hidden personas to observable surface behaviors. A third LLM call detects `dual_personalities`. All are saved in `profile.json`.

> **Realism.** Real users have motivational layers beneath their visible engagement. A user doesn't just "like boxing content" — they may have a parasocial attachment to a specific fighter, be nostalgic for a cultural era, privately compensate for romantic loneliness through couple content, or process relationship dynamics as a spectator. These hidden layers, grounded in established behavioral science, make the dataset useful for deep personalization beyond surface-level topic matching.

---

## 10. Step 8 — Per-App Sub-Personas

Each user receives **four distinct AppPersona objects** — one per platform — describing how they present on each app.

### AppPersona Structure

| Field | Values | Description |
|-------|--------|-------------|
| `use_purposes` | 2–4 items | Why this user uses this app |
| `friend_zones` | 2–4 items | Who they interact with here |
| `audience_type` | private / public / mixed | Visibility of their activity |
| `style_description` | 2–3 sentences | Tone and aesthetic |
| `posting_frequency` | daily / weekly / rarely / passive viewer only | How often they post |
| `topical_focus` | 3–5 domains | What topics dominate on this app |
| `chatbot_contexts` | 2–3 items (Chatbot only) | Types of AI conversations they have |

### Platform Archetypes

| Platform | Typical audience | Typical use |
|----------|-----------------|-------------|
| **Facebook** | Family-leaning, extended network | Updates, groups, events, marketplace |
| **Instagram** | Mixed (close friends + public) | Aesthetic sharing, reels, stories |
| **Threads** | Public | Opinions, current events, quick takes |
| **Chatbot** | Private (self only) | Task assistance, knowledge queries, reflection |

### Chatbot Contexts (8 predefined options)

The Chatbot AppPersona selects 2–3 from: professional emails, personal emails, composing chat messages, composing social media posts, multilingual translation, knowledge exploration, therapy and reflection, medical consultations.

> **Realism.** Real people compartmentalize their online presence. The same user might share polished fitness photos on Instagram, vent about politics on Threads, ask for parenting advice on Facebook, and use the AI Chatbot for recipe troubleshooting. Modeling four distinct sub-personas per user captures this compartmentalization and creates realistic cross-app evaluation scenarios.

---

## 11. Steps 9–11 — App Routing

Each surviving preference is assigned to exactly one primary app. The routing has three stages:

### Stage 1: LLM Routing

The LLM assigns each canonical preference to the best-fitting app based on the per-app sub-personas (use_purposes, topical_focus, audience_type).

**Target distribution:** ~40% Chatbot, ~20% each for Instagram, Facebook, Threads.

The Chatbot receives a larger share because implicit interaction signals (lingering, continuing a topic) naturally map to conversational engagement patterns.

### Stage 2: Session-Based Majority Vote

Source rows are grouped into temporal **sessions** (consecutive rows with timestamp gaps ≤ 5 seconds). Within each session:

1. Each row gets the majority-vote app of its preferences' canonical assignments.
2. Each session gets the majority-vote app across all its rows.
3. All rows in the session are overridden to the session's app.

This ensures all posts from the same scrolling session land on the same app.

### Stage 3: Noise Injection (8%)

After session routing, **8% of sessions** are randomly reassigned to a different app (`NOISE_REASSIGN_PROBABILITY = 0.08`). The entire session moves as a unit.

Additionally, `implicit_negative` rows are **never routed to Chatbot** — skipping content doesn't naturally occur in chatbot conversations. These are redirected to a random social app.

> **Realism.** Session-based routing mirrors real browsing behavior: if a user opened Instagram at 2:15 PM and scrolled for 3 minutes, all the posts they saw were on Instagram. The 8% noise simulates cross-app leakage — a parenting post might appear on Threads even though parenting content is mostly on Facebook. This prevents unrealistically clean app boundaries.

---

## 12. Step 12 — Interaction Formats

Each preference gets a platform-specific interaction action (e.g., "Liked", "Saved to a collection", "Replied").

### The Catalog

`PLATFORM_INTERACTION_FORMATS` is the **single source of truth**. The pipeline picks actions **verbatim** from this catalog — it never invents new actions or labels. Each entry has an `action` identifier, a human-readable `label`, and a `weight` reflecting real-world relative frequency.

**Example: Instagram explicit_positive actions**

| Action | Label | Weight |
|--------|-------|--------|
| `liked` | Liked | 50.0 |
| `double_tapped` | Double-tapped to like | 22.0 |
| `at_ai_recommend_more` | @ai comment: asked for MORE like this | 12.2 |
| `at_ai_focus_topic` | @ai comment: asked to focus on this topic | 12.2 |
| `saved_to_collection` | Saved to a collection | 8.0 |
| `reacted_to_story` | Reacted to the story | 5.0 |
| `commented` | Commented | 4.0 |
| `followed_creator` | Followed the creator | 3.0 |
| `dm_to_friend` | Sent via DM to a friend | 3.0 |
| `shared_to_close_friends_story` | Shared to Close Friends story | 2.0 |
| `reposted` | Reposted | 1.0 |

The pattern is consistent across all platforms: **passive actions >> active actions >> rare actions**. Likes dominate; saves, comments, and shares are progressively rarer; @ai comments are ~20% of the explicit positive bucket.

### Per-User Action Perturbation

Each user gets a **personalized copy** of the action weights:

1. Seed a random number generator with `user_id` (deterministic, reproducible).
2. Multiply each weight by `exp(N(0, 0.6))` — lognormal noise with `noise_strength = 0.6`.
3. Renormalize to preserve the original sum (keeps magnitudes comparable).

This means:
- The **same user** gets a consistent action distribution across all their preferences (their "behavior fingerprint").
- **Different users** get visibly different distributions: one user likes a lot, another saves a lot, a third comments frequently.
- The **overall shape** is preserved: likes are still the most common action for every user.

### Two Message-Bearing Action Groups

Most actions have `user_message = null`. Two groups require a generated natural-language message:

| Group | Platform | Actions | Message format |
|-------|----------|---------|---------------|
| **@ai Comments** (`AT_AI_ACTIONS`) | Instagram, Facebook, Threads | `at_ai_recommend_more`, `at_ai_focus_topic`, `at_ai_stop_recommending`, `at_ai_not_interested`, `at_ai_feels_off` | Starts with `@ai `, first-person, ~15–35 words |
| **Chatbot Turns** (`CHATBOT_TURN_ACTIONS`) | Chatbot | `asked_followup`, `requested_more_detail`, `continued_topic`, `asked_to_change_topic`, `edited_prompt_and_retried`, `regenerated`, `asked_to_forget`, `corrected_assumption` | Natural first-person, NO `@ai` prefix, ~15–35 words |

These represent two fundamentally different UX paradigms: @ai comments are public, brief, and directive (steering the feed); chatbot turns are private, conversational, and exploratory.

> **Realism.** The skewed weight distributions match publicly-reported engagement benchmarks (likes are ~50x more common than reposts). Per-user perturbation creates realistic individual variation — in real data, some users are "likers" and others are "savers." The lognormal noise preserves the shape while introducing visible per-user differences. Seeding on `user_id` ensures reproducibility across pipeline runs.

---

## 13. Step 13 — Chatbot Conversations

Every chatbot **event** (source row routed to Chatbot) gets a multi-turn conversation that naturally embeds ALL of that event's surviving preferences. Conversations are generated per-event, not per-canonical preference.

### Conversation Generation

| Probability | Type | Turns |
|-------------|------|-------|
| ~80% | Full multi-turn | 4–10 turns (scaled by preference count, always even) |
| ~20% | Minimal exchange | 2–6 turns (scaled by preference count, always even) |

Turn count scales with the number of preferences to ensure each preference can be naturally mentioned: `min(max(base, n_prefs * 2), 10)`.

### Conversation Types

Selected based on the user's `chatbot_contexts` from their AppPersona:

| Type | Weight | What it models |
|------|--------|---------------|
| `knowledge_query` | 30% | Factual or how-to questions revealing domain interest |
| `writing_help` | 25% | User pastes a draft; preference embedded in the draft content |
| `therapy_reflection` | 20% | Personal concerns; preference surfaces as incidental context |
| `health_consultation` | 15% | Health question; preference revealed through lifestyle details |
| `troubleshooting` | 10% | Practical problem; preference embedded in the problem context |
| `translation` | 10% | Translation request; preference embedded in the source text |
| `casual_chat` | 5% | Help composing a message; preference in the message content |

### Implicit Embedding

The user **never directly states** any preference. Instead, all preferences for the event are woven naturally into the conversation:
- For explicit interactions, the preference is "fairly apparent" through the task topic.
- For implicit interactions, the preference is "deeply embedded" — a side detail or cultural reference requiring reasoning to infer.
- When an event has multiple preferences, they are spread across turns — the primary task topic carries the most prominent preference(s), while others surface through details, follow-up questions, or contextual references.

**Example:** A user who enjoys Korean cooking doesn't say "I like Korean food." Instead, they ask: "What's the difference between gochugaru and gochujang in terms of fermentation?"

### Special Negative Scenarios

~70% of `explicit_negative` chatbot preferences get special treatment (`EXPLICIT_NEG_SPECIAL_FRACTION = 0.70`):

| Scenario | Structure | What it tests |
|----------|-----------|--------------|
| **Ask-to-forget** | 4 turns: user reveals → assistant acknowledges → user asks to forget → assistant confirms | Preference retraction / privacy |
| **Correction** | 4 turns: user message → assistant makes wrong assumption → user corrects → assistant adjusts | Error recovery / assumption handling |

> **Realism.** These conversations mirror real chatbot usage patterns. Users don't tell ChatGPT "I like hiking" — they ask "What's a good trail for a day hike near Portland with under 3,000 feet of elevation gain?" The ask-to-forget and correction scenarios test critical capabilities that existing personalization benchmarks rarely evaluate: can a system respect preference retraction and recover from incorrect assumptions?

---

## 14. Step 14 — Stereotype Annotation

Each preference (positive and negative) receives a stereotype mark based on the user's demographics.

### Scope: Demographics Only

Annotation considers **only**: gender, sexual orientation, race/ethnicity.

It does **not** consider: career, education, personality, or lifestyle. This is a deliberate choice — stereotype associations with career/education are less clear-cut and more controversial.

### Three Marks

| Mark | Definition | Frequency |
|------|-----------|-----------|
| `neutral` | No meaningful stereotypical association | Most (~80%+) |
| `stereotypical` | Aligns with a commonly recognized stereotype | Minority |
| `anti-stereotypical` | Contradicts or defies a commonly recognized stereotype | Minority |

The LLM is instructed to be conservative: "when in doubt, mark as neutral." Only widely recognized, well-documented associations are flagged.

> **Realism.** Stereotype annotations enable fairness evaluation of downstream personalization systems. By isolating annotation to immutable demographic characteristics (not career or education), we avoid conflating identity-based stereotypes with occupational associations. This makes the dataset useful for measuring whether personalization systems treat stereotypical and anti-stereotypical preferences equitably.

---

## 15. Steps 15–16 — Train/Test Split and Save

### Time-Based, Cross-App Split

1. Sort **all** positive survivors by `source_timestamp` ascending (globally, across all apps).
2. Scan newest → oldest, collecting items that pass the **high-confidence predicate** (`init ≥ 0.5 AND cross_ref > 10.0`) until reaching 20% of total positives.
3. These become **test candidates**. Everything else is `train`. All negatives are always `train`.

### Inferrability Gate

Each test candidate is evaluated by the LLM: can it be reasonably predicted from the train set?

- Conservative: when in doubt, mark NOT inferrable.
- Bridge signals: topical overlap, lifestyle coherence, demographic/cultural consistency with the train set.
- Non-inferrable items are **dropped entirely** — not demoted to `train`, but removed from the dataset. This prevents impossible evaluation tasks.

### Distractor Pairing

Each surviving test item is paired with an **over-personalization distractor**:

1. Python randomly shortlists 5 high-confidence train items.
2. LLM picks the one that is both **most topically irrelevant** and would be **most annoying/inappropriate** as a personalization recommendation at the moment of the test preference.

The distractor is stored as `over_personalization_irrelevant` on the test item.

> **Realism.** The time-based split matches real-world evaluation: predict future behavior from past history. The cross-app scope tests whether a system can infer preferences from one app to predict behavior on another. The inferrability gate ensures every test item is a fair evaluation target. The distractor pairing directly enables measuring over-personalization — a personalization system that recommends the distractor instead of the correct item is being overly aggressive.

---

## 16. Noise and Realism Summary

All noise is applied **after** the core ground-truth skeleton is established. The skeleton (Steps 1–2) is deterministic and filter-based; noise enters during platform-specific generation (Steps 5+).

| Source of noise | Stage | Parameter | Effect |
|----------------|-------|-----------|--------|
| Demographic sampling | Step 5 | `GENDER_ORIENTATION_DISTRIBUTION`, `RACE_ETHNICITY_DISTRIBUTION` | Random identity from weighted distributions |
| Per-user action weights | Step 8 | `noise_strength = 0.6`, seed = `user_id` | Lognormal perturbation of action frequencies |
| Session app reassignment | Step 7 | `NOISE_REASSIGN_PROBABILITY = 0.08` | 8% of sessions randomly re-routed |
| Exploratory below-threshold | Step 2 | 10% random retention | Some low-confidence preferences survive |
| Chatbot polarity mix | Step 11 | ~20% explicit / ~80% implicit | Adjusts chatbot event polarity labels |
| Conversation type selection | Step 8.5 | Weighted per `CHATBOT_CONVERSATION_TYPES` | Different users get different conversation mixes |
| Multi-turn vs. minimal | Step 8.5 | `MULTITURN_PROBABILITY = 0.80` | 80% full conversations, 20% minimal |
| Negative special treatment | Step 8.5 | `EXPLICIT_NEG_SPECIAL_FRACTION = 0.70` | 70% ask-to-forget/correction, 30% standard |

> The principle is: **establish the skeleton first, add noise last.** The preference set is the same regardless of which app it appears on. The noise makes the data look realistic without corrupting the ground truth.

---

## 17. Key Thresholds Reference

| Constant | Value | Purpose |
|----------|-------|---------|
| `MIN_PERSONA_INIT_CONFIDENCE` | 0.5 | Init filter floor — main knob for preference-list size |
| `HIGH_CONFIDENCE_INIT_THRESHOLD` | 0.5 | High-confidence predicate (test-split eligibility) |
| `HIGH_CONFIDENCE_CROSS_REF_THRESHOLD` | 10.0 | High-confidence predicate (corroboration floor) |
| `MIN_IMPLICIT_NEGATIVE_REPETITION` | 5 | Distinct source rows for implicit-only negative to survive |
| `IMPLICIT_NEGATIVE_PREFILTER_K` | 3 | Rows per hashtag signature required before LLM call |
| `MIN_PREF_CORROBORATION` | 2 | Independent hot-hashtag LLM calls needed for preference |
| `MIN_TEMPORAL_DAYS` | 3 | Calendar days negative rows must span |
| `SESSION_GAP_SECONDS` | 5 | Timestamp gap threshold for session grouping |
| `NOISE_REASSIGN_PROBABILITY` | 0.08 | Per-session app reassignment rate |
| `noise_strength` | 0.6 | Lognormal perturbation intensity for action weights |
| `IMPL_NEG_WEIGHT` | 1.0 | Net-sentiment: weight per implicit_negative row |
| `EXPL_POS_WEIGHT` | 3.0 | Net-sentiment: weight per explicit_positive row |
| `IMPL_POS_WEIGHT` | 1.5 | Net-sentiment: weight per implicit_positive row |
| `FADE_THRESHOLD_SECONDS` | 172,800 (48h) | Inactivity threshold for marking "faded" |
| `MAX_REINFORCED_ENTRIES` | 5 | Max recurrence samples in update history |
| `bottom_20_min_exempt` | 10.0 | Bottom-20% filter: exempt if cross_ref exceeds this |
| `MULTITURN_PROBABILITY` | 0.80 | Fraction of chatbot prefs getting full conversations |
| `EXPLICIT_NEG_SPECIAL_FRACTION` | 0.70 | Fraction of negative chatbot prefs getting special treatment |
| Test fraction | 0.20 | Latest 20% of high-confidence positives |
| Distractor shortlist | 5 | Candidates for over-personalization distractor |
| `MIN_HIDDEN_PERSONA_ROWS` | 20 | Minimum distinct source rows for a hidden persona cluster |
| `MIN_HIDDEN_PERSONA_DAYS` | 3 | Minimum temporal spread for hidden persona cluster |
| `HIDDEN_PERSONA_HASHTAG_MIN_FREQ` | 3 | Minimum hashtag occurrences to be considered |
| `HIDDEN_PERSONA_TOP_HASHTAGS` | 200 | Number of top hashtags passed to LLM for clustering |

> **Note:** Several thresholds (especially the high-confidence predicate values) are declared **tentative** and will be tuned empirically once real-scale statistics are available.
