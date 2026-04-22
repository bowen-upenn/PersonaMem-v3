"""
Generates a standalone HTML persona visualization for a user.

Reads backend/{user_id}/profile.json plus the four per-app JSON files
(instagram.json, facebook.json, threads.json, chatbot.json).

Supports both the new interaction-event format (nested preferences per
event) and the legacy flat format (one record per preference).

Design: minimalist, Apple/Anthropic-inspired aesthetic.
No external dependencies — pure HTML/CSS/JS.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

from data_preparation import utils


APPS = ["Instagram", "Facebook", "Threads", "Chatbot"]


def _load_app_events(user_dir: str) -> tuple[list[dict], list[dict]]:
    """Load per-app JSON files and return (events, flat_prefs).

    ``events`` is the interaction-event list (new format). Each event
    has event-level fields + a ``preferences`` list.

    ``flat_prefs`` is the flattened preference list (for backwards-compat
    counts and profile serialization).

    Both lists are sorted by source_timestamp ascending.
    """
    all_events: list[dict] = []
    flat_prefs: list[dict] = []

    for app in APPS:
        path = os.path.join(user_dir, app.lower() + ".json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            if "preferences" in entry:
                # New interaction-event format
                entry["_app"] = app
                all_events.append(entry)
                for pref in entry["preferences"]:
                    flat = dict(pref)
                    flat["assigned_app"] = app
                    flat["source_object_id"] = entry.get("source_object_id", "")
                    flat["source_timestamp"] = entry.get("source_timestamp", 0)
                    flat["formatted_timestamp"] = entry.get("formatted_timestamp", "")
                    flat["source_hashtags"] = entry.get("source_hashtags", [])
                    flat["source_interaction_type"] = entry.get("source_interaction_type", "")
                    flat["interaction_format"] = entry.get("interaction_format", {})
                    flat_prefs.append(flat)
            else:
                # Legacy flat format — wrap as single-pref event
                entry.setdefault("assigned_app", app)
                event = {
                    "source_object_id": entry.get("source_object_id", ""),
                    "source_timestamp": entry.get("source_timestamp", 0),
                    "formatted_timestamp": entry.get("formatted_timestamp", ""),
                    "source_hashtags": entry.get("source_hashtags", []),
                    "source_interaction_type": entry.get("source_interaction_type", ""),
                    "interaction_format": entry.get("interaction_format", {}),
                    "_app": app,
                    "preferences": [{
                        "persona_item": entry.get("persona_item", ""),
                        "category": entry.get("category", ""),
                        "confidence_score_init": entry.get("confidence_score_init", 0),
                        "confidence_cross_referenced": entry.get("confidence_cross_referenced", 0),
                        "stereotype_mark": entry.get("stereotype_mark", "neutral"),
                        "split": entry.get("split", ""),
                        "update_history": entry.get("update_history", []),
                        "over_personalization_irrelevant": entry.get("over_personalization_irrelevant", []),
                    }],
                    "conversation": entry.get("conversation"),
                    "conversation_type": entry.get("conversation_type"),
                    "ask_to_forget": entry.get("ask_to_forget", False),
                }
                all_events.append(event)
                flat_prefs.append(entry)

    all_events.sort(key=lambda e: (int(e.get("source_timestamp") or 0), e.get("source_object_id", "")))
    flat_prefs.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))
    return all_events, flat_prefs


# Keep legacy loader for any external callers
def _load_app_prefs(user_dir: str) -> list[dict]:
    """Load per-app JSONs and return a flat list of preferences (legacy compat)."""
    _, flat = _load_app_events(user_dir)
    return flat


def _load_profile(user_dir: str) -> dict | None:
    path = os.path.join(user_dir, "profile.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_persona_html(user_id: str, backend_dir: str = "backend") -> str:
    """Read backend/{user_id}/ JSON files and produce a self-contained HTML file."""
    user_dir = os.path.join(backend_dir, str(user_id))

    profile = _load_profile(user_dir)
    events, flat_prefs = _load_app_events(user_dir)

    now_str = datetime.now(tz=timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    # Serialize events for JS
    events_json = json.dumps(events)
    profile_json = json.dumps(profile) if profile else "null"

    # Counts
    n_events = len(events)
    n_prefs = len(flat_prefs)
    n_unique = len(set(r.get("persona_item", "") for r in flat_prefs))
    n_stereo = sum(1 for r in flat_prefs if r.get("stereotype_mark") == "stereotypical")
    n_anti = sum(1 for r in flat_prefs if r.get("stereotype_mark") == "anti-stereotypical")
    n_test = sum(1 for r in flat_prefs if r.get("split") == "test")
    n_train = n_prefs - n_test
    per_app_counts = {}
    for app in APPS:
        per_app_counts[app] = sum(1 for e in events if e.get("_app") == app)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Persona — User {user_id}</title>
<style>
  :root {{
    --bg: #F7F7F5;
    --bg-card: #FFFFFF;
    --text: #1D1D1F;
    --text-secondary: #86868B;
    --text-tertiary: #AEAEB2;
    --border: #E5E5EA;
    --radius: 10px;
    --shadow: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-hover: 0 2px 8px rgba(0,0,0,0.07);
    --font: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }}
  .container {{ max-width: 860px; margin: 0 auto; padding: 56px 24px; }}

  .header {{ margin-bottom: 40px; }}
  .header h1 {{ font-size: 28px; font-weight: 600; letter-spacing: -0.4px; margin-bottom: 6px; color: var(--text); }}
  .header .meta {{ color: var(--text-secondary); font-size: 13px; display: flex; flex-wrap: wrap; gap: 6px 18px; }}

  .profile-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .profile-card h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 10px; letter-spacing: -0.2px; }}
  .profile-card .bio {{ font-size: 14px; line-height: 1.65; margin-bottom: 14px; color: var(--text); }}
  .profile-card .details {{ font-size: 12px; color: var(--text-secondary); }}
  .profile-card .details span {{ margin-right: 14px; }}
  .profile-card .big-five {{ display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }}
  .profile-card .b5-item {{ font-size: 11px; padding: 3px 10px; border-radius: 20px; background: #F2F2F7; color: var(--text-secondary); }}
  .profile-card .mbti {{ margin-top: 10px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}

  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 16px; font-weight: 600; letter-spacing: -0.2px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--text); }}

  .event-grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .event-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 20px; box-shadow: var(--shadow); transition: box-shadow 0.15s ease; border-left: 3px solid var(--border); }}
  .event-card:hover {{ box-shadow: var(--shadow-hover); }}
  .event-card.app-Instagram {{ border-left-color: #C13584; }}
  .event-card.app-Facebook {{ border-left-color: #4A6FA5; }}
  .event-card.app-Threads {{ border-left-color: #636366; }}
  .event-card.app-Chatbot {{ border-left-color: #C8956C; }}
  .event-card.implicit-negative {{ background: #F0F0F0; border-left-color: #B0B0B0; opacity: 0.65; filter: grayscale(100%); }}
  .event-card.implicit-negative .event-meta {{ color: #999; }}
  .event-card.implicit-negative .event-header {{ border-bottom-color: #E0E0E0; }}
  .event-card.implicit-negative .hashtags {{ color: #888; }}
  .event-card.implicit-negative .badge {{ background: #E0E0E0 !important; color: #777 !important; }}
  .event-card.implicit-negative .pref-item {{ background: #E8E8E8; border-color: #D0D0D0; }}
  .event-card.implicit-negative .pref-item .item-text {{ color: #666; }}
  .event-card.implicit-negative .conf-inline {{ color: #999; }}

  .event-header {{ margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #F2F2F7; }}
  .event-header .event-meta {{ font-size: 11px; color: var(--text-secondary); margin-bottom: 4px; }}
  .event-header .hashtags {{ font-size: 12px; color: var(--text); margin-top: 4px; line-height: 1.5; }}

  .pref-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .content-block + .pref-list {{ margin-top: 14px; }}
  .pref-list-label {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #6B7280; margin-bottom: 2px; }}
  .pref-item {{ padding: 10px 14px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .pref-item .item-text {{ font-size: 13px; font-weight: 500; line-height: 1.45; color: var(--text); margin-bottom: 4px; }}
  .pref-item .pref-meta {{ font-size: 10px; color: var(--text-secondary); }}

  .update-history {{ margin-top: 6px; padding-left: 10px; border-left: 2px solid #E8E8ED; }}
  .update-entry {{ font-size: 10px; color: var(--text-secondary); margin-bottom: 2px; }}
  .update-entry .ut-type {{ font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }}
  .ut-reinforced {{ color: #2D6A4F; }}
  .ut-deepened {{ color: #1D4ED8; }}
  .ut-branched {{ color: #7C3AED; }}
  .ut-shifted {{ color: #B45309; }}
  .ut-intensified {{ color: #047857; }}
  .ut-contradicted {{ color: #B04050; }}
  .ut-faded {{ color: var(--text-tertiary); }}
  .ut-expanded {{ color: #1D4ED8; }}

  .conf-inline {{ font-size: 10px; color: var(--text-tertiary); font-variant-numeric: tabular-nums; }}
  .conf-inline span {{ margin-right: 10px; }}

  .badge {{ display: inline-block; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 4px; margin-right: 3px; letter-spacing: 0.1px; }}
  .badge.category {{ background: #F2F2F7; color: #636366; }}
  .badge.similar {{ background: #F2F2F7; color: #48854A; }}
  .badge.contradictory {{ background: #F2F2F7; color: #B04050; }}
  .badge.none {{ display: none; }}
  .badge.stereotypical {{ background: #FFF8E1; color: #8B6914; }}
  .badge.anti-stereotypical {{ background: #EEF2FF; color: #4A5DA8; }}
  .badge.test {{ background: #FDF2F8; color: #9B3068; }}
  .badge.train {{ background: #F2F2F7; color: var(--text-secondary); }}
  .badge.distractor {{ background: #FEF2F2; color: #9B2C2C; }}
  .badge.platform {{ font-weight: 600; font-size: 11px; padding: 2px 10px; }}
  .badge.platform.p-Instagram {{ background: #C13584; color: #fff; }}
  .badge.platform.p-Facebook {{ background: #4A6FA5; color: #fff; }}
  .badge.platform.p-Threads {{ background: #8E8E93; color: #fff; }}
  .badge.platform.p-Chatbot {{ background: #C8956C; color: #fff; }}
  .badge.action {{ background: #E8E8ED; color: #48484A; font-weight: 500; }}
  .badge.hidden-persona {{ background: #EDE9FE; color: #6D28D9; font-weight: 500; }}

  .hidden-section {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 22px; margin-bottom: 40px; box-shadow: var(--shadow); }}
  .hidden-section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 14px; letter-spacing: -0.2px; }}
  .hidden-summary {{ font-size: 13px; line-height: 1.7; color: var(--text); margin-bottom: 16px; padding: 12px 16px; background: #FAFAFA; border-radius: 8px; border-left: 3px solid #6D28D9; }}
  .hp-card {{ padding: 12px 16px; margin-bottom: 10px; border-radius: 8px; background: #FAFAFA; border: 1px solid #F2F2F7; }}
  .hp-card .hp-label {{ font-size: 14px; font-weight: 600; color: var(--text); margin-bottom: 2px; }}
  .hp-card .hp-type {{ font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px; color: #6D28D9; margin-bottom: 4px; }}
  .hp-card .hp-desc {{ font-size: 12px; color: var(--text); line-height: 1.6; margin-bottom: 6px; }}
  .hp-card .hp-meta {{ font-size: 10px; color: var(--text-secondary); }}
  .hp-card .hp-meta span {{ margin-right: 12px; }}
  .hp-card .hp-tags {{ font-size: 11px; color: var(--text-secondary); margin-top: 4px; }}
  .hp-card .hp-motivation {{ font-size: 11px; color: #6D28D9; margin-top: 4px; font-style: italic; }}
  .badge.interaction-type {{ font-weight: 600; padding: 2px 10px; }}
  .badge.interaction-type.explicit_positive {{ background: #D1FAE5; color: #065F46; }}
  .badge.interaction-type.implicit_positive {{ background: #EDF5E1; color: #3F6212; }}
  .badge.interaction-type.explicit_negative {{ background: #FEE2E2; color: #991B1B; }}
  .badge.interaction-type.implicit_negative {{ background: #FEF3C7; color: #92400E; }}

  .user-message {{ margin-top: 8px; padding: 8px 12px; background: #F2F2F7; border-left: 2px solid var(--text-tertiary); border-radius: 4px; font-size: 12px; color: var(--text); font-style: italic; }}

  /* Synthetic content (step 13b) rendering */
  .content-block {{ margin-top: 10px; padding: 12px 14px; background: #FAFAFC; border: 1px solid #ECECF1; border-radius: 8px; }}
  .content-block .c-type {{ font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #6B7280; margin-bottom: 6px; }}
  .content-block .c-type.c-text {{ color: #4A5DA8; }}
  .content-block .c-type.c-image {{ color: #9B3068; }}
  .content-block .c-type.c-short_video {{ color: #7C3AED; }}
  .content-block .c-title {{ font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .content-block .c-caption {{ font-size: 12px; color: var(--text); margin-bottom: 6px; line-height: 1.5; }}
  .content-block .c-desc {{ font-size: 12px; color: var(--text-secondary); line-height: 1.55; margin-bottom: 8px; font-style: italic; }}
  .content-block .c-text-body {{ font-size: 13px; color: var(--text); line-height: 1.65; white-space: pre-wrap; }}
  .content-block details {{ margin-top: 6px; }}
  .content-block details summary {{ font-size: 11px; color: var(--text-secondary); cursor: pointer; padding: 2px 0; user-select: none; }}
  .content-block details summary:hover {{ color: var(--text); }}
  .content-block details[open] summary {{ color: var(--text); }}
  .content-block .c-parts, .content-block .c-frames {{ margin-top: 6px; padding-left: 2px; }}
  .content-block .c-part, .content-block .c-frame {{ font-size: 11px; color: var(--text); padding: 4px 8px; margin-bottom: 2px; background: #F2F2F7; border-radius: 4px; line-height: 1.45; }}
  .content-block .c-part .region, .content-block .c-frame .ts {{ font-weight: 600; color: #636366; margin-right: 6px; font-variant-numeric: tabular-nums; }}
  .content-block .c-transcript {{ font-size: 11px; color: var(--text); padding: 6px 10px; margin-top: 6px; background: #F2F2F7; border-radius: 4px; line-height: 1.5; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: pre-wrap; }}
  .content-block .c-meta {{ margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .content-block .c-meta-chip {{ font-size: 10px; padding: 2px 8px; border-radius: 4px; background: #ECECF1; color: #636366; font-variant-numeric: tabular-nums; }}
  .event-card.implicit-negative .content-block {{ background: #ECECEC; border-color: #D8D8D8; }}

  .chat-thread {{ margin-top: 8px; display: flex; flex-direction: column; gap: 6px; }}
  .chat-bubble {{ max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 12px; line-height: 1.6; word-wrap: break-word; }}
  .chat-bubble.user-bubble {{ align-self: flex-end; background: #1B72E8; color: #fff; border-bottom-right-radius: 4px; }}
  .chat-bubble.assistant-bubble {{ align-self: flex-start; background: #E4E6EB; color: var(--text); border-bottom-left-radius: 4px; }}
  .chat-role {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
  .chat-bubble.user-bubble .chat-role {{ color: rgba(255,255,255,0.55); }}
  .chat-bubble.assistant-bubble .chat-role {{ color: var(--text-tertiary); }}
  .chat-conv-label {{ font-size: 10px; color: var(--text-tertiary); margin-top: 6px; margin-bottom: 2px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px; }}

  .empty {{ text-align: center; padding: 40px; color: var(--text-secondary); font-size: 13px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>User {user_id}</h1>
    <div class="meta">
      <span>{n_events} events</span>
      <span>{n_prefs} pref instances</span>
      <span>{n_unique} unique</span>
      <span>{n_train} train</span>
      <span>{n_test} test</span>
      <span>{n_stereo} stereo</span>
      <span>{n_anti} anti-stereo</span>
      <span>IG: {per_app_counts.get("Instagram", 0)}</span>
      <span>FB: {per_app_counts.get("Facebook", 0)}</span>
      <span>TH: {per_app_counts.get("Threads", 0)}</span>
      <span>AI: {per_app_counts.get("Chatbot", 0)}</span>
      <span>Generated {now_str}</span>
    </div>
  </div>

  <div id="profile-section"></div>
  <div id="hidden-personas-section"></div>

  <div class="section">
    <div class="section-title">Interaction Events (earliest &rarr; latest)</div>
    <div id="timeline-section"></div>
  </div>

</div>

<script>
const eventsData = {events_json};
const profileData = {profile_json};

// Label -> motivation lookup for hidden persona badge tooltips.
const hpMotivation = {{}};
if (profileData && profileData.hidden_personas) {{
  profileData.hidden_personas.forEach(hp => {{
    if (hp && hp.label) {{
      hpMotivation[hp.label] = hp.inferred_motivation || hp.description || '';
    }}
  }});
}}

// -- Profile card --
const ps = document.getElementById('profile-section');
if (profileData) {{
  const b5 = profileData.big_five || {{}};
  const b5Html = Object.entries(b5).map(([k,v]) => `<span class="b5-item">${{k}}: ${{v}}</span>`).join('');

  // MBTI block — reuse the Big Five chip style. One chip per dimension,
  // formatted as "axis: dominant letter". The MBTI type itself is just the
  // concatenation of these four letters, so we don't repeat it as a chip.
  let mbtiHtml = '';
  const mbti = profileData.mbti;
  if (mbti && mbti.dimensions) {{
    const dimOrder = ['E_I', 'S_N', 'T_F', 'J_P'];
    const dimChips = dimOrder.map(key => {{
      const d = mbti.dimensions[key];
      if (!d) return '';
      const [letterA, letterB] = key.split('_');
      const pA = Number(d[letterA] || 0);
      const pB = Number(d[letterB] || 0);
      const dominant = pA >= pB ? letterA : letterB;
      const pct = Math.round((pA >= pB ? pA : pB) * 100);
      const axis = `${{letterA}}/${{letterB}}`;
      const reason = (d.reason || '').replace(/"/g, '&quot;');
      return `<span class="b5-item" title="${{reason}}">${{axis}}: ${{dominant}} ${{pct}}%</span>`;
    }}).join('');
    if (dimChips) {{
      mbtiHtml = `<div class="mbti">${{dimChips}}</div>`;
    }}
  }}

  ps.innerHTML = `
    <div class="profile-card">
      <h2>${{profileData.name || ''}}</h2>
      <div class="bio">${{profileData.bio || ''}}</div>
      <div class="details">
        <span>${{profileData.gender || ''}}</span>
        <span>${{profileData.race_ethnicity || ''}}</span>
        <span>${{profileData.career || ''}}</span>
        <span>${{profileData.education || ''}}</span>
      </div>
      <div class="big-five">${{b5Html}}</div>
      ${{mbtiHtml}}
    </div>
  `;
}}

// -- Hidden Personas section --
const hps = document.getElementById('hidden-personas-section');
if (profileData && profileData.hidden_personas && profileData.hidden_personas.length > 0) {{
  let html = '<div class="hidden-section"><h2>Hidden Personas</h2>';

  // Summary paragraph
  if (profileData.hidden_persona_summary) {{
    html += `<div class="hidden-summary">${{profileData.hidden_persona_summary}}</div>`;
  }}

  // Individual hidden persona cards
  profileData.hidden_personas.forEach(hp => {{
    const tags = (hp.evidence_hashtags || []).join('  ');
    const ib = hp.interaction_breakdown || {{}};
    const ibStr = Object.entries(ib).map(([k,v]) => `${{k.replace(/_/g,' ')}}: ${{v}}`).join(' · ');
    const appDist = hp.app_distribution || {{}};
    const appStr = Object.entries(appDist).map(([k,v]) => `${{k}}: ${{v}}`).join(' · ');
    html += `
      <div class="hp-card">
        <div class="hp-type">${{hp.type || ''}}</div>
        <div class="hp-label">${{hp.label || ''}}</div>
        <div class="hp-desc">${{hp.description || ''}}</div>
        <div class="hp-meta">
          <span>${{hp.evidence_rows || 0}} rows (${{((hp.evidence_row_fraction || 0) * 100).toFixed(1)}}%)</span>
          <span>privacy: ${{((hp.privacy_ratio || 0) * 100).toFixed(0)}}%</span>
          <span>${{hp.temporal_spread_days || 0}} days</span>
          ${{appStr ? `<span>${{appStr}}</span>` : ''}}
        </div>
        <div class="hp-tags">${{tags}}</div>
        ${{hp.inferred_motivation ? `<div class="hp-motivation">"${{hp.inferred_motivation}}"</div>` : ''}}
      </div>
    `;
  }});

  html += '</div>';
  hps.innerHTML = html;
}}

// -- Render synthetic content (step 13b) --
// Produces an HTML block describing what the user saw on screen: the text,
// image, or short video. Returns empty string when the event has no content
// (Chatbot events and implicit_negative stubs).
function escapeHtml(s) {{
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}}

function renderContentMeta(meta) {{
  if (!meta || typeof meta !== 'object') return '';
  const chips = Object.entries(meta)
    .filter(([k, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => {{
      const val = typeof v === 'object' ? JSON.stringify(v) : String(v);
      return `<span class="c-meta-chip"><b>${{escapeHtml(k)}}</b> ${{escapeHtml(val)}}</span>`;
    }})
    .join('');
  return chips ? `<div class="c-meta">${{chips}}</div>` : '';
}}

function renderContent(ev) {{
  const ctype = ev.content_type;
  const content = ev.content;
  if (!ctype || !content || typeof content !== 'object') return '';

  const typeLabel = ctype.replace(/_/g, ' ');
  const header = `<div class="c-type c-${{ctype}}">${{typeLabel}}</div>`;

  if (ctype === 'text') {{
    return `<div class="content-block">${{header}}<div class="c-text-body">${{escapeHtml(content.text || '')}}</div></div>`;
  }}

  if (ctype === 'image') {{
    const caption = content.caption ? `<div class="c-caption">${{escapeHtml(content.caption)}}</div>` : '';
    const desc = content.overall_description ? `<div class="c-desc">${{escapeHtml(content.overall_description)}}</div>` : '';
    const parts = (content.parts || []).map(p =>
      `<div class="c-part"><span class="region">${{escapeHtml(p.region || '')}}</span>${{escapeHtml(p.description || '')}}</div>`
    ).join('');
    const partsBlock = parts
      ? `<details><summary>parts (${{content.parts.length}})</summary><div class="c-parts">${{parts}}</div></details>`
      : '';
    const metaBlock = renderContentMeta(content.metadata);
    const metaWrapped = metaBlock
      ? `<details><summary>metadata</summary>${{metaBlock}}</details>`
      : '';
    return `<div class="content-block">${{header}}${{caption}}${{desc}}${{partsBlock}}${{metaWrapped}}</div>`;
  }}

  if (ctype === 'short_video') {{
    const title = content.title ? `<div class="c-title">${{escapeHtml(content.title)}}</div>` : '';
    const caption = content.caption ? `<div class="c-caption">${{escapeHtml(content.caption)}}</div>` : '';
    const desc = content.overall_description ? `<div class="c-desc">${{escapeHtml(content.overall_description)}}</div>` : '';
    const frames = (content.key_frames || []).map(f => {{
      const ts = typeof f.timestamp_s === 'number' ? f.timestamp_s.toFixed(1) + 's' : String(f.timestamp_s || '');
      return `<div class="c-frame"><span class="ts">${{escapeHtml(ts)}}</span>${{escapeHtml(f.description || '')}}</div>`;
    }}).join('');
    const framesBlock = frames
      ? `<details open><summary>key frames (${{content.key_frames.length}})</summary><div class="c-frames">${{frames}}</div></details>`
      : '';
    const transcript = content.audio_transcript
      ? `<details><summary>audio transcript</summary><div class="c-transcript">${{escapeHtml(content.audio_transcript)}}</div></details>`
      : '';
    const metaBlock = renderContentMeta(content.metadata);
    const metaWrapped = metaBlock
      ? `<details><summary>metadata</summary>${{metaBlock}}</details>`
      : '';
    return `<div class="content-block">${{header}}${{title}}${{caption}}${{desc}}${{framesBlock}}${{transcript}}${{metaWrapped}}</div>`;
  }}

  return '';
}}

// -- Render update history --
function renderUpdateHistory(history) {{
  if (!history || !history.length) return '';
  const entries = history.map(h => {{
    const cls = 'ut-' + (h.update_type || 'expanded');
    let text = `<span class="ut-type ${{cls}}">${{h.update_type}}</span>`;
    if (h.preference) text += ` ${{h.preference}}`;
    if (h.description) text += ` — ${{h.description}}`;
    if (h.formatted_timestamp) text += ` <span style="opacity:0.6">(${{h.formatted_timestamp}})</span>`;
    if (h.source_app) text += ` <span class="badge platform p-${{h.source_app}}" style="font-size:9px;padding:1px 6px;">${{h.source_app}}</span>`;
    if (h.total_occurrences) text += ` <span style="opacity:0.6">[occ ${{h.occurrence}}/${{h.total_occurrences}}]</span>`;
    return `<div class="update-entry">${{text}}</div>`;
  }}).join('');
  return `<div class="update-history">${{entries}}</div>`;
}}

// -- Chronological timeline of interaction events --
const timeline = document.getElementById('timeline-section');
if (eventsData.length === 0) {{
  timeline.innerHTML = '<div class="empty">No interaction events available.</div>';
}} else {{
  const grid = document.createElement('div');
  grid.className = 'event-grid';

  eventsData.forEach((ev, idx) => {{
    const app = ev._app || 'Instagram';
    const fmt = ev.interaction_format || {{}};
    const prefs = ev.preferences || [];
    const hashtags = ev.source_hashtags || [];
    const itype = ev.source_interaction_type || '';
    const isImplicitNeg = itype === 'implicit_negative';

    const card = document.createElement('div');
    card.className = `event-card app-${{app}}${{isImplicitNeg ? ' implicit-negative' : ''}}`;

    // Event header
    let headerHtml = `
      <div class="event-header">
        <div class="event-meta">
          <span style="font-weight:600;color:var(--text);">Event #${{idx+1}}</span> &middot;
          ${{ev.formatted_timestamp || ''}} &middot;
          ${{prefs.length}} preference${{prefs.length !== 1 ? 's' : ''}}
        </div>
        <div>
          <span class="badge platform p-${{app}}">${{app}}</span>
          <span class="badge interaction-type ${{itype}}">${{itype.replace(/_/g, ' ')}}</span>
          ${{fmt.action_label ? `<span class="badge action">${{fmt.action_label}}</span>` : ''}}
        </div>
        ${{hashtags.length ? `<div class="hashtags">${{hashtags.join('  ')}}</div>` : ''}}
      </div>
    `;

    // Preferences list
    let prefsHtml = '<div class="pref-list">';
    if (prefs.length > 0) {{
      prefsHtml += '<div class="pref-list-label">Preferences</div>';
    }}
    prefs.forEach(p => {{
      let badges = `<span class="badge category">${{p.category || ''}}</span>`;
      if (p.split === 'test') badges += `<span class="badge test">test</span>`;
      if (p.stereotype_mark && p.stereotype_mark !== 'neutral') badges += `<span class="badge ${{p.stereotype_mark}}">${{p.stereotype_mark}}</span>`;
      if (p.hidden_persona_labels && p.hidden_persona_labels.length > 0) {{
        p.hidden_persona_labels.forEach(lbl => {{
          const motiv = (hpMotivation[lbl] || '').replace(/"/g, '&quot;');
          const titleAttr = motiv ? ` title="${{motiv}}"` : '';
          badges += `<span class="badge hidden-persona"${{titleAttr}}>${{lbl}}</span>`;
        }});
      }}

      let distractorLine = '';
      if (p.split === 'test' && Array.isArray(p.over_personalization_irrelevant) && p.over_personalization_irrelevant.length > 0) {{
        const distractorItems = p.over_personalization_irrelevant
          .map(d => (d && d.persona_item) ? d.persona_item : '')
          .filter(x => x)
          .join(' | ');
        if (distractorItems) {{
          distractorLine = `<div style="margin-top:4px;font-size:10px;"><span class="badge distractor">distractors</span> ${{distractorItems}}</div>`;
        }}
      }} else if (p.split === 'test' && typeof p.over_personalization_irrelevant === 'string' && p.over_personalization_irrelevant) {{
        distractorLine = `<div style="margin-top:4px;font-size:10px;"><span class="badge distractor">distractor</span> ${{p.over_personalization_irrelevant}}</div>`;
      }}

      const historyHtml = renderUpdateHistory(p.update_history);

      prefsHtml += `
        <div class="pref-item">
          <div class="item-text">${{p.persona_item || ''}}</div>
          <div class="conf-inline"><span>init ${{(p.confidence_score_init || 0).toFixed(2)}}</span><span>xref ${{(p.confidence_cross_referenced || 0).toFixed(1)}}</span></div>
          <div class="pref-meta">${{badges}}</div>
          ${{historyHtml}}
          ${{distractorLine}}
        </div>
      `;
    }});
    prefsHtml += '</div>';

    // Chatbot conversation
    let convHtml = '';
    if (ev.conversation && ev.conversation.length > 0) {{
      let convLabel = ev.conversation_type ? `<div class="chat-conv-label">${{ev.conversation_type.replace(/_/g, ' ')}}${{ev.ask_to_forget ? ' &middot; ask-to-forget' : ''}}</div>` : '';
      let bubbles = ev.conversation.map(t => {{
        const cls = t.role === 'user' ? 'user-bubble' : 'assistant-bubble';
        const label = t.role === 'user' ? 'You' : 'AI';
        return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{label}}</div>${{t.content}}</div>`;
      }}).join('');
      convHtml = `${{convLabel}}<div class="chat-thread">${{bubbles}}</div>`;
    }} else if (fmt.user_message) {{
      convHtml = `<div class="user-message">${{fmt.user_message}}</div>`;
    }}

    const contentHtml = renderContent(ev);
    card.innerHTML = headerHtml + contentHtml + prefsHtml + convHtml;
    grid.appendChild(card);
  }});

  timeline.appendChild(grid);
}}
</script>
</body>
</html>"""

    output_path = os.path.join(user_dir, "persona.html")
    os.makedirs(user_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"{utils.Colors.OKGREEN}Visualization saved to {output_path}{utils.Colors.ENDC}")
    return output_path
