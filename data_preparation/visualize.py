"""
Generates a standalone HTML persona visualization for a user.

Reads backend/{user_id}/profile.json plus the four per-app JSON files
(instagram.json, facebook.json, threads.json, chatbot.json).

Design: minimalist, Apple/Anthropic-inspired aesthetic.
No external dependencies — pure HTML/CSS/JS.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

from data_preparation import utils


APPS = ["Instagram", "Facebook", "Threads", "Chatbot"]


def _load_app_prefs(user_dir: str) -> list[dict]:
    """Load the per-app JSON files and return a flat list of preferences
    tagged with the app they came from."""
    all_rows: list[dict] = []
    for app in APPS:
        path = os.path.join(user_dir, app.lower() + ".json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            # Ensure assigned_app is present
            r.setdefault("assigned_app", app)
            all_rows.append(r)
    all_rows.sort(key=lambda r: (int(r.get("source_timestamp") or 0), r.get("persona_item", "")))
    return all_rows


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
    pref_rows = _load_app_prefs(user_dir)

    now_str = datetime.now(tz=timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    # Serialize for JS
    prefs_json = json.dumps([
        {
            "persona_item": r.get("persona_item", ""),
            "category": r.get("category", "uncategorized"),
            "confidence_score_init": float(r.get("confidence_score_init", 0) or 0),
            "confidence_cross_referenced": float(r.get("confidence_cross_referenced", 0) or 0),
            "relationship_type": r.get("relationship_type", "none"),
            "source_interaction_type": r.get("source_interaction_type", ""),
            "interaction_format": r.get("interaction_format", {}) or {},
            "formatted_timestamp": r.get("formatted_timestamp", ""),
            "source_timestamp": int(r.get("source_timestamp") or 0),
            "stereotype_mark": r.get("stereotype_mark", "neutral"),
            "split": r.get("split", "train") or "train",
            "over_personalization_irrelevant": r.get("over_personalization_irrelevant", ""),
            "over_personalization_irrelevant_category": r.get("over_personalization_irrelevant_category", ""),
            "assigned_app": r.get("assigned_app", ""),
            "conversation": r.get("conversation"),
            "conversation_type": r.get("conversation_type"),
            "ask_to_forget": r.get("ask_to_forget", False),
        }
        for r in pref_rows
    ])

    profile_json = json.dumps(profile) if profile else "null"

    # Counts
    n_stereo = sum(1 for r in pref_rows if r.get("stereotype_mark") == "stereotypical")
    n_anti = sum(1 for r in pref_rows if r.get("stereotype_mark") == "anti-stereotypical")
    n_test = sum(1 for r in pref_rows if r.get("split") == "test")
    n_train = sum(1 for r in pref_rows if (r.get("split") or "train") == "train")
    per_app_counts = {}
    for app in APPS:
        per_app_counts[app] = sum(1 for r in pref_rows if r.get("assigned_app") == app)

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
  .container {{ max-width: 820px; margin: 0 auto; padding: 56px 24px; }}

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

  .section {{ margin-bottom: 40px; }}
  .section-title {{ font-size: 16px; font-weight: 600; letter-spacing: -0.2px; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border); color: var(--text); }}

  .persona-grid {{ display: flex; flex-direction: column; gap: 8px; }}
  .persona-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 18px; box-shadow: var(--shadow); transition: box-shadow 0.15s ease; border-left: 3px solid var(--border); }}
  .persona-card:hover {{ box-shadow: var(--shadow-hover); }}
  /* Muted, elegant tints per app */
  .persona-card.app-Instagram {{ border-left-color: #C13584; background: #FDFAFE; }}
  .persona-card.app-Facebook {{ border-left-color: #4A6FA5; background: #F8FAFD; }}
  .persona-card.app-Threads {{ border-left-color: #636366; background: #FAFAFA; }}
  .persona-card.app-Chatbot {{ border-left-color: #C8956C; background: #FDFCFA; }}
  .persona-card .item-text {{ font-size: 14px; font-weight: 500; margin-bottom: 8px; line-height: 1.45; color: var(--text); }}
  .persona-card .meta-line {{ font-size: 11px; color: var(--text-secondary); margin-bottom: 3px; }}

  .conf-inline {{ font-size: 10px; color: var(--text-tertiary); font-variant-numeric: tabular-nums; margin-bottom: 6px; }}
  .conf-inline span {{ margin-right: 12px; }}

  .badge {{ display: inline-block; font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 4px; margin-top: 4px; margin-right: 3px; letter-spacing: 0.1px; }}
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
  .badge.platform.p-Threads {{ background: #636366; color: #fff; }}
  .badge.platform.p-Chatbot {{ background: #C8956C; color: #fff; }}
  .badge.action {{ background: var(--text); color: #fff; font-weight: 500; }}
  .badge.interaction-type {{ font-weight: 600; padding: 2px 10px; }}
  .badge.interaction-type.explicit_positive {{ background: #D1FAE5; color: #065F46; }}
  .badge.interaction-type.implicit_positive {{ background: #EDF5E1; color: #3F6212; }}
  .badge.interaction-type.explicit_negative {{ background: #FEE2E2; color: #991B1B; }}
  .badge.interaction-type.implicit_negative {{ background: #FEF3C7; color: #92400E; }}

  .user-message {{ margin-top: 10px; padding: 10px 12px; background: #F2F2F7; border-left: 2px solid var(--text-tertiary); border-radius: 4px; font-size: 12px; color: var(--text); font-style: italic; }}

  .chat-thread {{ margin-top: 10px; display: flex; flex-direction: column; gap: 6px; }}
  .chat-bubble {{ max-width: 85%; padding: 10px 14px; border-radius: 14px; font-size: 12px; line-height: 1.6; word-wrap: break-word; }}
  .chat-bubble.user-bubble {{ align-self: flex-end; background: var(--text); color: #fff; border-bottom-right-radius: 4px; }}
  .chat-bubble.assistant-bubble {{ align-self: flex-start; background: #F2F2F7; color: var(--text); border-bottom-left-radius: 4px; }}
  .chat-role {{ font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 2px; }}
  .chat-bubble.user-bubble .chat-role {{ color: rgba(255,255,255,0.55); }}
  .chat-bubble.assistant-bubble .chat-role {{ color: var(--text-tertiary); }}
  .chat-conv-label {{ font-size: 10px; color: var(--text-tertiary); margin-top: 8px; margin-bottom: 2px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px; }}

  .empty {{ text-align: center; padding: 40px; color: var(--text-secondary); font-size: 13px; }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>User {user_id}</h1>
    <div class="meta">
      <span>{len(pref_rows)} preferences</span>
      <span>{n_train} train</span>
      <span>{n_test} test</span>
      <span>{n_stereo} stereotypical</span>
      <span>{n_anti} anti-stereotypical</span>
      <span>IG: {per_app_counts.get("Instagram", 0)}</span>
      <span>FB: {per_app_counts.get("Facebook", 0)}</span>
      <span>TH: {per_app_counts.get("Threads", 0)}</span>
      <span>AI: {per_app_counts.get("Chatbot", 0)}</span>
      <span>Generated {now_str}</span>
    </div>
  </div>

  <div id="profile-section"></div>

  <div class="section">
    <div class="section-title">All Preferences (earliest &rarr; latest)</div>
    <div id="timeline-section"></div>
  </div>

</div>

<script>
const prefsData = {prefs_json};
const profileData = {profile_json};

const APPS = ['Instagram', 'Facebook', 'Threads', 'Chatbot'];

// -- Profile card --
const ps = document.getElementById('profile-section');
if (profileData) {{
  const b5 = profileData.big_five || {{}};
  const b5Html = Object.entries(b5).map(([k,v]) => `<span class="b5-item">${{k}}: ${{v}}</span>`).join('');
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
    </div>
  `;
}}

// -- Chronological timeline --
const timeline = document.getElementById('timeline-section');
if (prefsData.length === 0) {{
  timeline.innerHTML = '<div class="empty">No preferences available.</div>';
}} else {{
  // prefsData is already sorted by source_timestamp ascending
  const grid = document.createElement('div');
  grid.className = 'persona-grid';

  prefsData.forEach((p, idx) => {{
    const card = document.createElement('div');
    card.className = `persona-card app-${{p.assigned_app}}`;
    const relClass = p.relationship_type === 'similar' ? 'similar' : p.relationship_type === 'contradictory' ? 'contradictory' : 'none';
    const fmt = p.interaction_format || {{}};

    // Primary badges: app, interaction type, action — these are the most important
    let primaryBadges = `<span class="badge platform p-${{p.assigned_app}}">${{p.assigned_app}}</span>`;
    primaryBadges += `<span class="badge interaction-type ${{p.source_interaction_type}}">${{p.source_interaction_type.replace(/_/g, ' ')}}</span>`;
    if (fmt.action_label) primaryBadges += `<span class="badge action">${{fmt.action_label}}</span>`;

    // Secondary badges: category, split, etc.
    let secondaryBadges = `<span class="badge category">${{p.category}}</span>`;
    secondaryBadges += `<span class="badge ${{p.split}}">${{p.split}}</span>`;
    if (p.relationship_type !== 'none') secondaryBadges += `<span class="badge ${{relClass}}">${{p.relationship_type}}</span>`;
    if (p.stereotype_mark !== 'neutral') secondaryBadges += `<span class="badge ${{p.stereotype_mark}}">${{p.stereotype_mark}}</span>`;

    let userMsgBlock = '';
    if (p.conversation && p.conversation.length > 0) {{
      let convLabel = p.conversation_type ? `<div class="chat-conv-label">${{p.conversation_type.replace(/_/g, ' ')}}${{p.ask_to_forget ? ' · ask-to-forget' : ''}}</div>` : '';
      let bubbles = p.conversation.map(t => {{
        const cls = t.role === 'user' ? 'user-bubble' : 'assistant-bubble';
        const label = t.role === 'user' ? 'You' : 'AI';
        return `<div class="chat-bubble ${{cls}}"><div class="chat-role">${{label}}</div>${{t.content}}</div>`;
      }}).join('');
      userMsgBlock = `${{convLabel}}<div class="chat-thread">${{bubbles}}</div>`;
    }} else if (fmt.user_message) {{
      userMsgBlock = `<div class="user-message">${{fmt.user_message}}</div>`;
    }}

    let distractorLine = '';
    if (p.split === 'test' && p.over_personalization_irrelevant) {{
      distractorLine = `<div class="meta-line" style="margin-top:10px;"><span class="badge distractor">distractor</span> ${{p.over_personalization_irrelevant}} <span style="opacity:0.6;">(${{p.over_personalization_irrelevant_category}})</span></div>`;
    }}

    card.innerHTML = `
      <div class="meta-line" style="margin-bottom:4px;"><span style="font-weight:600;color:var(--text);">#${{idx+1}}</span> &middot; ${{p.formatted_timestamp}}</div>
      <div class="item-text">${{p.persona_item}}</div>
      <div style="margin-bottom:6px;">${{primaryBadges}}</div>
      <div class="conf-inline"><span>init ${{p.confidence_score_init.toFixed(2)}}</span><span>xref ${{p.confidence_cross_referenced.toFixed(0)}}</span></div>
      <div>${{secondaryBadges}}</div>
      ${{userMsgBlock}}
      ${{distractorLine}}
    `;
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
