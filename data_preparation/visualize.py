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
    --bg: #FAFAF9;
    --bg-card: #FFFFFF;
    --text: #1A1A1A;
    --text-secondary: #6B6B6B;
    --accent: #D97757;
    --accent-light: #F4E8E2;
    --accent-contradictory: #B8336A;
    --accent-similar: #2D936C;
    --border: #E8E8E6;
    --radius: 12px;
    --shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-hover: 0 4px 12px rgba(0,0,0,0.08);
    --font: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", "Segoe UI", Roboto, sans-serif;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: var(--font); background: var(--bg); color: var(--text); line-height: 1.6; -webkit-font-smoothing: antialiased; }}
  .container {{ max-width: 960px; margin: 0 auto; padding: 48px 24px; }}

  .header {{ margin-bottom: 48px; }}
  .header h1 {{ font-size: 32px; font-weight: 600; letter-spacing: -0.5px; margin-bottom: 8px; }}
  .header .meta {{ color: var(--text-secondary); font-size: 14px; }}
  .header .meta span {{ margin-right: 20px; }}

  .profile-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 48px; box-shadow: var(--shadow); }}
  .profile-card h2 {{ font-size: 20px; font-weight: 600; margin-bottom: 12px; }}
  .profile-card .bio {{ font-size: 15px; line-height: 1.6; margin-bottom: 16px; color: var(--text); }}
  .profile-card .details {{ font-size: 13px; color: var(--text-secondary); }}
  .profile-card .details span {{ margin-right: 16px; }}
  .profile-card .big-five {{ display: flex; gap: 12px; margin-top: 12px; flex-wrap: wrap; }}
  .profile-card .b5-item {{ font-size: 12px; padding: 4px 10px; border-radius: 20px; background: #F0F0EE; color: var(--text-secondary); }}

  .section {{ margin-bottom: 48px; }}
  .section-title {{ font-size: 20px; font-weight: 600; letter-spacing: -0.3px; margin-bottom: 20px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }}

  .persona-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .persona-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; box-shadow: var(--shadow); transition: box-shadow 0.2s ease, transform 0.2s ease; }}
  .persona-card:hover {{ box-shadow: var(--shadow-hover); transform: translateY(-1px); }}
  .persona-card .item-text {{ font-size: 15px; font-weight: 500; margin-bottom: 10px; line-height: 1.4; }}
  .persona-card .meta-line {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }}

  .confidence-row {{ display: flex; align-items: center; margin-bottom: 6px; font-size: 12px; color: var(--text-secondary); }}
  .confidence-row .label {{ width: 90px; flex-shrink: 0; }}
  .confidence-bar-track {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin: 0 8px; }}
  .confidence-bar-fill {{ height: 100%; border-radius: 2px; transition: width 0.4s ease; }}
  .confidence-bar-fill.init {{ background: var(--accent); }}
  .confidence-bar-fill.cross {{ background: var(--accent-similar); }}
  .confidence-row .value {{ width: 32px; text-align: right; font-variant-numeric: tabular-nums; }}

  .badge {{ display: inline-block; font-size: 11px; font-weight: 500; padding: 2px 10px; border-radius: 20px; margin-top: 6px; margin-right: 4px; }}
  .badge.category {{ background: #EDE9FE; color: #6D28D9; }}
  .badge.similar {{ background: #E6F4EE; color: var(--accent-similar); }}
  .badge.contradictory {{ background: #F8E4EE; color: var(--accent-contradictory); }}
  .badge.none {{ background: #F0F0EE; color: var(--text-secondary); }}
  .badge.stereotypical {{ background: #FEF3C7; color: #92400E; }}
  .badge.anti-stereotypical {{ background: #DBEAFE; color: #1E40AF; }}
  .badge.test {{ background: #FCE7F3; color: #9D174D; }}
  .badge.train {{ background: #EFF6FF; color: #1D4ED8; }}
  .badge.distractor {{ background: #FEE2E2; color: #991B1B; }}
  .badge.platform {{ background: #F0FDF4; color: #166534; }}
  .badge.action {{ background: #FEF3C7; color: #78350F; }}
  .badge.ai-msg {{ background: #E0E7FF; color: #3730A3; }}

  .app-section {{ margin-bottom: 36px; }}
  .app-section h3 {{ font-size: 16px; font-weight: 600; margin-bottom: 14px; color: var(--text); }}
  .app-persona-block {{ background: var(--accent-light); border-radius: 10px; padding: 12px 16px; margin-bottom: 14px; font-size: 12px; color: var(--text-secondary); line-height: 1.5; }}
  .app-persona-block .ap-title {{ font-weight: 600; color: var(--text); margin-bottom: 4px; }}
  .user-message {{ margin-top: 10px; padding: 10px 12px; background: #EEF2FF; border-left: 3px solid #6366F1; border-radius: 6px; font-size: 13px; color: #1E1B4B; font-style: italic; }}

  .empty {{ text-align: center; padding: 40px; color: var(--text-secondary); font-size: 14px; }}
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
    <div class="section-title">Preferences (grouped by app)</div>
    <div id="app-sections"></div>
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

// -- Per-app sections --
const appsContainer = document.getElementById('app-sections');
if (prefsData.length === 0) {{
  appsContainer.innerHTML = '<div class="empty">No preferences available.</div>';
}} else {{
  const byApp = {{}};
  APPS.forEach(a => byApp[a] = []);
  prefsData.forEach(p => {{
    const a = p.assigned_app || 'Instagram';
    if (!byApp[a]) byApp[a] = [];
    byApp[a].push(p);
  }});

  const appPersonas = (profileData && profileData.app_personas) || {{}};

  APPS.forEach(app => {{
    const rows = byApp[app] || [];
    if (rows.length === 0) return;
    const section = document.createElement('div');
    section.className = 'app-section';

    // Header + app persona block
    let personaBlock = '';
    const ap = appPersonas[app];
    if (ap) {{
      const purposes = (ap.use_purposes || []).join(' · ');
      const zones = (ap.friend_zones || []).join(' · ');
      const topics = (ap.topical_focus || []).join(' · ');
      personaBlock = `
        <div class="app-persona-block">
          <div class="ap-title">${{app}} persona — ${{ap.audience_type || ''}}, posts ${{ap.posting_frequency || ''}}</div>
          <div>${{ap.style_description || ''}}</div>
          <div style="margin-top:6px;"><b>Uses:</b> ${{purposes}}</div>
          <div><b>Audience:</b> ${{zones}}</div>
          <div><b>Topics:</b> ${{topics}}</div>
        </div>
      `;
    }}

    section.innerHTML = `<h3>${{app}} · ${{rows.length}} preferences</h3>${{personaBlock}}<div class="persona-grid"></div>`;
    const grid = section.querySelector('.persona-grid');

    rows.forEach(p => {{
      const card = document.createElement('div');
      card.className = 'persona-card';
      const relClass = p.relationship_type === 'similar' ? 'similar' : p.relationship_type === 'contradictory' ? 'contradictory' : 'none';
      let badges = `<span class="badge category">${{p.category}}</span>`;
      badges += `<span class="badge ${{p.split}}">${{p.split}}</span>`;
      if (p.relationship_type !== 'none') badges += `<span class="badge ${{relClass}}">${{p.relationship_type}}</span>`;
      if (p.stereotype_mark !== 'neutral') badges += `<span class="badge ${{p.stereotype_mark}}">${{p.stereotype_mark}}</span>`;

      const fmt = p.interaction_format || {{}};
      if (fmt.action_label) badges += `<span class="badge action">${{fmt.action_label}}</span>`;

      let userMsgBlock = '';
      if (fmt.user_message) {{
        userMsgBlock = `<div class="user-message">${{fmt.user_message}}</div>`;
      }}

      let distractorLine = '';
      if (p.split === 'test' && p.over_personalization_irrelevant) {{
        distractorLine = `<div class="meta-line" style="margin-top:10px;"><span class="badge distractor">distractor</span> ${{p.over_personalization_irrelevant}} <span style="opacity:0.6;">(${{p.over_personalization_irrelevant_category}})</span></div>`;
      }}

      card.innerHTML = `
        <div class="item-text">${{p.persona_item}}</div>
        <div class="meta-line">${{p.formatted_timestamp}} &middot; ${{p.source_interaction_type}}</div>
        <div class="confidence-row">
          <span class="label">Initial</span>
          <div class="confidence-bar-track"><div class="confidence-bar-fill init" style="width:${{(p.confidence_score_init*100).toFixed(0)}}%"></div></div>
          <span class="value">${{p.confidence_score_init.toFixed(2)}}</span>
        </div>
        <div class="confidence-row">
          <span class="label">Cross-ref</span>
          <div class="confidence-bar-track"><div class="confidence-bar-fill cross" style="width:${{Math.min(p.confidence_cross_referenced*100, 100).toFixed(0)}}%"></div></div>
          <span class="value">${{p.confidence_cross_referenced.toFixed(2)}}</span>
        </div>
        ${{badges}}
        ${{userMsgBlock}}
        ${{distractorLine}}
      `;
      grid.appendChild(card);
    }});

    appsContainer.appendChild(section);
  }});
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
