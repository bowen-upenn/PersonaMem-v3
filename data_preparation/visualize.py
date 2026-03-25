"""
Generates a standalone HTML persona visualization for a user.

Design: minimalist, Apple/Anthropic-inspired aesthetic.
No external dependencies — pure HTML/CSS/JS.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone

from data_preparation import utils


def generate_persona_html(user_id: str, backend_dir: str = "backend") -> str:
    """Read backend CSVs for a user and produce a self-contained HTML file.

    Returns the output file path.
    """
    base = os.path.join(backend_dir, f"{user_id}")

    # Load data
    atomic_rows = utils.load_rows_from_csv(f"{base}_raw.csv")
    cross_rows = utils.load_rows_from_csv(f"{base}_cross_referenced.csv")
    temporal_rows = utils.load_rows_from_csv(f"{base}_temporal.csv")

    # Build temporal groups
    temporal_groups = {}
    for row in temporal_rows:
        topic = row.get("topic", "unknown")
        if topic not in temporal_groups:
            temporal_groups[topic] = {
                "topic": topic,
                "interpretation": row.get("interpretation", ""),
                "timeline": [],
            }
        temporal_groups[topic]["timeline"].append({
            "persona_item": row.get("persona_item", ""),
            "formatted_timestamp": row.get("formatted_timestamp", ""),
            "confidence_score_init": float(row.get("confidence_score_init", 0)),
            "confidence_cross_referenced": float(row.get("confidence_cross_referenced", 0)),
        })

    now_str = datetime.now(tz=timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    # Serialize data for JS
    cross_json = json.dumps([
        {
            "category": r.get("category", "uncategorized"),
            "persona_item": r.get("persona_item", ""),
            "confidence_score_init": float(r.get("confidence_score_init", 0)),
            "confidence_cross_referenced": float(r.get("confidence_cross_referenced", 0)),
            "relationship_type": r.get("relationship_type", "none"),
            "related_personas": json.loads(r.get("related_personas", "[]")) if isinstance(r.get("related_personas", "[]"), str) else r.get("related_personas", []),
            "formatted_timestamp": r.get("formatted_timestamp", ""),
            "source_interaction_type": r.get("source_interaction_type", ""),
        }
        for r in cross_rows
    ])
    temporal_json = json.dumps(list(temporal_groups.values()))
    atomic_json = json.dumps([
        {
            "category": r.get("category", "uncategorized"),
            "persona_item": r.get("persona_item", ""),
            "confidence_score_init": float(r.get("confidence_score_init", 0)),
            "formatted_timestamp": r.get("formatted_timestamp", ""),
            "source_interaction_type": r.get("source_interaction_type", ""),
            "source_hashtags": r.get("source_hashtags", "[]"),
        }
        for r in atomic_rows
    ])

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

  body {{
    font-family: var(--font);
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}

  .container {{
    max-width: 960px;
    margin: 0 auto;
    padding: 48px 24px;
  }}

  /* Header */
  .header {{
    margin-bottom: 48px;
  }}
  .header h1 {{
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
  }}
  .header .meta {{
    color: var(--text-secondary);
    font-size: 14px;
  }}
  .header .meta span {{
    margin-right: 20px;
  }}

  /* Section */
  .section {{
    margin-bottom: 48px;
  }}
  .section-title {{
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.3px;
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  /* Persona Cards */
  .persona-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }}
  .persona-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
  }}
  .persona-card:hover {{
    box-shadow: var(--shadow-hover);
    transform: translateY(-1px);
  }}
  .persona-card .item-text {{
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 12px;
    line-height: 1.4;
  }}
  .persona-card .timestamp {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 12px;
  }}

  /* Confidence bars */
  .confidence-row {{
    display: flex;
    align-items: center;
    margin-bottom: 6px;
    font-size: 12px;
    color: var(--text-secondary);
  }}
  .confidence-row .label {{
    width: 90px;
    flex-shrink: 0;
  }}
  .confidence-bar-track {{
    flex: 1;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin: 0 8px;
  }}
  .confidence-bar-fill {{
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
  }}
  .confidence-bar-fill.init {{
    background: var(--accent);
  }}
  .confidence-bar-fill.cross {{
    background: var(--accent-similar);
  }}
  .confidence-row .value {{
    width: 32px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}

  /* Badges */
  .badge {{
    display: inline-block;
    font-size: 11px;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 20px;
    margin-top: 8px;
    margin-right: 4px;
  }}
  .badge.similar {{
    background: #E6F4EE;
    color: var(--accent-similar);
  }}
  .badge.contradictory {{
    background: #F8E4EE;
    color: var(--accent-contradictory);
  }}
  .badge.none {{
    background: #F0F0EE;
    color: var(--text-secondary);
  }}
  .badge.category {{
    background: #EDE9FE;
    color: #6D28D9;
  }}

  /* Timeline */
  .timeline-group {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
  }}
  .timeline-group .topic {{
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 4px;
  }}
  .timeline-group .interpretation {{
    font-size: 13px;
    color: var(--text-secondary);
    margin-bottom: 16px;
    font-style: italic;
  }}
  .timeline {{
    position: relative;
    padding-left: 28px;
  }}
  .timeline::before {{
    content: '';
    position: absolute;
    left: 8px;
    top: 4px;
    bottom: 4px;
    width: 2px;
    background: var(--border);
    border-radius: 1px;
  }}
  .timeline-node {{
    position: relative;
    margin-bottom: 20px;
  }}
  .timeline-node:last-child {{
    margin-bottom: 0;
  }}
  .timeline-node::before {{
    content: '';
    position: absolute;
    left: -24px;
    top: 6px;
    width: 10px;
    height: 10px;
    background: var(--accent);
    border: 2px solid var(--bg-card);
    border-radius: 50%;
    box-shadow: 0 0 0 2px var(--accent);
  }}
  .timeline-node .node-time {{
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 2px;
  }}
  .timeline-node .node-text {{
    font-size: 14px;
    font-weight: 500;
  }}
  .timeline-node .node-scores {{
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }}

  /* Collapsible raw data */
  .collapsible-header {{
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    user-select: none;
  }}
  .collapsible-header .arrow {{
    transition: transform 0.2s ease;
    font-size: 12px;
  }}
  .collapsible-header.open .arrow {{
    transform: rotate(90deg);
  }}
  .collapsible-body {{
    display: none;
    margin-top: 16px;
  }}
  .collapsible-body.open {{
    display: block;
  }}

  /* Table */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  .data-table th {{
    text-align: left;
    padding: 8px 12px;
    border-bottom: 2px solid var(--border);
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .data-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }}
  .data-table tr:last-child td {{
    border-bottom: none;
  }}
  .data-table tr:hover td {{
    background: var(--accent-light);
  }}

  /* Empty state */
  .empty {{
    text-align: center;
    padding: 40px;
    color: var(--text-secondary);
    font-size: 14px;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>User {user_id}</h1>
    <div class="meta">
      <span>{len(cross_rows)} personas</span>
      <span>{len(atomic_rows)} raw inferences</span>
      <span>{len(temporal_groups)} temporal topics</span>
      <span>Generated {now_str}</span>
    </div>
  </div>

  <!-- Persona Overview -->
  <div class="section">
    <div class="section-title">Persona Overview</div>
    <div id="persona-grid" class="persona-grid"></div>
  </div>

  <!-- Temporal Evolution -->
  <div class="section">
    <div class="section-title">Temporal Evolution</div>
    <div id="temporal-container"></div>
  </div>

  <!-- Raw Data -->
  <div class="section">
    <div class="collapsible-header" id="raw-toggle" onclick="toggleRaw()">
      <span class="arrow">&#9654;</span>
      <span class="section-title" style="border:none;margin:0;padding:0;">Raw Atomic Personas ({len(atomic_rows)})</span>
    </div>
    <div class="collapsible-body" id="raw-body"></div>
  </div>

</div>

<script>
const crossData = {cross_json};
const temporalData = {temporal_json};
const atomicData = {atomic_json};

// -- Persona cards --
const grid = document.getElementById('persona-grid');
if (crossData.length === 0) {{
  grid.innerHTML = '<div class="empty">No cross-referenced personas available.</div>';
}} else {{
  crossData.sort((a, b) => (b.confidence_score_init + b.confidence_cross_referenced) - (a.confidence_score_init + a.confidence_cross_referenced));
  crossData.forEach(p => {{
    const card = document.createElement('div');
    card.className = 'persona-card';
    const badgeClass = p.relationship_type === 'similar' ? 'similar' : p.relationship_type === 'contradictory' ? 'contradictory' : 'none';
    card.innerHTML = `
      <div class="item-text">${{p.persona_item}}</div>
      <div class="timestamp">${{p.formatted_timestamp}} &middot; ${{p.source_interaction_type}}</div>
      <span class="badge category">${{p.category}}</span>
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
      <span class="badge ${{badgeClass}}">${{p.relationship_type}}</span>
    `;
    grid.appendChild(card);
  }});
}}

// -- Temporal timeline --
const tc = document.getElementById('temporal-container');
if (temporalData.length === 0) {{
  tc.innerHTML = '<div class="empty">No contradictions detected — no temporal evolution to display.</div>';
}} else {{
  temporalData.forEach(group => {{
    const div = document.createElement('div');
    div.className = 'timeline-group';
    let nodesHtml = '';
    group.timeline.forEach(n => {{
      nodesHtml += `
        <div class="timeline-node">
          <div class="node-time">${{n.formatted_timestamp}}</div>
          <div class="node-text">${{n.persona_item}}</div>
          <div class="node-scores">init: ${{n.confidence_score_init.toFixed(2)}} &middot; cross-ref: ${{n.confidence_cross_referenced.toFixed(2)}}</div>
        </div>`;
    }});
    div.innerHTML = `
      <div class="topic">${{group.topic}}</div>
      <div class="interpretation">${{group.interpretation}}</div>
      <div class="timeline">${{nodesHtml}}</div>
    `;
    tc.appendChild(div);
  }});
}}

// -- Raw data table --
function toggleRaw() {{
  const header = document.getElementById('raw-toggle');
  const body = document.getElementById('raw-body');
  header.classList.toggle('open');
  body.classList.toggle('open');

  if (body.innerHTML === '') {{
    if (atomicData.length === 0) {{
      body.innerHTML = '<div class="empty">No raw data available.</div>';
      return;
    }}
    let rows = '';
    atomicData.forEach(r => {{
      rows += `<tr>
        <td>${{r.category}}</td>
        <td>${{r.persona_item}}</td>
        <td>${{r.confidence_score_init.toFixed(2)}}</td>
        <td>${{r.formatted_timestamp}}</td>
        <td>${{r.source_interaction_type}}</td>
      </tr>`;
    }});
    body.innerHTML = `<table class="data-table">
      <thead><tr>
        <th>Category</th><th>Persona</th><th>Init</th><th>Timestamp</th><th>Interaction</th>
      </tr></thead>
      <tbody>${{rows}}</tbody>
    </table>`;
  }}
}}
</script>
</body>
</html>"""

    output_path = os.path.join(backend_dir, f"{user_id}_persona.html")
    os.makedirs(backend_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"{utils.Colors.OKGREEN}Visualization saved to {output_path}{utils.Colors.ENDC}")
    return output_path
