"""Pipeline Extension B — post-process augmentation of backend/{user_id}/.

Runs AFTER the main 22-step pipeline. Adds:
- profile.json `friends[]` — named friend graph with deliberate name
  collision for T17 wrong-recipient probes.
- {app}.json self-authored post events (is_self_authored=True,
  author_id="self", source_interaction_type="explicit_positive").
- {app}.json DM thread entries — is_dm=true entries appended to each
  social app's main JSON, with the full messages[] embedded (no
  separate {app}_dms.json file anymore).
- trending.json — frozen trending-hashtag list for T19 probes.

Doesn't touch the core pipeline. Purely additive — can be re-run, deleted,
or replaced independently of persona_agent.py.

Entry point:
    python -m data_preparation.extension_b --user_id 115
"""

from data_preparation.extension_b.main import run_extension_b  # noqa: F401
