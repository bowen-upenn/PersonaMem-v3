#!/usr/bin/env python3
"""Step 2a of the standardized pair audit (see AUDIT.md, Slice A).

Read the manifest produced by build_workunits.py and emit a self-contained
Claude Code Workflow script (JS) that judges every work-unit on 3 axes and
adversarially verifies every flag with 2 independent skeptics. The manifest is
embedded as an inline JS literal so the workflow needs no filesystem access to
enumerate work-units (only the per-work-unit agents read their own file).

Usage:
  python scripts/qa_pair_audit/gen_workflow.py --workdir /path/to/workdir
Then run the emitted <workdir>/verify_pairs_workflow.js via the Workflow tool,
and save its returned JSON to <workdir>/wf_result.json for aggregate.py.
"""
import argparse, json
from pathlib import Path

JS_BODY = r'''

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { rows: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: {
      query_id: { type: 'string' },
      naturalness_pass: { type: 'boolean' }, naturalness_reason: { type: 'string' },
      example_pass: { type: 'boolean' }, example_reason: { type: 'string' },
      inferior_pass: { type: 'boolean' }, inferior_reason: { type: 'string' },
      severity: { type: 'string', enum: ['none', 'minor', 'major'] },
    },
    required: ['query_id','naturalness_pass','naturalness_reason','example_pass','example_reason','inferior_pass','inferior_reason','severity'],
  }}}, required: ['rows'],
};
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: {
    query_id: { type: 'string' },
    naturalness_verdict: { type: 'string', enum: ['real_problem','refuted','na'] },
    example_verdict: { type: 'string', enum: ['real_problem','refuted','na'] },
    inferior_verdict: { type: 'string', enum: ['real_problem','refuted','na'] },
    reason: { type: 'string' },
  },
  required: ['query_id','naturalness_verdict','example_verdict','inferior_verdict','reason'],
};

function judgePrompt(wu) {
  return `You are a meticulous data-quality auditor for a persona-personalization benchmark.
Use your file tools to read the JSON work-unit file at this path:
  ${wu.file}
It contains: persona_card (who the user is), task_type, and rows[]. Each row has: user_query, prior_conversation (often null = single-turn), example_response (the GOLD answer), inferior_response {text, flaw_kind, flaw_evidence} (a deliberately-worse FOIL), groundtruth_preference, held_out_preference, top_k_relevant_prefs, privacy_flagged_prefs.

For EACH row rule PASS/FAIL on three INDEPENDENT axes. Be strict but fair — flag genuine defects, not stylistic nitpicks. A clean row passes all three. Use persona_card (career, bio, big_five, preferences) to judge whether the query is natural for THIS person and whether a cited preference is really theirs.

This file's task_type is: ${wu.task_type}. Apply the matching contract below.

=== chatbot_personalized_response  (the user SHOULD be personalized to) ===
The row has a held_out_preference. The GOLD should weave that preference in to give a more helpful, tailored answer; the FOIL (flaw_kind=missed_personalization) should be a plausible GENERIC answer that OMITS it.
- naturalness: PASS iff user_query is something THIS persona would really type to an AI chatbot — natural, casual, human phrasing; plausible motivation; consistent with the persona and prior_conversation. FAIL if stilted/templated, unnaturally self-disclosing, on-the-nose about the held-out preference (leaking the intended answer), implausible for this persona, or a non-sequitur.
- example (GOLD): PASS iff (a) example_response actually incorporates the held-out preference / the personal info (it is NOT just a generic answer), AND (b) that personalization genuinely makes the answer MORE helpful and fits the query naturally — the kind of tailoring a great assistant would do, not shoehorned, forced, creepy, or presumptuous. FAIL if the gold does not really use the personal info; uses it but it does not fit the query (forced/random); overdoes it (creepy/presumptuous); personalizes on a disliked / privacy-flagged / contradicted preference; personalizes on a preference NOT present in the persona (fabricated); or leans on an adjacent theme rather than the cited held-out item (label drift).
- inferior (FOIL): PASS iff inferior_response genuinely commits missed_personalization — a reasonable, on-topic, generic answer that does NOT use the held-out preference and is therefore clearly LESS ideal for this user than the gold. FAIL if the foil actually IS personalized (uses the held-out pref) or is as good/better than the gold; OR it is broken/off-topic/nonsensical/absurdly short (bad for a reason OTHER than missing personalization → unfair pair); OR it differs from the gold on a major non-personalization axis (length, format, a refusal) so the comparison is not clean.

=== over_personalization_chatbot_text  (the user should NOT be personalized to / RESTRAINT) ===
The query does NOT call for personalization. The GOLD should answer it generically with NO injected persona. The FOIL (flaw_kind=over_personalization) injects inferior_response.flaw_evidence.persona_item where it does not belong.
- naturalness: PASS iff user_query is a natural thing this persona would ask. Generic/everyday queries are fine and need NOT relate to the persona, but must still be plausible for THIS person to type. FAIL as above.
- example (GOLD, no-personalization): PASS iff (a) the query genuinely does NOT warrant personalization — a good assistant answering ANY user would give essentially the same generic answer; personalizing would be unwarranted or odd — AND (b) example_response is a solid, helpful GENERIC answer that does NOT inject the user's persona/preferences. FAIL if the query actually DOES call for personalization (so restraint is the wrong gold and the test is invalid); OR the gold itself sneaks in the persona; OR the gold is a poor/unhelpful answer.
- inferior (FOIL, over-personalized): PASS iff inferior_response genuinely OVER-personalizes — it injects flaw_evidence.persona_item (or another real persona interest) into the answer in an oblique, unprompted, unwarranted way a good assistant would NOT, the injected item is really THIS user's preference (check persona_card / top_k_relevant_prefs) and is NOT one they dislike, AND the injection is genuinely unwarranted for the query. FAIL if the claimed injection is NOT actually present in the foil text (word coincidence only); OR the injected item is fabricated (not this user's preference) or one the persona DISLIKES; OR the injection is actually reasonable/warranted for the query — e.g. the persona_item is the query's own topic (so it is not really over-personalization); OR the foil is broken for an unrelated reason (unfair pair).

Return {rows:[{query_id, naturalness_pass, naturalness_reason, example_pass, example_reason, inferior_pass, inferior_reason, severity}]} covering EVERY row in the file. severity in {none,minor,major}: none if all three pass; major if a defect makes the row invalid/unusable as a test; minor otherwise. Keep each reason to one or two sentences that cite the concrete text.`;
}

function verifyPrompt(wu, row, flagged, k) {
  const axesTxt = flagged.map(f => '- ' + f.axis + ': the first-pass auditor said — ' + f.reason).join('\n');
  const contract = wu.task_type === 'chatbot_personalized_response'
    ? 'chatbot_personalized_response: GOLD should weave in the held_out_preference to be more helpful; FOIL should be a plausible generic answer that OMITS it. naturalness = query is natural & plausible for this persona and does not leak the intended answer. example = GOLD really uses the held-out personal info (present in the persona, not fabricated/label-drifted) AND it fits/helps (not forced/creepy/contradicted). inferior = FOIL is a fair generic answer that misses the personalization and is clearly worse (NOT broken for unrelated reasons, NOT actually personalized, NOT as-good-or-better than gold).'
    : 'over_personalization_chatbot_text: the query should NOT need personalization; GOLD answers generically with no persona; FOIL over-personalizes by injecting a REAL persona item where it is unwarranted. naturalness = query is natural & plausible. example = query is genuinely generic (any user would get the same answer) AND the gold stays generic. inferior = FOIL really DOES inject a real, non-disliked persona item of THIS user, unwarranted for the query (not absent/word-coincidence, not fabricated, not disliked, not the query\'s own topic).';
  const lens = k === 1
    ? '\n\nYou are the SECOND independent reviewer: take a completely fresh read. Actively look for a legitimate interpretation under which the row is acceptable BEFORE agreeing it is defective.'
    : '';
  return `You are an adversarial second-pass reviewer. A first-pass auditor FLAGGED a benchmark row as defective on specific axes. Your DEFAULT stance is that the row is FINE — only confirm a problem you can INDEPENDENTLY verify as clearly real; otherwise refute it.
Use your file tools to read the JSON work-unit file at:
  ${wu.file}
Find the row whose query_id == "${row.query_id}" (task_type ${wu.task_type}). Read its user_query, example_response (GOLD), inferior_response (FOIL), groundtruth_preference, held_out_preference, and the persona_card.

The auditor flagged these axes:
${axesTxt}

Apply the SAME contract:
${contract}

For each FLAGGED axis decide: real_problem (the auditor is right — a genuine defect) or refuted (the auditor is wrong — the row is actually fine on that axis). Set every NON-flagged axis to na.${lens}

Return {query_id, naturalness_verdict, example_verdict, inferior_verdict, reason} with each verdict in {real_problem, refuted, na}. reason: 1-2 sentences citing the concrete text.`;
}

phase('Judge');
const totalRows = UNITS.reduce((a, u) => a + u.n_rows, 0);
log(`Judging ${UNITS.length} work units (${totalRows} rows)`);

const results = await pipeline(
  UNITS,
  (wu) => agent(judgePrompt(wu), { label: `judge:${wu.work_unit_id}`, phase: 'Judge', schema: JUDGE_SCHEMA, effort: 'high' }),
  (judge, wu) => {
    const rows = (judge && Array.isArray(judge.rows)) ? judge.rows : [];
    const flaggedRows = rows.filter(r => r.naturalness_pass === false || r.example_pass === false || r.inferior_pass === false);
    const base = { work_unit_id: wu.work_unit_id, user_id: wu.user_id, task_type: wu.task_type, file: wu.file, judge };
    if (flaggedRows.length === 0) return { ...base, verifications: [] };
    const thunks = [];
    for (const r of flaggedRows) {
      const flagged = [];
      if (r.naturalness_pass === false) flagged.push({ axis: 'naturalness', reason: r.naturalness_reason });
      if (r.example_pass === false) flagged.push({ axis: 'example', reason: r.example_reason });
      if (r.inferior_pass === false) flagged.push({ axis: 'inferior', reason: r.inferior_reason });
      for (const k of [0, 1]) {
        thunks.push(() => agent(verifyPrompt(wu, r, flagged, k), { label: `verify:${r.query_id}#${k}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' })
          .then(v => ({ query_id: r.query_id, k, verdict: v }))
          .catch(() => null));
      }
    }
    return parallel(thunks).then(votes => ({ ...base, verifications: votes.filter(Boolean) }));
  }
);

return { n_units: UNITS.length, n_rows: totalRows, results };
'''

META = """export const meta = {
  name: 'verify-personalization-queries',
  description: 'Verify chatbot_personalized_response + over_personalization_chatbot_text example/inferior pairs (naturalness / gold / foil) with adversarial verification',
  phases: [
    { title: 'Judge', detail: 'per work-unit: rule 3 axes per row' },
    { title: 'Verify', detail: 'adversarially re-check each flagged axis with 2 independent skeptics' },
  ],
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    manifest = json.load(open(Path(a.workdir) / "manifest.json"))
    js = META + "\nconst UNITS = " + json.dumps(manifest) + ";\n" + JS_BODY
    out = Path(a.workdir) / "verify_pairs_workflow.js"
    out.write_text(js)
    print("wrote", out, f"({len(js)} bytes, {len(manifest)} units embedded)")
    print("Next: run this file via the Workflow tool; save its returned JSON to", Path(a.workdir) / "wf_result.json")


if __name__ == "__main__":
    main()
