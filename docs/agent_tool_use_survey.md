# Agent Tool-Use SOTA Survey & PersonaMem-v3 Comparison

*Research conducted April 30, 2026. Survey window: April 2025 – April 2026.*

## TL;DR

- **PersonaMem-v3 has two halves with different tool-use designs**, and the "is our approach SOTA?" question has different answers for each.
- **Data-preparation pipeline (NL prompt → parse-JSON → `Write`): KEEP.** Prompt-only + parse-JSON is the de-facto norm for persona-inference / data-generation pipelines in 2025–2026 papers (Polypersona, German GSS Personas, PersonaMem-v2, OpenCharacter all do the same). The single-Claude-Code-subagent-per-user pattern matches Anthropic's own multi-agent research system, which spawns subagents with NL objectives and reports +90.2% over single-agent on internal evals [[1]](#refs). Optional refinement: lift the existing JSON schemas from `prompts.py` into Structured Outputs (`response_format`) to drop PARSE's reported 11.97% invalid-JSON rate without changing the architecture.
- **Evaluation harness (FastMCP servers + 4-mode comparison + write-overlay): KEEP, possibly EXTEND with a `code_action` mode.** The `mcp_agent` / `agent_tools` / `agent_longctx` / `llm_longctx` matrix over a τ-bench-style write overlay is at or beyond SOTA. AppWorld added an MCP toggle to its existing code-as-action API in 2025 — exactly the same dual-mode pattern PersonaMem-v3 uses [[8]](#refs). MCP-Universe (GPT-5-High = 44%) and MCP-Bench have validated MCP as a benchmark target [[9]](#refs). One concrete gap: no `code_action` mode mirroring Anthropic's Programmatic Tool Calling (Nov 2025) or the OpenHands + CodeAct pattern that hits 68.4% on SWE-bench Verified [[10]](#refs).
- **"MCP vs function calling" is a false dichotomy.** Function calling is the LLM's intent format at the decoder (BFCL v4 standard); MCP is the cross-vendor transport for shared tool servers. All four hyperscalers (Anthropic, OpenAI, Google, Microsoft) ship MCP by Q1 2026, and FastMCP's `@mcp.tool` decorators auto-generate the function-calling JSON schemas Claude actually consumes. PersonaMem-v3's `mcp_agent` mode uses **both** — that's the SOTA stack, not a choice between them.

---

## 1. SOTA Agent Benchmarks (Apr 2025 – Apr 2026)

The agent-benchmark landscape stabilized around two protocol patterns: **native function calling** for tool invocation, and **MCP** for shared cross-vendor tool ecosystems. Coding benchmarks saturated; harder long-horizon and multi-turn benchmarks took their place. Multi-turn / dual-control / async has fully replaced single-turn.

| Benchmark | Domain | Tool protocol | Top model 2025–26 (score) | Citation |
|-----------|--------|---------------|---------------------------|----------|
| SWE-bench Verified | Coding (GitHub issues) | Containerized harness; ReAct-style + bash tool (function calling) | Claude Opus 4.7 — 87.6% (Apr 2026) | [[2]](#refs) |
| SWE-bench Pro | Long-horizon coding (1865 multi-file tasks) | Docker harness, multi-file patches | Claude Opus 4.7 — 64.3% public | [arXiv:2509.16941](https://arxiv.org/abs/2509.16941) |
| GAIA / Gaia2 | General assistant + web/tool use | Native function calling (Inspect AI / smolagents); Gaia2 adds async via Meta ARE | Gaia2: GPT-5-high — 42% pass@1 | [arXiv:2509.17158](https://arxiv.org/abs/2509.17158) |
| τ-bench / τ²-bench | Customer support (retail/airline/telecom) | OpenAI-style function-calling JSON tools; τ² adds dual-control user tools (Dec-POMDP) | τ²-Telecom: LongCat-Flash-Thinking — 99.3% | [arXiv:2506.07982](https://arxiv.org/abs/2506.07982) |
| BFCL v4 (Berkeley) | Function calling + agentic | Native function calling (AST + executable checks); v4 adds web-search + memory + multi-turn | GPT-5 / Claude Opus class lead | [[3]](#refs) |
| OSWorld / OSWorld-Verified | OS/desktop GUI | PyAutoGUI mouse/keyboard + screenshot/a11y observation | Holo3-35B-A3B — 82.6% | [[4]](#refs) |
| **AppWorld** | API/coding agent over 9 apps | **Code-as-action Python REPL + new MCP server/client toggle** | RL'd Qwen2.5-32B (LOOP) ~71% TGC | [arXiv:2407.18901](https://arxiv.org/abs/2407.18901) |
| MLE-bench / RE-Bench | ML engineering / R&D | Code-as-action (sandboxed) | METR: AI 4× human at 2 h budget | [arXiv:2410.07095](https://arxiv.org/abs/2410.07095), [arXiv:2411.15114](https://arxiv.org/abs/2411.15114) |
| TheAgentCompany | Simulated software company | Browser + code + chat tools (OpenHands harness) | Claude 3.5 Sonnet — 24% (NeurIPS 2025) | [arXiv:2412.14161](https://arxiv.org/abs/2412.14161) |
| AgentBench v3 | 8 environments | Refactored to **function-calling + AgentRL** (Oct 2025) | OSS LLM ranking | [arXiv:2308.03688](https://arxiv.org/abs/2308.03688) |
| **MCP-Universe / MCP-Bench / OSWorld-MCP** | **Real MCP servers** | **Native MCP** (multi-server tool catalog) | MCP-Universe: GPT-5-High — 44.16% | [arXiv:2508.14704](https://arxiv.org/abs/2508.14704) |
| HAL (Princeton) | Meta-leaderboard, 11 benchmarks | Standardized HAL harness, cost-aware | Aggregates SWE-bench, GAIA, Cybench, etc. | [arXiv:2510.11977](https://arxiv.org/abs/2510.11977) |

**Most relevant analogs to PersonaMem-v3:**
- **τ²-bench** (Sierra Research, Jun 2025): customer-support agent, multi-turn, dual-control conversational, function-calling — closest parallel to PersonaMem-v3's `mcp_agent` mode + the agentic T6–T19 task surface.
- **AppWorld** (ACL'24, with 2025 MCP add-on): **identical dual-mode design** — code-as-action *and* MCP server/client over the same task suite. PersonaMem-v3's `mcp_agent` / `agent_tools` / `llm_longctx` is the same idea with a different second mode (filesystem instead of code).
- **MCP-Universe / MCP-Bench / OSWorld-MCP**: head-to-head MCP-server evaluations. PersonaMem-v3's `mcp_agent` mode is directly comparable.

---

## 2. Tool-Use Protocols & Adoption

### 2.1 MCP (Model Context Protocol)

Launched by Anthropic Nov 2024. By Q1 2026 it has become the cross-vendor transport for shared tool ecosystems.

- OpenAI added MCP support to the **Responses API on May 21, 2025** [[5]](#refs); OpenAI joined the MCP steering committee.
- Google rolled out MCP support across **Maps, BigQuery, GCE, GKE** in 2025–early 2026 [[6]](#refs).
- Microsoft built MCP into **VS Code and Copilot**.
- **Donated to the Linux Foundation, Dec 2025**.
- ~17K public MCP servers indexed Q1 2026; ~97M monthly SDK downloads (March 2026 figures).
- **Critical RCE design flaw** disclosed Apr 2026 affecting 7K+ public servers and 150M+ downloads [[7]](#refs); MCP security is now the #1 2026 roadmap focus.

### 2.2 Native function calling

Dominant baseline. All major model SDKs (OpenAI `tools=`, Anthropic `tools=`, Gemini function calling) ship it. **BFCL v4** [[3]](#refs) (last updated Apr 12, 2026) remains the canonical tool-use leaderboard, evaluating both native function-calling (`FC`) and prompt-based (`Prompt`) approaches — V4's headline addition is "holistic agentic evaluation" with sub-categories for Agentic, Multi-Turn, Live, Non-Live, and Hallucination. **Structured Outputs** (constrained-decoding JSON) replaced the older "JSON mode" mid-2025 — JSON mode is "largely obsolete" per the 2026 production guide.

### 2.3 ReAct & descendants

**Pure ReAct is rare in production.** It has been absorbed as one option among many in agent frameworks — LangGraph's `create_agent` is the "ReAct" pattern, and Reflexion / LATS / Tree-of-Thoughts are now the more common loop choices. The Sep 2025 *Landscape of Agentic RL for LLMs* survey (500+ papers reviewed) describes the trajectory as "from ReAct prompting → tool-integrated reasoning trained via RL" — i.e., ReAct is the conceptual baseline, not the runtime [[11]](#refs).

### 2.4 Code-as-action

Rising in coding-agent territory. **CodeAct** (arXiv:2402.01030, ICML 2024) reported +20% over JSON tools on multi-step tasks; **OpenHands + CodeAct v3** hits **68.4% on SWE-bench Verified** with Claude Opus 4.6 — matching proprietary scaffolds [[10]](#refs). **Anthropic's Programmatic Tool Calling** (Nov 24, 2025) lets the agent orchestrate tools via Python in a sandbox rather than emit one tool-call per turn. **OpenAI's Agents SDK** added sandboxing + a code-mode + subagents in Apr 2026.

### 2.5 Multi-agent / NL-prompt subagent spawning

**Recognized SOTA pattern.** Anthropic's *How we built our multi-agent research system* (Jun 2025) explicitly describes the lead agent decomposing queries into subtasks and spawning subagents with **natural-language objectives** — *not* structured function-call schemas. The blog reports a multi-agent system (Opus-4 lead + Sonnet-4 subagents) **outperformed single-agent Opus-4 by 90.2% on internal research evals** (verbatim, verified) [[1]](#refs). Multi-agent workflows grew **+327% Jun–Oct 2025** in the Microsoft developer ecosystem. Cursor (late 2025), VS Code Copilot agent mode (Feb 2026), Claude Code SDK, Devin, Replit Agent 3, and Cline have all productized conversational subagent invocation.

The structure is on the *return*, not the spawn. The subagent is told what to do in natural language; what it produces is constrained (a JSON blob, a file write, a tool-call sequence). PersonaMem-v3's data-prep mode is exactly this pattern.

### Protocol comparison matrix

| Protocol | Pushed by | Adoption signal | Strengths | Weaknesses | Momentum (2026) |
|----------|-----------|-----------------|-----------|------------|-----------------|
| MCP | Anthropic; OpenAI/Google/Microsoft/AWS joined 2025 | 17K+ servers; LF donation; all hyperscalers ship | Decoupled, reusable, cross-vendor ("USB-C for AI") | Auth weak by default; RCE flaws; ~55K tokens for 58 tools | **Rising fast** |
| Native function calling | OpenAI, Anthropic, Google | All SDKs; BFCL v4 standard | Tight integration, low latency, parallel calls, schema-validated | App-coupled; doesn't scale across teams | **Stable / dominant baseline** |
| ReAct & descendants | Yao 2022; absorbed by frameworks | LangGraph default loop | Simple, debuggable | Token-inefficient | **Absorbed / stable** |
| Code-as-action | Wang 2024; Anthropic, OpenAI | OpenHands+CodeAct v3 = 68.4% SWE-bench | +20% on multi-step; expressive | Sandboxing/security cost | **Rising in coding agents** |
| Multi-agent / NL spawn | Anthropic, AutoGen→MAF, LangGraph, CrewAI | +327% Jun–Oct 2025; +90.2% over single-agent | Parallelism, context isolation | ~15× token cost, coordination failures | **Rising** |

---

## 3. Patterns in Recent Agent Algorithm Papers (2025–2026)

Twelve representative agent systems / papers from the last 12 months:

1. **Claude Code (Anthropic, 2025)** — function calling + MCP + Skills (progressive-disclosure markdown bundles). [[12]](#refs)
2. **OpenAI Operator / o3-Operator (Jan / May 2025)** — vision-grounded GUI actions + native function calls.
3. **Anthropic Computer Use / Advanced Tool Use (Nov 24, 2025)** — Tool Search, Programmatic Tool Calling, Tool Use Examples. [[10]](#refs)
4. **Cursor Composer 2 (Oct 2025)** — MoE coding model trained with agentic RL; native function calling + 1800+ MCP servers.
5. **Devin 2.0 (Cognition, 2025)** — shell + editor + browser inside sandboxed VM; multi-agent dispatch.
6. **Replit Agent 3 (Sep 2025)** — long-horizon agent that spawns subagents.
7. **Google Jules (Aug 2025 GA)** — async Gemini-2.5-Pro agent over a cloud VM.
8. **GitHub Copilot Agent Mode (Feb 2025; cloud-agent GA Sep 2025)** — built-in tool calls + MCP servers.
9. **Cline (open source, 2025)** — fastest-growing GitHub AI project; function calling + heavy MCP integration.
10. ***Landscape of Agentic RL for LLMs* (Sep 2025)** — 500+ paper survey; "tool-integrated reasoning is no longer niche but baseline." [[11]](#refs)
11. ***Beyond Pipelines: Model-Native Agentic AI* (Oct 2025)** — argues field moves "from JSON-style function calling to code-as-action, standardized connector layers such as MCP."
12. ***Natural Language Tools* (Oct 2025)** — contrarian; replacing JSON tool-calling with YES/NO NL selection gave +18.4 pp tool-accuracy gain across 10 models, 6,400 trials. [[13]](#refs)

### Patterns observed (industry agents, 9 sampled)

- **9/9** support native JSON-schema function calling as primary tool surface.
- **7/9** ship first-class MCP support.
- **3/9** explicitly use code-as-action / programmatic tool calling.
- **5/9** are multi-agent / subagent-spawning.
- **0/9** rely primarily on prompt-only + parse-JSON for control flow.

### But for **data-processing pipelines**, the picture inverts

Persona-generation / structured-extraction papers from Nov–Dec 2025 (Polypersona [[14]](#refs), German GSS Personas, the 83-prompt analysis, OpenCharacter, **PersonaMem-v2** [[15]](#refs)) **all use prompt-and-parse-JSON**. The 83-prompt analysis found "more than half of prompts require persona output in structured format such as JSON." OpenAI's own Structured Outputs guidance and LlamaIndex's docs explicitly recommend non-tool structured outputs for **extraction/classification**, reserving function calling for **action triggering**.

PARSE (arXiv:2510.08623, Oct 2025) reports GPT-4 has an **11.97% invalid-JSON rate** on complex extraction — a known but accepted cost of the prompt-and-parse approach. The mitigation in 2026 is **constrained-decoding Structured Outputs**, not function calling.

---

## 4. PersonaMem-v3's Approach

PersonaMem-v3 has two distinct halves in its codebase, with different tool-use designs.

### 4.1 Data preparation — `data_preparation/`

- **Subagent mode** (default, per `skill.md`): Claude Code spawns one subagent per user, prompted with a verbatim instruction set drawn from `data_preparation/prompts.py`. The subagent reasons in natural language through 22 pipeline steps and produces 5 JSON files (`profile.json`, `instagram.json`, `facebook.json`, `threads.json`, `chatbot.json`) under `backend/{user_id}/` via the built-in `Write` tool.
- **API mode** (`scripts/run_persona_pipeline.py`): a Python `ThreadPoolExecutor` loop; `PersonaAgent.run_pipeline()` issues 22 sequential LLM calls per user via `query_llm.py` (multi-provider: OpenAI / Azure / Anthropic / Gemini). Each prompt instructs the model to emit JSON; `extract_json_from_response()` parses it.
- **Both modes use the exact same prompts**, so LLM choice is the only variable.
- **No** `tools=[...]`, **no** MCP, **no** ReAct loop, **no** function-calling schemas anywhere in the data-prep path. Verified by repo-wide grep.

### 4.2 Evaluation harness — `evaluation/`

- **5 mock MCP servers** built on **FastMCP**: `instagram_mcp_server`, `facebook_mcp_server`, `threads_mcp_server`, `chatbot_mcp_server`, plus a `google_search_mcp_server`. Tools defined via `@mcp.tool()` decorators (the schemas are auto-generated and consumed by Claude as native function calls). Per-app tool surface: `get_feed`, `get_post`, `search`, `list_dms`, `get_dm_thread`, `create_post`, `react`, `comment`, `send_dm`. Chatbot adds `get_history`, `search_history`, `send_post_to_app` (cross-app dispatch), `summarize_inbox`.
- **4 evaluation modes** dispatched from `evaluation/inference_utils.py::dispatch_agent_run`:

| Mode | Mechanism | Tools | Isolates |
|------|-----------|-------|----------|
| `mcp_agent` | `claude -p --mcp-config` with FastMCP servers | MCP `mcp__<app>__*` tools | Structured-API agentic behavior |
| `agent_tools` | `claude -p` with path-scoped `Read` over a time-masked filesystem snapshot | Filesystem `Read` only | Filesystem-agent retrieval |
| `agent_longctx` | `claude -p` with no tools; full pre-`T_test` history pre-loaded | none | Claude Code framework effect, no retrieval |
| `llm_longctx` | Single `QueryLLM.query_llm` call (Azure / OpenAI / Anthropic / Gemini) | none | Pure long-context baseline, no agent framework |

- **τ-bench-style write overlay** (`evaluation/mcp_overlay.py`): MCP write tools (`create_post`, `react`, `send_dm`, etc.) append to `writes.jsonl`; subsequent reads in the same run union the frozen backend with the overlay so the agent sees its own posts in its own feed (matching real-app semantics). The grader reads the same `writes.jsonl` for `final_state_diff` rubric scoring on agentic tasks T6–T19.
- **14 agentic tasks** (T6–T19): community digest, moment recommendation, DM digest, cross-app repost, auto-reply on behalf, vague refind, agent-composed post, chatbot→app dispatch, draft-audit privacy, saved-collection curation, group-DM summary, wrong-recipient probe, proactive daily briefing, trending alert. Each has rubric content-checks + tool-call regex rules + `final_state_diff` checks.
- **Universal personalization rubric** (`evaluation/personalization_rubric.py`): 7 dimensions — `preference_alignment`, `avoid_leak`, `privacy_leak`, `over_personalization`, `stale_preference_use`, `relationship_aware`, `voice_match`. Hard-rule failures (avoid_leak, privacy_leak, stale_preference_use) zero the task score regardless of other metrics — the same philosophy as τ²-bench's "technically-correct outputs that violate user constraints are not good outputs."

---

## 5. Comparison

| Dimension | 2025–2026 SOTA norm | PersonaMem-v3 (data prep) | PersonaMem-v3 (eval) | Gap? |
|-----------|---------------------|---------------------------|----------------------|------|
| Tool-use protocol | Function calling + MCP | Prompt + parse-JSON + `Write` | FastMCP `@mcp.tool` (`mcp_agent`) + Claude Code `Read` (`agent_tools`) | None |
| Multi-agent spawning | NL task descriptions to subagents | One Claude Code subagent per user, NL prompt | One Claude Code subagent per query | Aligned |
| Output structure | Structured Outputs (constrained decoding) replacing JSON mode | Free-form JSON + parse with retry | n/a | **Could adopt SO** |
| Evaluation comparability | Head-to-head mode/protocol comparison (AppWorld, MCP-Universe, OSWorld-MCP) | n/a | 4-mode comparison (`mcp` / `fs` / `longctx` / `llm`) | **Beyond SOTA** |
| Write/state-diff scoring | τ-bench-style final-state diff (τ², AppWorld, OSWorld) | n/a | `writes.jsonl` overlay + `final_state_diff` for T6–T19 | Aligned |
| Code-as-action | OpenHands + CodeAct, Programmatic Tool Calling | n/a | Not implemented | **Potential 5th mode** |
| Structured tool schemas | Native FC (BFCL v4) | Not used | Auto-generated by FastMCP `@mcp.tool` | Aligned via MCP |
| Security posture | MCP RCE flaws disclosed Apr 2026 | n/a | stdio-only, env-scoped, `--strict-mcp-config`, path-scoped sandbox | Aligned |

---

## 6. Verdict & Recommendation

### 6.1 Data-preparation pipeline: KEEP AS IS

The prompt-only + parse-JSON pattern with `Write` for output is the *de facto* norm for persona / data-generation pipelines in 2025–2026 papers. The single-Claude-Code-subagent-per-user pattern matches Anthropic's own multi-agent research-system architecture (Jun 2025), which spawns subagents with NL objectives — verified verbatim quote: *"Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries."*

The contrarian *Natural Language Tools* paper (arXiv:2510.14453) provides additional cover: forcing JSON tool-calling drops GSM8K reasoning by 27.3 pp; replacing it with NL selection gave +18.4 pp on tool accuracy across 10 models. For cognitive tasks where the model must reason then emit a structured artifact, the field has not converged on function calling — it has stayed with prompt + parse.

**Optional refinement**: switch the API-mode path from free-form JSON parsing-with-retry to **OpenAI/Anthropic Structured Outputs** (constrained decoding). The schemas are already specified in `prompts.py`; lifting them into `response_format` parameters in `query_llm.py` is a contained change. This drops PARSE's reported 11.97% invalid-JSON rate to effectively zero without altering the architecture. Subagent mode keeps using `Write` (no API surface to change there).

### 6.2 Evaluation harness: KEEP, EXTEND WITH ONE MODE

The 4-mode dispatch over a τ-bench-style write overlay is at or beyond the current SOTA bar. **AppWorld** added an MCP server/client toggle in 2025 to its existing code-as-action API — exactly the same dual-mode comparison PersonaMem-v3 implements. **MCP-Universe** (GPT-5-High = 44%), **MCP-Bench**, and **OSWorld-MCP** are head-to-head MCP-server evaluations directly comparable to `mcp_agent`. The personalization rubric's hard-constraint failures (`avoid_leak`, `privacy_leak`, `stale_preference_use`) match τ²-bench's "technically correct outputs that violate user constraints are not good outputs."

**Concrete extension**: add a `code_action` mode that gives the subagent a sandboxed Python REPL over the existing `BackendQuery` API. This mirrors Anthropic's Programmatic Tool Calling (Nov 2025) and the OpenHands + CodeAct pattern that hits 68.4% on SWE-bench Verified. Implementation cost is small: one new branch in `dispatch_agent_run`, one new subprocess wrapper analogous to `claude_subagent.run_subagent` but with a Python sandbox instead of `claude -p`. The 5-mode head-to-head then covers the four protocol families that matter in 2026: structured-tool MCP, filesystem retrieval, code-as-action, long-context with-framework, and long-context raw-LLM.

**MCP security note**: a critical RCE design flaw was disclosed in MCP servers in Apr 2026 [[7]](#refs) affecting 7K+ public servers. PersonaMem-v3's mock servers run **stdio-only over `python -m`**, scoped via env vars (`PM3_USER_ID`, `PM3_T_TEST`, `PM3_OVERLAY_PATH`), with `--strict-mcp-config` and `--setting-sources ""` to prevent permission inheritance — i.e., the 2026-hardened config. Worth a short threat-model note in `EVAL.md`.

### 6.3 Stop framing it as "MCP vs function calling"

These are **complementary layers**, not alternatives. Function calling is the LLM's intent format at the decoder (BFCL v4 standardized eval); MCP is the cross-vendor *transport* for shared tool servers. PersonaMem-v3's `mcp_agent` mode uses both: FastMCP exposes tools whose function-calling JSON schemas are auto-generated from `@mcp.tool` decorators, and Claude consumes them via native function calling. That's the SOTA stack — not a binary choice.

### 6.4 What PersonaMem-v3 brings that most SOTA work doesn't

Two design choices worth highlighting in the paper's related-work section:

1. **The protocol-isolation experimental design.** Most 2025–2026 benchmarks pick one tool-use surface (MCP-Universe = MCP only; AppWorld = code OR MCP; τ²-bench = function calling). PersonaMem-v3 runs the *same* test instances through 4 surfaces and asks "does structured MCP access beat raw filesystem? does Claude Code's framework beat raw LLM?". This is closer to the *Beyond Pipelines* survey's call for "comparing model-native vs pipelined agent designs head-to-head" than most benchmark papers manage.
2. **Personalization-aware hard-constraint scoring.** Across the 14 agentic tasks (T6–T19), the rubric treats `avoid_leak` / `privacy_leak` / `stale_preference_use` as binary hard failures that zero the score. That's a personalization analog of τ²-bench's policy violations and is rarer in the literature than it should be — an opportunity to claim contribution.

---

## References <a id="refs"></a>

1. Anthropic. *How we built our multi-agent research system.* Jun 2025. <https://www.anthropic.com/engineering/multi-agent-research-system> — verified verbatim: *"a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval."*
2. SWE-bench leaderboards (Verified / Multimodal / Pro). <https://www.swebench.com/>
3. Patil et al. *Berkeley Function Calling Leaderboard v4.* Berkeley, ICML 2025; leaderboard updated Apr 12, 2026. <https://gorilla.cs.berkeley.edu/leaderboard.html>
4. Xie et al. *OSWorld* / OSWorld-Verified. arXiv:2404.07972 + xlang.ai blog Jul 2025.
5. OpenAI. *Introducing support for remote MCP servers in the Responses API.* May 21, 2025. <https://community.openai.com/t/introducing-support-for-remote-mcp-servers-image-generation-code-interpreter-and-more-in-the-responses-api/1266973>
6. Google Cloud. *Announcing official MCP support for Google services.* 2025–2026. <https://cloud.google.com/blog/products/ai-machine-learning/announcing-official-mcp-support-for-google-services>
7. The Register. *MCP design flaw allows RCE in 7K+ public servers.* April 16, 2026. <https://www.theregister.com/2026/04/16/anthropic_mcp_design_flaw/>
8. Trivedi et al. *AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents.* ACL 2024; MCP server/client toggle added 2025. arXiv:2407.18901. <https://arxiv.org/abs/2407.18901>
9. *MCP-Universe* (arXiv:2508.14704), *MCP-Bench*, *OSWorld-MCP* — first wave of MCP-native benchmarks.
10. Wang et al. *Executable Code Actions Elicit Better LLM Agents (CodeAct).* arXiv:2402.01030, ICML 2024. Anthropic. *Advanced Tool Use* (Tool Search, Programmatic Tool Calling, Tool Use Examples). Nov 24, 2025. <https://www.anthropic.com/engineering/advanced-tool-use>. SWE-bench Verified leaderboard: OpenHands + CodeAct v3 = 68.4% on Claude Opus 4.6.
11. *The Landscape of Agentic Reinforcement Learning for LLMs.* Sep 2025. arXiv:2509.02547. <https://arxiv.org/abs/2509.02547>
12. *Inside Claude Code: architecture behind tools, memory, hooks, and MCP.* 2025. <https://www.penligent.ai/hackinglabs/inside-claude-code-the-architecture-behind-tools-memory-hooks-and-mcp/>; Anthropic. *Skills explained.* <https://claude.com/blog/skills-explained>
13. *Natural Language Tools.* Oct 2025. arXiv:2510.14453. <https://arxiv.org/html/2510.14453v1> — +18.4 pp tool-accuracy across 10 models, 6,400 trials, when JSON tool-calling is replaced with NL YES/NO selection.
14. *Polypersona* (arXiv:2512.14562); *German GSS Personas* (arXiv:2511.21722); *83-prompt analysis* (arXiv:2508.13047); *OpenCharacter* (arXiv:2501.15427).
15. PersonaMem-v2. arXiv:2512.06688. <https://arxiv.org/abs/2512.06688>
16. *Beyond Pipelines: Model-Native Agentic AI.* Oct 2025. arXiv:2510.16720.
17. *τ-bench* (arXiv:2406.12045) and *τ²-bench* (arXiv:2506.07982). Sierra Research.
18. *PARSE: Prompt-And-paRse Structured Extraction.* Oct 2025. arXiv:2510.08623 — 11.97% invalid-JSON rate on complex extraction.
19. *MCP year-in-review.* Pento Engineering, 2026. <https://www.pento.ai/blog/a-year-of-mcp-2025-review>
20. *Structured Outputs vs JSON mode vs Function Calling: 2026 production guide.* <https://www.buildmvpfast.com/blog/structured-output-llm-json-mode-function-calling-production-guide-2026>
21. *Quantifying Structured Output Impact via Causal Inference.* Sep 2025. arXiv:2509.21791 — pushes back on the "JSON degrades reasoning" claim.

---

*Survey conducted via three parallel research subagents (general-purpose, with WebSearch + WebFetch). Two key citations verified via direct WebFetch (Anthropic multi-agent system blog quoted verbatim; AppWorld arXiv abstract confirmed but MCP-toggle claim sourced from secondary leaderboard pages, not the original ACL'24 paper). Code-state claims about PersonaMem-v3 verified by repo-wide grep — `evaluation/mcp_servers/` is the only location where `@mcp.tool` / `FastMCP` appear (27 references across 3 files); `data_preparation/` has zero MCP/function-calling references.*
