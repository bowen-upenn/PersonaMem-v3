"""Real `mem0ai` memory baseline, configured fully on Azure OpenAI.

This is the SECOND memory baseline (mode `mem0`), distinct from the in-house
persona/preference text memory (`llm_memory`, in `memory_builder.py`). It wraps
the *real* `mem0ai` library — we do NOT reimplement it:

  - LLM provider  : `azure_openai`, deployment `gpt-5.5` (fact extraction +
                    ADD/UPDATE/DELETE/NOOP decisions — mem0 owns those prompts).
  - embedder      : `azure_openai`, deployment `text-embedding-3-large` (3072-d).
  - vector store  : local on-disk `qdrant` namespaced per user under the run dir.

Two correctness properties we add on top of the library:

  1. gpt-5.5 is a flagship reasoning deployment that rejects non-default
     `temperature` and `max_tokens` (it wants `max_completion_tokens` only) —
     exactly the param set mem0 already strips for its "reasoning model" path.
     mem0 2.0.4 does NOT classify `gpt-5.5` as reasoning, so we extend its
     detector (`_register_gpt55_as_reasoning`) to send only `model`+`messages`.

  2. TIME-MASKING (firewall): every event is `.add()`-ed in ascending time with
     `metadata={"ts": event_ts}`, and after each add we reconcile the `ts` of
     EVERY memory the add touched (ADD or UPDATE) to that chunk's timestamp.
     Because adds are monotonic, a memory's `ts` is then the latest event that
     contributed to it, so a per-query `search(..., filters={"ts": {"lt": T}})`
     can never surface a fact informed by an event at/after the test moment.
     (Verified empirically: mem0 strongly prefers emitting a new dated ADD over
     mutating an old fact, and the `lt` filter is honored by the qdrant store.)

Retrieval is per-query top-k semantic search (mem0's actual value), rendered to
a compact fact list capped at the same token budget as `llm_memory`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from evaluation.backend_query import BackendQuery
from evaluation.inference_utils import count_tokens
from evaluation.memory_builder import (
    _render_event_line,
    _should_close_chunk,
    build_global_stream,
    default_memory_config,
)

# Deliberately GENERAL fact-extraction guidance — NOT tuned to any evaluation
# dimension (no dislikes/restraint/over-personalization special-casing). mem0
# owns its own ADD/UPDATE/DELETE/NOOP + retrieval prompts; this only nudges
# extraction toward durable, personalization-relevant, grounded facts.
MEM0_CUSTOM_INSTRUCTIONS = (
    "Extract concise, durable facts about the user that would help personalize "
    "future interactions — their interests and preferences, traits, habits, "
    "relationships, places, and current goals — including preferences implied by "
    "recurring behavior, not only those stated outright. Ground each fact in the "
    "observed activity; infer from patterns but do not fabricate or invent "
    "demographics."
)

EMBED_DIMS = 3072  # text-embedding-3-large

_PATCHED = False


def _register_gpt55_as_reasoning() -> None:
    """Make mem0 treat `gpt-5.5*` like a reasoning model (send only
    model+messages, dropping temperature/max_tokens which the deployment 400s on).
    Idempotent; preserves mem0's behavior for every other model."""
    global _PATCHED
    if _PATCHED:
        return
    from mem0.llms.base import LLMBase

    _orig = LLMBase._is_reasoning_model

    def _patched(self, model):  # noqa: ANN001
        base = (model or "").lower().rsplit("/", 1)[-1]
        if base.startswith("gpt-5.5"):
            return True
        return _orig(self, model)

    LLMBase._is_reasoning_model = _patched
    _PATCHED = True


def _azure_env() -> dict:
    """Resolve the Azure creds + deployments from the environment (.env)."""
    embed = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING")
        or "text-embedding-3-large"
    )
    return {
        "key": os.getenv("AZURE_OPENAI_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "llm_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or "gpt-5.5",
        "embed_deployment": embed,
    }


def _mem0_config(store_dir: Path, collection: str, llm_deployment: str) -> dict:
    env = _azure_env()
    missing = [k for k in ("key", "endpoint", "api_version") if not env[k]]
    if missing:
        raise RuntimeError(f"mem0 Azure config missing env: {missing}")
    azure = lambda dep: {  # noqa: E731
        "api_key": env["key"],
        "azure_deployment": dep,
        "azure_endpoint": env["endpoint"],
        "api_version": env["api_version"],
    }
    return {
        "version": "v1.1",
        "llm": {
            "provider": "azure_openai",
            "config": {"model": llm_deployment, "azure_kwargs": azure(llm_deployment)},
        },
        "embedder": {
            "provider": "azure_openai",
            "config": {
                "model": env["embed_deployment"],
                "embedding_dims": EMBED_DIMS,
                "azure_kwargs": azure(env["embed_deployment"]),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "path": str(store_dir),
                "on_disk": True,
                "embedding_model_dims": EMBED_DIMS,
            },
        },
        "custom_fact_extraction_prompt": None,  # use mem0 default + custom_instructions
        "custom_instructions": MEM0_CUSTOM_INSTRUCTIONS,
    }


class Mem0Backend:
    """Per-user real-mem0 store with monotonic, time-masked ingest + per-query
    top-k retrieval. Build once up to the max test moment, then retrieve at any
    earlier moment via the `ts < T_test` filter (firewall-safe)."""

    def __init__(self, user_id: str, store_dir, *, llm_deployment: str | None = None,
                 token_cap: int = 2048, fresh: bool = True):
        _register_gpt55_as_reasoning()
        from mem0 import Memory  # imported lazily so non-mem0 modes don't need it

        self.user_id = str(user_id)
        self.token_cap = token_cap
        self.store_dir = Path(store_dir) / "mem0_store"
        if fresh and self.store_dir.exists():
            shutil.rmtree(self.store_dir, ignore_errors=True)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        env = _azure_env()
        cfg = _mem0_config(self.store_dir, f"u{self.user_id}",
                           llm_deployment or env["llm_deployment"])
        self.memory = Memory.from_config(cfg)
        self.build_stats = {"n_events": 0, "n_chunks": 0, "n_memories": 0}

    # -- build -------------------------------------------------------------
    def _reconcile_ts(self, add_result: dict, chunk_ts: int) -> None:
        """Force every memory the add touched to carry `chunk_ts` so a memory's
        timestamp is always the latest event that informed it (no leak)."""
        results = (add_result or {}).get("results", []) if isinstance(add_result, dict) else []
        for r in results:
            mid = r.get("id")
            if not mid:
                continue
            try:
                self.memory.vector_store.update(vector_id=mid, payload={"ts": int(chunk_ts)})
            except Exception:
                pass  # ADD already stamped with chunk_ts; UPDATE drift is rare

    def build(self, bq: BackendQuery, t_max: int, cfg: dict | None = None) -> dict:
        """Ingest every cross-app event with `source_timestamp < t_max`, in
        ascending time, chunked like the `llm_memory` builder (fairness)."""
        cfg = {**default_memory_config(), **(cfg or {})}
        model = cfg.get("builder_model")
        rows = build_global_stream(bq, self.user_id, int(t_max))
        self.build_stats["n_events"] = len(rows)
        buffer: list[dict] = []

        def _flush() -> None:
            if not buffer:
                return
            chunk_ts = max(int(c.get("t") or 0) for c in buffer)
            messages = [{"role": "user", "content": _render_event_line(c)} for c in buffer]
            res = self.memory.add(messages, metadata={"ts": chunk_ts}, infer=True,
                                  **{"user_id": self.user_id})
            self._reconcile_ts(res, chunk_ts)
            self.build_stats["n_chunks"] += 1
            buffer.clear()

        for row in rows:
            buffer.append(row)
            if _should_close_chunk(buffer, cfg, model):
                _flush()
        _flush()
        try:
            allm = self.memory.get_all(filters={"user_id": self.user_id}, top_k=10000)
            items = allm.get("results", allm) if isinstance(allm, dict) else allm
            self.build_stats["n_memories"] = len(items or [])
        except Exception:
            pass
        return self.build_stats

    # -- retrieve ----------------------------------------------------------
    def retrieve(self, query: str | None, t_test: int, *, top_k: int = 30) -> tuple[str, dict]:
        """Per-query top-k facts with `ts < t_test`, rendered to a compact list
        capped at `token_cap`. With no query, fall back to the most recent facts."""
        t = int(t_test)
        flt = {"user_id": self.user_id, "ts": {"lt": t}}
        try:
            if query:
                res = self.memory.search(query, filters=flt, top_k=top_k)
            else:
                res = self.memory.get_all(filters=flt, top_k=top_k)
            items = res.get("results", res) if isinstance(res, dict) else res
        except Exception as exc:  # never crash an eval row on a retrieval hiccup
            return "", {"memory_mode": True, "mem0": True, "error": str(exc),
                        "total_tokens": 0, "per_app": {}}
        items = items or []
        # Most relevant first when searching; else most recent first.
        if not query:
            items.sort(key=lambda it: int((it.get("metadata") or {}).get("ts") or 0), reverse=True)

        lines: list[str] = ["# Retrieved memory about this user (most relevant facts)"]
        used = count_tokens(lines[0])
        n = 0
        for it in items:
            mem = (it.get("memory") or "").strip()
            if not mem:
                continue
            line = f"- {mem}"
            lt = count_tokens(line)
            if used + lt > self.token_cap:
                break
            lines.append(line)
            used += lt
            n += 1
        text = "\n".join(lines) if n else ""
        return text, {
            "memory_mode": True,
            "mem0": True,
            "total_tokens": used if n else 0,
            "n_retrieved": n,
            "per_app": {},
        }
