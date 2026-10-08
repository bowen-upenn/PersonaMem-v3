"""Offline checks for local LLM routing; no model downloads or API calls."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.mem0_backend import _mem0_config
from query_llm import QueryLLM


class LocalQwenTests(unittest.TestCase):
    def test_answer_client_prefers_local_endpoint_over_azure(self):
        env = {
            "PM3_LLM_BASE_URL": "http://127.0.0.1:8002/v1",
            "AZURE_OPENAI_KEY": "unused-cloud-key",
            "AZURE_OPENAI_ENDPOINT": "https://unused.invalid",
        }
        client = QueryLLM.__new__(QueryLLM)
        client.args = {"models": {"llm_model": "Qwen/Qwen2.5-1.5B-Instruct"}}
        with patch.dict(os.environ, env, clear=True), \
                patch("query_llm.OpenAI") as local, \
                patch("query_llm.AzureOpenAI") as azure:
            client._setup_client()
            azure.assert_not_called()
            self.assertEqual(local.call_args.kwargs["base_url"], env["PM3_LLM_BASE_URL"])
            self.assertEqual(client.model, client.args["models"]["llm_model"])

    def test_memory_uses_local_llm_and_matching_embedding_dimensions(self):
        with patch.dict(os.environ, {"PM3_LLM_BASE_URL": "http://localhost:8002/v1"}, clear=True):
            cfg = _mem0_config(Path("/tmp/test-qwen-store"), "u115", "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(cfg["llm"]["config"]["openai_base_url"], "http://localhost:8002/v1")
        self.assertEqual(cfg["embedder"]["provider"], "huggingface")
        self.assertEqual(cfg["embedder"]["config"]["embedding_dims"],
                         cfg["vector_store"]["config"]["embedding_model_dims"])

    def test_local_memory_rejects_openrouter_override(self):
        with patch.dict(os.environ, {"PM3_LLM_BASE_URL": "http://localhost:8002/v1",
                                     "OPENROUTER_API_KEY": "unused"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER"):
                _mem0_config(Path("/tmp/test-qwen-store"), "u115", "Qwen")

    def test_original_azure_config_still_works(self):
        env = {"AZURE_OPENAI_KEY": "unused", "AZURE_OPENAI_ENDPOINT": "https://unused.invalid",
               "AZURE_OPENAI_API_VERSION": "test"}
        with patch.dict(os.environ, env, clear=True):
            cfg = _mem0_config(Path("/tmp/test-azure-store"), "u115", "gpt-5.5")
        self.assertEqual(cfg["llm"]["provider"], "azure_openai")
        self.assertEqual(cfg["vector_store"]["config"]["embedding_model_dims"], 3072)

    def test_custom_local_embedding_keeps_collection_dimensions_in_sync(self):
        env = {"PM3_LLM_BASE_URL": "http://localhost:8002/v1",
               "PM3_EMBED_MODEL": "sentence-transformers/all-mpnet-base-v2",
               "PM3_EMBED_DIMS": "768"}
        with patch.dict(os.environ, env, clear=True):
            cfg = _mem0_config(Path("/tmp/test-store"), "u115", "Qwen")
        self.assertEqual(cfg["embedder"]["config"]["model"], env["PM3_EMBED_MODEL"])
        self.assertEqual(cfg["embedder"]["config"]["embedding_dims"], 768)
        self.assertEqual(cfg["vector_store"]["config"]["embedding_model_dims"], 768)

    def test_custom_azure_embedding_sets_deployment_and_collection_dimensions(self):
        env = {"AZURE_OPENAI_KEY": "unused", "AZURE_OPENAI_ENDPOINT": "https://unused.invalid",
               "AZURE_OPENAI_API_VERSION": "test", "PM3_EMBED_MODEL": "small-deployment",
               "PM3_EMBED_DIMS": "1536"}
        with patch.dict(os.environ, env, clear=True):
            cfg = _mem0_config(Path("/tmp/test-store"), "u115", "gpt-5.5")
        embed = cfg["embedder"]["config"]
        self.assertEqual(embed["model"], "small-deployment")
        self.assertEqual(embed["azure_kwargs"]["azure_deployment"], "small-deployment")
        self.assertEqual(embed["embedding_dims"], 1536)
        self.assertEqual(cfg["vector_store"]["config"]["embedding_model_dims"], 1536)

    def test_legacy_azure_embedding_deployment_aliases_still_work(self):
        for key in ("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED", "AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING"):
            with self.subTest(key=key), patch.dict(os.environ, {
                    "AZURE_OPENAI_KEY": "unused", "AZURE_OPENAI_ENDPOINT": "https://unused.invalid",
                    "AZURE_OPENAI_API_VERSION": "test", key: "existing-deployment"}, clear=True):
                cfg = _mem0_config(Path("/tmp/test-store"), "u115", "gpt-5.5")
            self.assertEqual(cfg["embedder"]["config"]["model"], "existing-deployment")
            self.assertEqual(cfg["vector_store"]["config"]["embedding_model_dims"], 3072)

    def test_custom_model_requires_explicit_dimensions(self):
        with patch.dict(os.environ, {"PM3_LLM_BASE_URL": "http://localhost:8002/v1",
                                     "PM3_EMBED_MODEL": "custom-model"}, clear=True):
            with self.assertRaisesRegex(ValueError, "PM3_EMBED_DIMS"):
                _mem0_config(Path("/tmp/test-store"), "u115", "Qwen")

    def test_embedding_dimensions_reject_invalid_values(self):
        for dims in ("0", "-1", "1.5", "abc", ""):
            with self.subTest(dims=dims), patch.dict(os.environ, {
                    "PM3_LLM_BASE_URL": "http://localhost:8002/v1", "PM3_EMBED_DIMS": dims}, clear=True):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    _mem0_config(Path("/tmp/test-store"), "u115", "Qwen")


if __name__ == "__main__":
    unittest.main()
