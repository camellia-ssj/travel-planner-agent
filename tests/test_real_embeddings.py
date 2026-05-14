import math
import os
import unittest

from travel_agent.rag.config import EmbeddingProviderName, RagConfig
from travel_agent.rag.embeddings import build_embeddings

PROBE_TEXT = "杭州灵隐寺周末交通拥挤，建议使用公共交通。"


def _assert_embedding_vector(test_case: unittest.TestCase, vector: list[float]) -> None:
    test_case.assertTrue(vector)
    test_case.assertTrue(all(math.isfinite(value) for value in vector))


class RealEmbeddingVerificationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TRAVEL_RAG_VERIFY_BGE_M3") == "1",
        "设置 TRAVEL_RAG_VERIFY_BGE_M3=1 以下载并验证 BAAI/bge-m3。",
    )
    def test_bge_m3_sentence_transformers_embedding_smoke(self) -> None:
        embeddings = build_embeddings(
            RagConfig(
                embedding_provider=EmbeddingProviderName.SENTENCE_TRANSFORMERS,
                sentence_transformers_model="BAAI/bge-m3",
            )
        )

        doc_vector = embeddings.embed_documents([PROBE_TEXT])[0]
        query_vector = embeddings.embed_query(PROBE_TEXT)

        _assert_embedding_vector(self, doc_vector)
        _assert_embedding_vector(self, query_vector)
        self.assertEqual(len(doc_vector), len(query_vector))

    @unittest.skipUnless(
        os.getenv("TRAVEL_RAG_VERIFY_QWEN") == "1" and os.getenv("DASHSCOPE_API_KEY"),
        "设置 TRAVEL_RAG_VERIFY_QWEN=1 和 DASHSCOPE_API_KEY 以验证 text-embedding-v4。",
    )
    def test_qwen_text_embedding_v4_smoke(self) -> None:
        embeddings = build_embeddings(RagConfig(embedding_provider=EmbeddingProviderName.QWEN))

        vector = embeddings.embed_query(PROBE_TEXT)

        _assert_embedding_vector(self, vector)

    @unittest.skipUnless(
        os.getenv("TRAVEL_RAG_VERIFY_OPENAI") == "1" and os.getenv("OPENAI_API_KEY"),
        "设置 TRAVEL_RAG_VERIFY_OPENAI=1 和 OPENAI_API_KEY 以验证 OpenAI embeddings。",
    )
    def test_openai_embedding_smoke(self) -> None:
        embeddings = build_embeddings(RagConfig(embedding_provider=EmbeddingProviderName.OPENAI))

        vector = embeddings.embed_query(PROBE_TEXT)

        _assert_embedding_vector(self, vector)


if __name__ == "__main__":
    unittest.main()
