import json
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from typer.testing import CliRunner

from travel_agent.rag import (
    TravelRag,
    answer_destination_question,
    get_destination_retriever,
    query_destination_knowledge,
    retrieve_destination_evidence,
    search_destination_knowledge,
)
from travel_agent.rag.cli import app
from travel_agent.rag.config import EmbeddingProviderName, RagConfig, RetrievalMode
from travel_agent.rag.embeddings import LocalHashEmbeddings, build_embeddings
from travel_agent.rag.evaluation import evaluate_quality_gate
from travel_agent.rag.metadata import CHUNK_METADATA_FIELDS, split_markdown_sections
from travel_agent.rag.service import RagService
from travel_agent.rag.vector_store import vector_store_metadatas


def _test_temp_dir() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    root = project_root / "data" / "test_tmp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fresh_case_dir(name: str) -> Path:
    root = _test_temp_dir() / name
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_filter_docs(docs: Path) -> None:
    docs.mkdir()
    (docs / "kyoto.md").write_text(
        "---\n"
        "destination: Kyoto\n"
        "city: Kyoto\n"
        "country: Japan\n"
        "travel_type: family_free_independent\n"
        "season: spring,autumn\n"
        "source_type: destination_guide\n"
        "updated_at: 2026-05-08\n"
        "---\n"
        "# Kyoto\n"
        "Kyoto overview for family travel.\n\n"
        "## 交通\n"
        "Kyoto rail and bus transit advice for families.\n\n"
        "## 预算\n"
        "Kyoto family budget should include temple tickets and transit passes.\n\n"
        "## 天气风险\n"
        "Kyoto autumn rain can make temple paths slippery.\n",
        encoding="utf-8",
    )
    (docs / "osaka.md").write_text(
        "---\n"
        "destination: Osaka\n"
        "city: Osaka\n"
        "country: Japan\n"
        "travel_type: solo_food\n"
        "season: winter\n"
        "source_type: destination_guide\n"
        "updated_at: 2026-05-08\n"
        "---\n"
        "# Osaka\n"
        "Osaka overview for solo food travel.\n\n"
        "## 交通\n"
        "Osaka subway transit is convenient for food districts.\n\n"
        "## 预算\n"
        "Osaka solo food budget can focus on snacks and short subway rides.\n",
        encoding="utf-8",
    )


class _FakeOpenAIEmbeddings:
    call_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        self.__class__.call_kwargs = kwargs


class RagPipelineTest(unittest.TestCase):
    def test_default_retrieval_mode_is_hybrid(self) -> None:
        self.assertEqual(RagConfig().retrieval_mode, RetrievalMode.HYBRID)

    def test_default_qwen_embedding_model_is_text_embedding_v4(self) -> None:
        settings = RagConfig()

        self.assertEqual(settings.qwen_embedding_model, "text-embedding-v4")
        self.assertEqual(settings.qwen_embedding_dimensions, 1024)
        self.assertEqual(settings.qwen_embedding_batch_size, 10)
        self.assertEqual(
            settings.qwen_base_url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_auto_embedding_prefers_qwen_when_dashscope_key_is_present(self) -> None:
        settings = RagConfig(embedding_provider=EmbeddingProviderName.AUTO)

        with (
            patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True),
            patch(
                "travel_agent.rag.embeddings._openai_embeddings_class",
                return_value=_FakeOpenAIEmbeddings,
            ),
        ):
            embeddings = build_embeddings(settings)

        self.assertIsInstance(embeddings, _FakeOpenAIEmbeddings)
        kwargs = _FakeOpenAIEmbeddings.call_kwargs
        self.assertEqual(kwargs["model"], "text-embedding-v4")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["base_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.assertEqual(kwargs["dimensions"], 1024)
        self.assertEqual(kwargs["chunk_size"], 10)
        self.assertFalse(kwargs["tiktoken_enabled"])
        self.assertFalse(kwargs["check_embedding_ctx_length"])

    def test_auto_embedding_falls_back_to_local_without_api_keys(self) -> None:
        settings = RagConfig(embedding_provider=EmbeddingProviderName.AUTO)

        with patch.dict(os.environ, {}, clear=True):
            embeddings = build_embeddings(settings)

        self.assertIsInstance(embeddings, LocalHashEmbeddings)

    def test_rag_quality_gate_fails_below_threshold(self) -> None:
        gate = evaluate_quality_gate(
            metrics={"recall_at_k": 0.8, "mrr_at_k": 1.0},
            thresholds={"recall_at_k": 0.9, "mrr_at_k": 0.9},
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["thresholds"]["recall_at_k"], 0.9)
        self.assertIn("recall_at_k=0.8000 below threshold 0.9000", gate["failures"])

    def test_ingest_markdown_and_query_returns_required_fields(self) -> None:
        root = _fresh_case_dir("ingest_query")
        docs = root / "docs"
        docs.mkdir()
        (docs / "hangzhou.md").write_text(
            "---\ndestination: Hangzhou\n---\n"
            "# Hangzhou\nWest Lake is good for slow family travel. "
            "Lingyin Temple is crowded on weekends.",
            encoding="utf-8",
        )
        (docs / "tokyo.md").write_text(
            "---\ndestination: Tokyo\n---\n"
            "# Tokyo\nTokyo Disney Resort needs a full day for families.",
            encoding="utf-8",
        )

        service = RagService(
            RagConfig(
                persist_dir=root / "chroma",
                embedding_provider=EmbeddingProviderName.LOCAL,
                chunk_size=120,
                chunk_overlap=20,
                default_top_k=2,
            )
        )

        report = service.ingest_markdown(docs)
        response = service.query("family trip to West Lake", top_k=1)

        self.assertEqual(report.scanned_files, 2)
        self.assertEqual(report.loaded_documents, 2)
        self.assertGreaterEqual(report.indexed_chunks, 2)
        self.assertTrue(response.results)
        result = response.results[0]
        self.assertIn("West Lake", result.content)
        self.assertTrue(result.source.endswith("hangzhou.md"))
        self.assertEqual(result.destination, "Hangzhou")
        self.assertIsInstance(result.score, float)

    def test_ingest_documents_supports_markdown_text_and_pdf(self) -> None:
        root = _fresh_case_dir("ingest_multiple_file_types")
        docs = root / "docs"
        docs.mkdir()
        (docs / "hangzhou.md").write_text(
            "---\ndestination: Hangzhou\n---\n# Hangzhou\nWest Lake family travel notes.",
            encoding="utf-8",
        )
        (docs / "kyoto.txt").write_text(
            "Kyoto rail transit advice for family temple visits.",
            encoding="utf-8",
        )
        (docs / "seoul.pdf").write_bytes(b"%PDF-1.4\n")

        class FakePyPDFLoader:
            def __init__(self, path: str) -> None:
                self.path = path

            def load(self) -> list[Document]:
                return [
                    Document(
                        page_content="Seoul palace PDF guide with subway transit advice.",
                        metadata={"source": self.path},
                    )
                ]

        service = RagService(
            RagConfig(
                persist_dir=root / "chroma",
                embedding_provider=EmbeddingProviderName.LOCAL,
                chunk_size=120,
                chunk_overlap=20,
                default_top_k=5,
            )
        )

        with patch("langchain_community.document_loaders.PyPDFLoader", FakePyPDFLoader):
            report = service.ingest_documents(docs)

        metadatas = vector_store_metadatas(service.vector_store)
        file_types = {metadata.get("file_type") for metadata in metadatas}
        sources = {metadata.get("source") for metadata in metadatas}

        self.assertEqual(report.scanned_files, 3)
        self.assertEqual(report.loaded_documents, 3)
        self.assertGreaterEqual(report.indexed_chunks, 3)
        self.assertTrue({"markdown", "text", "pdf"}.issubset(file_types))
        self.assertTrue({"hangzhou.md", "kyoto.txt", "seoul.pdf"}.issubset(sources))

    def test_destination_filter_limits_results(self) -> None:
        root = _fresh_case_dir("destination_filter")
        docs = root / "docs"
        docs.mkdir()
        (docs / "hangzhou.md").write_text(
            "---\ndestination: Hangzhou\n---\n# Hangzhou\nWest Lake and tea villages.",
            encoding="utf-8",
        )
        (docs / "tokyo.md").write_text(
            "---\ndestination: Tokyo\n---\n# Tokyo\nDisney and rail transit.",
            encoding="utf-8",
        )

        service = RagService(
            RagConfig(
                persist_dir=root / "chroma",
                embedding_provider=EmbeddingProviderName.LOCAL,
                default_top_k=5,
            )
        )
        service.ingest_markdown(docs)
        results = service.retrieve("rail transit", destination="Tokyo")

        self.assertTrue(results)
        self.assertTrue(all(result.destination == "Tokyo" for result in results))
        self.assertTrue(all(result.source and result.content for result in results))

    def test_metadata_filters_limit_results(self) -> None:
        root = _fresh_case_dir("metadata_filters")
        docs = root / "docs"
        _write_filter_docs(docs)

        service = RagService(
            RagConfig(
                persist_dir=root / "chroma",
                embedding_provider=EmbeddingProviderName.LOCAL,
                chunk_size=220,
                chunk_overlap=20,
                default_top_k=10,
            )
        )
        service.ingest_markdown(docs)

        budget_results = service.retrieve("budget transit", section="budget", top_k=10)
        family_results = service.retrieve(
            "travel advice",
            travel_type="family_free_independent",
            top_k=10,
        )
        autumn_results = service.retrieve("rain temple", season="autumn", top_k=10)

        self.assertTrue(budget_results)
        self.assertTrue(all(result.metadata["section"] == "budget" for result in budget_results))
        self.assertTrue(family_results)
        self.assertTrue(
            all(
                result.metadata["travel_type"] == "family_free_independent"
                for result in family_results
            )
        )
        self.assertTrue(autumn_results)
        self.assertTrue(all(result.metadata["destination"] == "Kyoto" for result in autumn_results))

    def test_reingest_replaces_existing_chunks_for_scanned_sources(self) -> None:
        root = _fresh_case_dir("reingest_replaces_sources")
        docs = root / "docs"
        docs.mkdir()
        source = docs / "hangzhou.md"
        source.write_text(
            "---\ndestination: Hangzhou\n---\n"
            "# Hangzhou\n旧内容：西湖适合慢游。\n",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
            top_k=5,
        )
        first_report = rag.ingest(docs)
        source.write_text(
            "---\ndestination: Hangzhou\n---\n"
            "# Hangzhou\n新内容：灵隐寺周末停车压力较高。\n",
            encoding="utf-8",
        )
        second_report = rag.ingest(docs)
        results = rag.search("杭州灵隐寺停车", destination="Hangzhou", top_k=5)
        joined = "\n".join(result.content for result in results)

        self.assertGreater(first_report.indexed_chunks, 0)
        self.assertEqual(second_report.deleted_chunks, first_report.indexed_chunks)
        self.assertEqual(rag.stats()["chunks"], second_report.indexed_chunks)
        self.assertIn("新内容", joined)
        self.assertNotIn("旧内容", joined)

    def test_incremental_ingest_skips_unchanged_documents(self) -> None:
        root = _fresh_case_dir("incremental_ingest")
        docs = root / "docs"
        docs.mkdir()
        (docs / "tokyo.md").write_text(
            "---\ndestination: Tokyo\n---\n# Tokyo\nDisney and rail transit.",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        first_report = rag.ingest(docs, incremental=True)
        second_report = rag.ingest(docs, incremental=True)

        self.assertEqual(first_report.skipped_unchanged, 0)
        self.assertEqual(second_report.skipped_unchanged, 1)
        self.assertEqual(second_report.indexed_chunks, 0)
        self.assertTrue(Path(second_report.manifest_path).exists())

    def test_keyword_retrieval_mode_hits_obvious_keyword_case(self) -> None:
        root = _fresh_case_dir("keyword_retrieval")
        docs = root / "docs"
        _write_filter_docs(docs)

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        results = rag.search(
            "temple tickets transit passes",
            destination="Kyoto",
            section="budget",
            top_k=3,
            retrieval_mode=RetrievalMode.KEYWORD,
        )

        self.assertTrue(results)
        self.assertEqual(results[0].metadata["section"], "budget")
        self.assertIn("temple tickets", results[0].content)

    def test_hybrid_retrieval_mode_returns_stable_results(self) -> None:
        root = _fresh_case_dir("hybrid_retrieval")
        docs = root / "docs"
        _write_filter_docs(docs)

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        results = rag.search(
            "Kyoto autumn rain temple paths",
            destination="Kyoto",
            section="weather_risk",
            top_k=3,
            retrieval_mode="hybrid",
        )

        self.assertTrue(results)
        self.assertEqual(results[0].source, "kyoto.md")
        self.assertTrue(all(result.metadata["section"] == "weather_risk" for result in results))

    def test_retrieval_trace_includes_stage_hits_and_filters(self) -> None:
        root = _fresh_case_dir("retrieval_trace_details")
        docs = root / "docs"
        _write_filter_docs(docs)

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
            top_k=3,
        )
        rag.ingest(docs)

        evidence = rag.retrieve_evidence(
            "Kyoto autumn rain temple paths",
            destination="Kyoto",
            section="weather_risk",
            top_k=3,
            retrieval_mode="hybrid",
        )
        trace = evidence.trace

        self.assertEqual(trace.metadata_filters["retriever"]["destination"], "Kyoto")
        self.assertEqual(trace.metadata_filters["retriever"]["section"], "weather_risk")
        self.assertTrue(trace.vector_hits)
        self.assertTrue(trace.keyword_hits)
        self.assertTrue(trace.fused_hits)
        self.assertTrue(trace.reranked_hits)
        self.assertEqual(trace.reranked_hits[0]["rank"], 1)
        self.assertIn("source", trace.reranked_hits[0])
        self.assertIn("section", trace.reranked_hits[0])
        self.assertIn("score", trace.reranked_hits[0])
        self.assertEqual(trace.empty_result_reason, "")

    def test_empty_retrieval_trace_records_reason(self) -> None:
        root = _fresh_case_dir("retrieval_trace_empty_reason")

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
            top_k=3,
        )

        evidence = rag.retrieve_evidence("Atlantis family budget", destination="Atlantis", top_k=3)

        self.assertFalse(evidence.results)
        self.assertTrue(evidence.trace.empty_result)
        self.assertEqual(evidence.trace.empty_result_reason, "empty_collection")

    def test_section_filter_is_inferred_from_question(self) -> None:
        root = _fresh_case_dir("infer_section")
        docs = root / "docs"
        _write_filter_docs(docs)

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        results = rag.search("Kyoto 交通怎么样？", destination="Kyoto", top_k=5)
        answer = rag.ask("Kyoto 交通怎么样？", destination="Kyoto", top_k=5)

        self.assertTrue(results)
        self.assertTrue(all(result.metadata["section"] == "traffic" for result in results))
        self.assertTrue(answer.results)
        self.assertTrue(all(result.metadata["section"] == "traffic" for result in answer.results))
        self.assertIn("rail and bus transit", answer.answer)

    def test_answer_bullets_include_destination_labels(self) -> None:
        root = _fresh_case_dir("answer_destination_labels")
        docs = root / "docs"
        _write_filter_docs(docs)

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        answer = rag.ask("travel advice", top_k=4)

        self.assertTrue(answer.results)
        self.assertRegex(answer.answer, r"- \[(Kyoto|Osaka)( / [^\]]+)?\] ")

    def test_known_destination_answer_labels_use_chinese_names(self) -> None:
        root = _fresh_case_dir("answer_chinese_destination_labels")
        docs = root / "docs"
        docs.mkdir()
        (docs / "beijing.md").write_text(
            "---\n"
            "destination: Beijing\n"
            "city: Beijing\n"
            "country: China\n"
            "travel_type: family_free_independent\n"
            "season: spring,autumn\n"
            "source_type: destination_guide\n"
            "updated_at: 2026-05-08\n"
            "---\n"
            "# Beijing\n\n"
            "## 交通\n"
            "北京地铁覆盖主要景点，换乘站较大。\n",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        answer = rag.ask("北京交通怎么样？", top_k=3)

        self.assertTrue(answer.results)
        self.assertIn("[北京 / 交通]", answer.answer)

    def test_travel_rag_facade_and_public_api_support_metadata_filters(self) -> None:
        root = _fresh_case_dir("public_metadata_filters")
        docs = root / "docs"
        _write_filter_docs(docs)

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        search_results = rag.search(
            "budget",
            section="budget",
            travel_type="family_free_independent",
            top_k=5,
        )
        answer = rag.ask(
            "What should Kyoto budget include?",
            section="budget",
            season="autumn",
            top_k=5,
        )
        one_shot_results = search_destination_knowledge(
            "budget",
            section="budget",
            travel_type="family_free_independent",
            top_k=5,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )
        one_shot_query = query_destination_knowledge(
            "rain",
            season="autumn",
            top_k=5,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )
        one_shot_answer = answer_destination_question(
            "budget",
            section="budget",
            top_k=5,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )
        evidence = retrieve_destination_evidence(
            "What should Kyoto budget include?",
            section="budget",
            top_k=5,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )
        retriever = get_destination_retriever(
            destination="Kyoto",
            section="budget",
            top_k=5,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )
        retriever_docs = retriever.invoke("budget")

        self.assertTrue(search_results)
        self.assertTrue(all(result.metadata["section"] == "budget" for result in search_results))
        self.assertTrue(answer.results)
        self.assertTrue(all(result.metadata["section"] == "budget" for result in answer.results))
        self.assertIn("Kyoto family budget", answer.answer)
        self.assertTrue(one_shot_results)
        self.assertTrue(
            all(
                result.metadata["travel_type"] == "family_free_independent"
                for result in one_shot_results
            )
        )
        self.assertTrue(one_shot_query.results)
        self.assertTrue(
            all(result.metadata["destination"] == "Kyoto" for result in one_shot_query.results)
        )
        self.assertTrue(one_shot_answer.results)
        self.assertTrue(
            all(result.metadata["section"] == "budget" for result in one_shot_answer.results)
        )
        self.assertTrue(evidence.results)
        self.assertEqual(evidence.trace.section, "budget")
        self.assertGreaterEqual(evidence.trace.total_latency_ms, 0)
        self.assertTrue(retriever_docs)
        self.assertTrue(
            all(document.metadata["section"] == "budget" for document in retriever_docs)
        )
        self.assertTrue(
            all(document.metadata["destination"] == "Kyoto" for document in retriever_docs)
        )

    def test_cli_query_supports_metadata_filters_and_section_column(self) -> None:
        root = _fresh_case_dir("cli_metadata_filters")
        docs = root / "docs"
        _write_filter_docs(docs)
        persist_dir = root / "chroma"
        runner = CliRunner()

        ingest_result = runner.invoke(
            app,
            [
                "ingest",
                str(docs),
                "--persist-dir",
                str(persist_dir),
                "--embedding-provider",
                "local",
            ],
        )
        self.assertEqual(ingest_result.exit_code, 0, ingest_result.output)

        json_result = runner.invoke(
            app,
            [
                "query",
                "budget",
                "--persist-dir",
                str(persist_dir),
                "--embedding-provider",
                "local",
                "--section",
                "budget",
                "--travel-type",
                "family_free_independent",
                "--retrieval-mode",
                "keyword",
                "--json",
            ],
        )
        self.assertEqual(json_result.exit_code, 0, json_result.output)
        payload = json.loads(json_result.output)
        self.assertTrue(payload["results"])
        self.assertTrue(
            all(result["metadata"]["section"] == "budget" for result in payload["results"])
        )

        table_result = runner.invoke(
            app,
            [
                "query",
                "budget",
                "--persist-dir",
                str(persist_dir),
                "--embedding-provider",
                "local",
                "--section",
                "budget",
            ],
        )
        self.assertEqual(table_result.exit_code, 0, table_result.output)
        self.assertIn("Section", table_result.output)

    def test_cli_verify_embedding_supports_local_json(self) -> None:
        runner = CliRunner()

        result = runner.invoke(
            app,
            [
                "verify-embedding",
                "--embedding-provider",
                "local",
                "--json",
            ],
        )
        payload = json.loads(result.output)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["embedding_provider"], "local")
        self.assertTrue(payload["finite"])
        self.assertTrue(payload["consistent_dimensions"])

    def test_front_matter_and_markdown_sections_enter_chunk_metadata(self) -> None:
        root = _fresh_case_dir("metadata_sections")
        docs = root / "docs"
        docs.mkdir()
        (docs / "kyoto.md").write_text(
            "---\n"
            "destination: Kyoto\n"
            "city: Kyoto\n"
            "country: Japan\n"
            "travel_type: family\n"
            "season: autumn\n"
            "source_type: destination_guide\n"
            "updated_at: 2026-05-08\n"
            "---\n"
            "# Kyoto Family Trip\n"
            "Kyoto is suitable for slow family travel.\n\n"
            "## 交通\n"
            "Use rail and buses between Kyoto Station, Arashiyama, and Gion.\n\n"
            "## 玩法\n"
            "Plan temples, riverside walks, and one relaxed food market visit.\n\n"
            "## 预算\n"
            "Budget for transit passes, meals, and temple tickets.\n\n"
            "## 风险提醒\n"
            "Popular temple areas are crowded during peak foliage season.\n",
            encoding="utf-8",
        )

        service = RagService(
            RagConfig(
                persist_dir=root / "chroma",
                embedding_provider=EmbeddingProviderName.LOCAL,
                chunk_size=160,
                chunk_overlap=20,
                default_top_k=5,
            )
        )

        report = service.ingest_markdown(docs)
        metadatas = vector_store_metadatas(service.vector_store)
        sections = {str(metadata.get("section")) for metadata in metadatas}
        section_titles = {str(metadata.get("section_title")) for metadata in metadatas}

        self.assertGreaterEqual(report.indexed_chunks, 5)
        self.assertTrue(metadatas)
        for metadata in metadatas:
            for field in CHUNK_METADATA_FIELDS:
                self.assertIn(field, metadata)
            self.assertEqual(metadata["destination"], "Kyoto")
            self.assertEqual(metadata["city"], "Kyoto")
            self.assertEqual(metadata["country"], "Japan")
            self.assertEqual(metadata["travel_type"], "family")
            self.assertEqual(metadata["season"], "autumn")
            self.assertEqual(metadata["source_type"], "destination_guide")
            self.assertEqual(metadata["updated_at"], "2026-05-08")
            self.assertEqual(metadata["language"], "zh")
            self.assertTrue(metadata["document_id"])
            self.assertTrue(metadata["document_hash"])
            self.assertTrue(metadata["section"])
            self.assertTrue(metadata["section_title"])

        self.assertTrue({"traffic", "itinerary", "budget", "risk"}.issubset(sections))
        self.assertTrue({"交通", "玩法", "预算", "风险提醒"}.issubset(section_titles))

    def test_planning_section_aliases_are_mapped(self) -> None:
        document = Document(
            page_content=(
                "# Test\n\n"
                "## 概览\nA\n"
                "## 适合人群\nB\n"
                "## 交通\nC\n"
                "## 玩法\nD\n"
                "## 预算\nE\n"
                "## 住宿\nF\n"
                "## 餐饮\nG\n"
                "## 拥挤风险\nH\n"
                "## 天气风险\nI\n"
                "## 备选方案\nJ\n"
            ),
            metadata={"title": "Test", "source": "test.md", "destination": "Test"},
        )

        sections = [
            section.metadata["section"]
            for section in split_markdown_sections(document)
            if section.metadata["section_title"] != "Test"
        ]

        self.assertEqual(
            sections,
            [
                "overview",
                "audience",
                "traffic",
                "itinerary",
                "budget",
                "lodging",
                "dining",
                "crowd_risk",
                "weather_risk",
                "alternatives",
            ],
        )

    def test_public_api_can_be_called_by_external_code(self) -> None:
        root = _fresh_case_dir("public_api")
        docs = root / "docs"
        docs.mkdir()
        (docs / "tokyo.md").write_text(
            "---\ndestination: Tokyo\n---\n# Tokyo\nDisney and rail transit.",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        report = rag.ingest(docs)
        results = rag.search("rail transit", destination="Tokyo", top_k=1)
        one_shot_results = search_destination_knowledge(
            "rail transit",
            destination="Tokyo",
            top_k=1,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )

        self.assertEqual(report.indexed_chunks, 1)
        self.assertEqual(results[0].destination, "Tokyo")
        self.assertEqual(one_shot_results[0].destination, "Tokyo")

    def test_public_api_can_answer_from_current_knowledge_base(self) -> None:
        root = _fresh_case_dir("public_answer")
        docs = root / "docs"
        docs.mkdir()
        (docs / "hangzhou.md").write_text(
            "---\ndestination: Hangzhou\n---\n"
            "# Hangzhou\n灵隐寺周边停车位紧张，周末上午 9 点后车流压力较高。",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)

        response = rag.ask("灵隐寺周末拥挤吗？", destination="Hangzhou", top_k=1)
        one_shot_response = answer_destination_question(
            "灵隐寺周末拥挤吗？",
            destination="Hangzhou",
            top_k=1,
            persist_dir=root / "chroma",
            embedding_provider="local",
        )

        self.assertIn("灵隐寺", response.answer)
        self.assertIn("hangzhou.md", response.answer)
        self.assertIn("灵隐寺", one_shot_response.answer)

    def test_ask_infers_destination_from_question(self) -> None:
        root = _fresh_case_dir("infer_destination")
        docs = root / "docs"
        docs.mkdir()
        (docs / "hangzhou.md").write_text(
            "---\ndestination: Hangzhou\n---\n# Hangzhou\n杭州西湖周末游客较多。",
            encoding="utf-8",
        )
        (docs / "tokyo.md").write_text(
            "---\ndestination: Tokyo\n---\n# Tokyo\n东京迪士尼周末和日本节假日拥挤风险较高。",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)
        response = rag.ask("东京周末拥挤吗？", top_k=2)

        self.assertTrue(response.results)
        self.assertTrue(all(result.destination == "Tokyo" for result in response.results))
        self.assertIn("东京迪士尼", response.answer)

    def test_ask_infers_new_destination_from_chinese_alias(self) -> None:
        root = _fresh_case_dir("infer_new_destination")
        docs = root / "docs"
        docs.mkdir()
        (docs / "hangzhou.md").write_text(
            "---\ndestination: Hangzhou\n---\n# Hangzhou\n杭州适合西湖慢游。",
            encoding="utf-8",
        )
        (docs / "suzhou.md").write_text(
            "---\ndestination: Suzhou\n---\n# Suzhou\n苏州适合园林古城自由行。",
            encoding="utf-8",
        )

        rag = TravelRag.create(
            persist_dir=root / "chroma",
            embedding_provider=EmbeddingProviderName.LOCAL,
        )
        rag.ingest(docs)
        response = rag.ask("苏州适合怎么玩？", top_k=2)

        self.assertTrue(response.results)
        self.assertTrue(all(result.destination == "Suzhou" for result in response.results))
        self.assertIn("苏州适合园林古城自由行", response.answer)


if __name__ == "__main__":
    unittest.main()
