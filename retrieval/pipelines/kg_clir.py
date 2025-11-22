"""KG-CLIR 系统封装,可供脚本和单元测试复用"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from logger import get_logger
from retrieval.dense.dense_search import DenseSearcher
from retrieval.sparse.sparse_search import SparseSearcher
from retrieval.kg_expansion.entity_linking import EntityLinker
from retrieval.kg_expansion.hop_expand import HopExpander
from retrieval.kg_expansion.kg_path_score import KGPathScorer
from retrieval.rerank.fusion_rerank import FusionReranker


logger = get_logger(__name__)


class KGCLIRSystem:
    """KG增强的跨语言信息检索系统"""

    def __init__(
        self,
        dense_index_dir: Path,
        sparse_index_dir: Path,
        use_kg: bool = True,
        alpha: float = None,
        beta: float = None,
        gamma: float = None,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None,
        max_hops: int = 2,
        top_k_per_source: int = 50
    ):
        self.dense_searcher = DenseSearcher(index_dir=dense_index_dir)
        self.sparse_searcher = SparseSearcher(index_dir=sparse_index_dir)

        self.use_kg = use_kg
        if use_kg:
            self.entity_linker = EntityLinker(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password
            )
            self.hop_expander = HopExpander(
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password
            )
            self.kg_scorer = KGPathScorer()
        else:
            self.entity_linker = None
            self.hop_expander = None
            self.kg_scorer = None

        self.reranker = FusionReranker(alpha=alpha, beta=beta, gamma=gamma if use_kg else 0)
        self.max_hops = max_hops
        self.top_k_per_source = top_k_per_source

    def search(self, query: str, lang: str = "auto", top_k: int = 10, explain: bool = False) -> List[Dict]:
        start_time = time.time()
        dense_results = self.dense_searcher.search(query=query, top_k=self.top_k_per_source)
        sparse_results = self.sparse_searcher.search(
            query=query,
            lang=None if lang == "auto" else lang,
            top_k=self.top_k_per_source
        )

        kg_results: List[Dict] = []
        if self.use_kg and self.entity_linker:
            linked = self.entity_linker.link_query(query, lang=lang)
            entity_ids = [e.get("kg_id") for e in linked if e.get("kg_id")]
            if entity_ids:
                expanded = self.hop_expander.expand_multi_entities(
                    entity_ids=entity_ids,
                    max_hops=self.max_hops
                )
                kg_results = self.kg_scorer.score_documents(
                    query_entities=entity_ids,
                    expanded_graph=expanded,
                    corpus_map=self.dense_searcher.corpus_map
                )

        final_results = self.reranker.rerank(
            dense_results=dense_results,
            sparse_results=sparse_results,
            kg_results=kg_results,
            top_k=top_k,
            corpus=self.dense_searcher.corpus_map
        )

        if explain:
            for item in final_results:
                item["fusion_config"] = {
                    "alpha": self.reranker.alpha,
                    "beta": self.reranker.beta,
                    "gamma": self.reranker.gamma
                }
                item["query"] = query
                item["query_lang"] = lang

        logger.info("检索完成, 返回 %d 个结果 (%.2fs)", len(final_results), time.time() - start_time)
        return final_results

    def batch_search(self, queries: List[Dict], top_k: int = 10, explain: bool = False) -> Dict[str, List[Dict]]:
        results = {}
        for item in queries:
            qid = item.get("qid") or item.get("id")
            query = item.get("query", "")
            lang = item.get("lang", "auto")
            if not query:
                continue
            results[qid] = self.search(query=query, lang=lang, top_k=top_k, explain=explain)
        return results

    def close(self):
        components = [self.dense_searcher, self.sparse_searcher, self.entity_linker, self.hop_expander]
        for component in components:
            close_fn = getattr(component, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as exc:
                    logger.debug(f"关闭组件失败: {exc}")


def print_results(results: List[Dict], top_k: int = 10) -> None:
    print("\n" + "=" * 80)
    print(f"检索结果 (Top {min(len(results), top_k)})")
    print("=" * 80)

    for idx, result in enumerate(results[:top_k], 1):
        print(f"[{idx}] {result.get('doc_id')} | {result.get('title', '')}")
        print(f"  Lang: {result.get('lang', 'N/A')} | Score: {result.get('fused_score', 0.0):.4f}")
        snippet = (result.get("content", "") or "")[:160].replace("\n", " ")
        if snippet:
            print(f"  Snippet: {snippet}...")
        contrib = result.get("score_contributions", {})
        if contrib:
            print(
                f"  Contributions -> Dense: {contrib.get('dense', 0):.3f}, "
                f"Sparse: {contrib.get('sparse', 0):.3f}, KG: {contrib.get('kg', 0):.3f}"
            )
        print()
