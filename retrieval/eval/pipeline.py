"""CLIR 评测流程封装"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from logger import get_logger
from config import config
from retrieval.dense.dense_search import DenseSearcher
from retrieval.sparse.sparse_search import SparseSearcher
from retrieval.kg_expansion.entity_linking import EntityLinker
from retrieval.kg_expansion.hop_expand import HopExpander
from retrieval.kg_expansion.kg_path_score import KGPathScorer
from retrieval.rerank.fusion_rerank import FusionReranker
from retrieval.eval.run_eval import Evaluator


logger = get_logger(__name__)


class CLIREvaluationPipeline:
    """CLIR评测流程"""

    def __init__(
        self,
        corpus_dir: Path,
        queries_file: Path,
        qrels_file: Path,
        output_dir: Path,
        dense_index_dir: Path = None,
        sparse_index_dir: Path = None,
        use_kg: bool = True,
        top_k: int = 100
    ):
        self.corpus_dir = Path(corpus_dir)
        self.queries_file = Path(queries_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dense_index_dir = Path(dense_index_dir) if dense_index_dir else None
        self.sparse_index_dir = Path(sparse_index_dir) if sparse_index_dir else None
        self.use_kg = use_kg
        self.top_k = top_k

        self.queries = self._load_queries()
        self.corpus_files = self._discover_corpus_files()

        self.evaluator = Evaluator(
            qrels_file=self.qrels_file,
            metrics=["ndcg", "mrr", "recall"],
            k_values={"ndcg": 10, "recall": 50}
        )

        logger.info(f"评测流程初始化: {len(self.queries)} 个查询")

    def _load_queries(self) -> Dict[str, Dict]:
        queries = {}
        if not self.queries_file.exists():
            logger.error(f"查询文件不存在: {self.queries_file}")
            return queries

        with open(self.queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    qid = obj["qid"]
                    queries[qid] = obj
                except Exception as e:
                    logger.error(f"解析查询失败: {e}")

        logger.info(f"加载查询: {len(queries)} 条")
        return queries

    def _discover_corpus_files(self) -> List[Path]:
        patterns = ["corpus_*_cleaned.jsonl", "*.jsonl"]
        for pattern in patterns:
            matches = sorted(self.corpus_dir.glob(pattern))
            if matches:
                return matches
        logger.warning(f"在 {self.corpus_dir} 未找到清洗后的语料文件")
        return []

    def _load_corpus_to_searcher(self, searcher: DenseSearcher) -> None:
        if self.corpus_files:
            searcher.load_corpus(self.corpus_files)
        else:
            logger.warning("未加载语料, 结果将不包含文档详情")

    def run_dense_only(self) -> Path:
        logger.info("\n=== 运行 Dense-only 基线 ===")
        searcher = DenseSearcher(index_dir=self.dense_index_dir)
        self._load_corpus_to_searcher(searcher)

        results_dict = {}
        for qid, query_obj in self.queries.items():
            query_text = query_obj["text"] if "text" in query_obj else query_obj.get("query", "")
            results = searcher.search(query=query_text, top_k=self.top_k)
            results_dict[qid] = [r["doc_id"] for r in results]

        output_file = self.output_dir / "results_dense_only.jsonl"
        self._export_results(results_dict, output_file, run_name="dense_only")
        logger.info(f"Dense-only 结果已保存: {output_file}")
        return output_file

    def run_sparse_only(self) -> Path:
        logger.info("\n=== 运行 Sparse-only 基线 ===")
        searcher = SparseSearcher(index_dir=self.sparse_index_dir)

        results_dict = {}
        for qid, query_obj in self.queries.items():
            query_text = query_obj.get("text") or query_obj.get("query", "")
            query_lang = query_obj.get("lang", "fr")
            results = searcher.search(query=query_text, lang=query_lang, top_k=self.top_k)
            results_dict[qid] = [r["doc_id"] for r in results]

        output_file = self.output_dir / "results_sparse_only.jsonl"
        self._export_results(results_dict, output_file, run_name="sparse_only")
        logger.info(f"Sparse-only 结果已保存: {output_file}")
        return output_file

    def run_kg_clir(self) -> Path:
        logger.info("\n=== 运行 KG-CLIR (Ours) ===")
        dense_searcher = DenseSearcher(index_dir=self.dense_index_dir)
        self._load_corpus_to_searcher(dense_searcher)
        sparse_searcher = SparseSearcher(index_dir=self.sparse_index_dir)

        if self.use_kg:
            try:
                entity_linker = EntityLinker()
                hop_expander = HopExpander()
                path_scorer = KGPathScorer()
            except Exception as e:
                logger.warning(f"KG组件初始化失败: {e}, 退化为Dense+Sparse")
                self.use_kg = False
                entity_linker = None
                hop_expander = None
                path_scorer = None
        else:
            entity_linker = None
            hop_expander = None
            path_scorer = None

        reranker = FusionReranker(
            alpha=config.ALPHA_DENSE,
            beta=config.BETA_SPARSE,
            gamma=config.GAMMA_KG
        )

        results_dict = {}
        for qid, query_obj in self.queries.items():
            query_text = query_obj.get("text") or query_obj.get("query", "")
            query_lang = query_obj.get("lang", "fr")

            dense_results = dense_searcher.search(query=query_text, top_k=self.top_k)
            sparse_results = sparse_searcher.search(
                query=query_text,
                lang=query_lang,
                top_k=self.top_k
            )

            kg_scores = {}
            if self.use_kg and entity_linker and hop_expander and path_scorer:
                entities = entity_linker.link_query(query_text, lang=query_lang)
                entity_ids = [e.get("kg_id") for e in entities if e.get("kg_id")]
                if entity_ids:
                    expansion = hop_expander.expand_multi_entities(entity_ids, max_hops=config.KG_HOP_LIMIT)
                    kg_docs = path_scorer.score_documents(
                        query_entities=entity_ids,
                        expanded_graph=expansion,
                        corpus_map=dense_searcher.corpus_map
                    )
                    kg_scores = {doc["doc_id"]: doc.get("score", 0.0) for doc in kg_docs}

            fused_results = reranker.fuse_scores(
                dense_results=dense_results,
                sparse_results=sparse_results,
                kg_scores=kg_scores,
                method="weighted_sum"
            )

            results_dict[qid] = [r["doc_id"] for r in fused_results]

        output_file = self.output_dir / "results_kg_clir.jsonl"
        self._export_results(results_dict, output_file, run_name="kg_clir")
        logger.info(f"KG-CLIR 结果已保存: {output_file}")
        return output_file

    def _export_results(self, results_dict: Dict[str, List[str]], output_file: Path, run_name: str = "run"):
        with open(output_file, 'w', encoding='utf-8') as f:
            for qid, doc_ids in results_dict.items():
                for rank, doc_id in enumerate(doc_ids, 1):
                    obj = {
                        "qid": qid,
                        "doc_id": doc_id,
                        "rank": rank,
                        "run_name": run_name
                    }
                    f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    def run_evaluation(self):
        logger.info("\n" + "=" * 60)
        logger.info("开始CLIR评测实验")
        logger.info("=" * 60)

        results_files = {}

        if self.dense_index_dir and self.dense_index_dir.exists():
            try:
                dense_file = self.run_dense_only()
                results_files["Dense-only"] = dense_file
            except Exception as e:
                logger.error(f"Dense-only 失败: {e}")
        else:
            logger.warning("Dense索引不存在,跳过Dense-only")

        if self.sparse_index_dir and self.sparse_index_dir.exists():
            try:
                sparse_file = self.run_sparse_only()
                results_files["Sparse-only"] = sparse_file
            except Exception as e:
                logger.error(f"Sparse-only 失败: {e}")
        else:
            logger.warning("Sparse索引不存在,跳过Sparse-only")

        try:
            kg_clir_file = self.run_kg_clir()
            results_files["KG-CLIR (Ours)"] = kg_clir_file
        except Exception as e:
            logger.error(f"KG-CLIR 失败: {e}")

        if results_files:
            logger.info("\n" + "=" * 60)
            logger.info("对比评测结果")
            logger.info("=" * 60)

            all_metrics = self.evaluator.compare_runs(results_files)
            summary_file = self.output_dir / "evaluation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            logger.info(f"\n评测汇总已保存: {summary_file}")
            self._generate_paper_table(all_metrics)
        else:
            logger.error("没有成功运行任何方法,评测失败")

    def _generate_paper_table(self, all_metrics: Dict[str, Dict[str, float]]):
        logger.info("\n=== 论文表格(LaTeX格式) ===")
        latex = r"""
\begin{table}[h]
\centering
\caption{Cross-lingual Information Retrieval Performance Comparison}
\label{tab:clir_results}
\begin{tabular}{lccc}
\toprule
Method & nDCG@10 & MRR & Recall@50 \\
\midrule
"""

        method_order = ["Dense-only", "Sparse-only", "KG-CLIR (Ours)"]
        for method in method_order:
            if method not in all_metrics:
                continue
            metrics = all_metrics[method]
            ndcg = metrics.get("ndcg@10", 0.0)
            mrr = metrics.get("mrr", 0.0)
            recall = metrics.get("recall@50", 0.0)
            best_mark = r" \textbf{*}" if method == "KG-CLIR (Ours)" else ""
            latex += f"{method}{best_mark} & {ndcg:.3f} & {mrr:.3f} & {recall:.3f} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

        print(latex)
        table_file = self.output_dir / "paper_table.tex"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(latex)
        logger.info(f"\nLaTeX表格已保存: {table_file}")

    # Remaining methods identical to previous class (copy).跳? Need to copy methods from script.
