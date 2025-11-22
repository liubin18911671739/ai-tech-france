#!/usr/bin/env python3
"""
完整CLIR评测流程

运行Dense-only, Sparse-only, KG-CLIR三种方法的对比实验
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from logger import get_logger
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
        """
        初始化评测流程
        
        Args:
            corpus_dir: 语料目录
            queries_file: 查询文件
            qrels_file: 相关性标注
            output_dir: 输出目录
            dense_index_dir: Dense索引
            sparse_index_dir: Sparse索引
            use_kg: 是否使用KG增强
            top_k: 返回结果数
        """
        self.corpus_dir = Path(corpus_dir)
        self.queries_file = Path(queries_file)
        self.qrels_file = Path(qrels_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dense_index_dir = Path(dense_index_dir) if dense_index_dir else None
        self.sparse_index_dir = Path(sparse_index_dir) if sparse_index_dir else None
        self.use_kg = use_kg
        self.top_k = top_k
        
        # 加载查询
        self.queries = self._load_queries()
        
        # 初始化评测器
        self.evaluator = Evaluator(
            qrels_file=self.qrels_file,
            metrics=["ndcg", "mrr", "recall"],
            k_values={"ndcg": 10, "recall": 50}
        )
        
        logger.info(f"评测流程初始化: {len(self.queries)} 个查询")
    
    def _load_queries(self) -> Dict[str, Dict]:
        """
        加载查询
        
        Returns:
            {qid: {"text": "...", "lang": "fr"}}
        """
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
                    continue
        
        logger.info(f"加载查询: {len(queries)} 条")
        return queries
    
    def run_dense_only(self) -> Path:
        """
        运行Dense-only基线
        
        Returns:
            结果文件路径
        """
        logger.info("\n=== 运行 Dense-only 基线 ===")
        
        # 初始化Dense检索器
        searcher = DenseSearcher(
            index_dir=self.dense_index_dir,
            corpus_file=self.corpus_dir / "corpus_cleaned.jsonl"
        )
        
        # 检索
        results_dict = {}
        for qid, query_obj in self.queries.items():
            query_text = query_obj["text"]
            
            results = searcher.search(
                query=query_text,
                top_k=self.top_k
            )
            
            results_dict[qid] = [r["doc_id"] for r in results]
        
        # 导出结果
        output_file = self.output_dir / "results_dense_only.jsonl"
        self._export_results(results_dict, output_file, run_name="dense_only")
        
        logger.info(f"Dense-only 结果已保存: {output_file}")
        return output_file
    
    def run_sparse_only(self) -> Path:
        """
        运行Sparse-only基线
        
        Returns:
            结果文件路径
        """
        logger.info("\n=== 运行 Sparse-only 基线 ===")
        
        # 初始化Sparse检索器
        searcher = SparseSearcher(
            index_dir=self.sparse_index_dir
        )
        
        # 检索
        results_dict = {}
        for qid, query_obj in self.queries.items():
            query_text = query_obj["text"]
            query_lang = query_obj.get("lang", "fr")
            
            results = searcher.search(
                query=query_text,
                lang=query_lang,
                top_k=self.top_k
            )
            
            results_dict[qid] = [r["doc_id"] for r in results]
        
        # 导出结果
        output_file = self.output_dir / "results_sparse_only.jsonl"
        self._export_results(results_dict, output_file, run_name="sparse_only")
        
        logger.info(f"Sparse-only 结果已保存: {output_file}")
        return output_file
    
    def run_kg_clir(self) -> Path:
        """
        运行KG-CLIR(完整方法)
        
        Returns:
            结果文件路径
        """
        logger.info("\n=== 运行 KG-CLIR (Ours) ===")
        
        # 初始化组件
        dense_searcher = DenseSearcher(
            index_dir=self.dense_index_dir,
            corpus_file=self.corpus_dir / "corpus_cleaned.jsonl"
        )
        
        sparse_searcher = SparseSearcher(
            index_dir=self.sparse_index_dir
        )
        
        # KG组件(如果启用)
        if self.use_kg:
            try:
                entity_linker = EntityLinker()
                hop_expander = HopExpander()
                path_scorer = KGPathScorer()
            except Exception as e:
                logger.warning(f"KG组件初始化失败: {e}, 退化为Dense+Sparse")
                self.use_kg = False
        
        # 融合器
        reranker = FusionReranker(
            alpha=config.FUSION_ALPHA,
            beta=config.FUSION_BETA,
            gamma=config.FUSION_GAMMA
        )
        
        # 检索
        results_dict = {}
        
        for qid, query_obj in self.queries.items():
            query_text = query_obj["text"]
            query_lang = query_obj.get("lang", "fr")
            
            # Dense检索
            dense_results = dense_searcher.search(
                query=query_text,
                top_k=self.top_k
            )
            
            # Sparse检索
            sparse_results = sparse_searcher.search(
                query=query_text,
                lang=query_lang,
                top_k=self.top_k
            )
            
            # KG增强(如果启用)
            kg_scores = {}
            if self.use_kg:
                try:
                    # 实体链接
                    entities = [{"text": query_text, "type": "CONCEPT"}]
                    linked = entity_linker.link_entities(entities, lang=query_lang)
                    
                    if linked:
                        # N-hop扩展
                        node_ids = [e["neo4j_id"] for e in linked]
                        expansion = hop_expander.expand_from_nodes(
                            node_ids=node_ids,
                            hops=config.KG_MAX_HOPS
                        )
                        
                        # 路径评分
                        paths = expansion.get("paths", [])
                        if paths:
                            kg_scores = path_scorer.score_nodes_from_paths(
                                paths,
                                method="max"
                            )
                
                except Exception as e:
                    logger.warning(f"查询 {qid} KG增强失败: {e}")
            
            # 融合排序
            fused_results = reranker.fuse_scores(
                dense_results=dense_results,
                sparse_results=sparse_results,
                kg_scores=kg_scores,
                method="weighted_sum"
            )
            
            results_dict[qid] = [r["doc_id"] for r in fused_results]
        
        # 导出结果
        output_file = self.output_dir / "results_kg_clir.jsonl"
        self._export_results(results_dict, output_file, run_name="kg_clir")
        
        logger.info(f"KG-CLIR 结果已保存: {output_file}")
        return output_file
    
    def _export_results(
        self,
        results_dict: Dict[str, List[str]],
        output_file: Path,
        run_name: str = "run"
    ):
        """
        导出检索结果(JSONL格式)
        
        Args:
            results_dict: {qid: [doc_ids]}
            output_file: 输出文件
            run_name: 运行名称
        """
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
        """运行完整评测流程"""
        logger.info("\n" + "=" * 60)
        logger.info("开始CLIR评测实验")
        logger.info("=" * 60)
        
        results_files = {}
        
        # 运行Dense-only
        if self.dense_index_dir and self.dense_index_dir.exists():
            try:
                dense_file = self.run_dense_only()
                results_files["Dense-only"] = dense_file
            except Exception as e:
                logger.error(f"Dense-only 失败: {e}")
        else:
            logger.warning("Dense索引不存在,跳过Dense-only")
        
        # 运行Sparse-only
        if self.sparse_index_dir and self.sparse_index_dir.exists():
            try:
                sparse_file = self.run_sparse_only()
                results_files["Sparse-only"] = sparse_file
            except Exception as e:
                logger.error(f"Sparse-only 失败: {e}")
        else:
            logger.warning("Sparse索引不存在,跳过Sparse-only")
        
        # 运行KG-CLIR
        try:
            kg_clir_file = self.run_kg_clir()
            results_files["KG-CLIR (Ours)"] = kg_clir_file
        except Exception as e:
            logger.error(f"KG-CLIR 失败: {e}")
        
        # 对比评测
        if results_files:
            logger.info("\n" + "=" * 60)
            logger.info("对比评测结果")
            logger.info("=" * 60)
            
            all_metrics = self.evaluator.compare_runs(results_files)
            
            # 导出汇总
            summary_file = self.output_dir / "evaluation_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\n评测汇总已保存: {summary_file}")
            
            # 生成论文表格
            self._generate_paper_table(all_metrics)
        
        else:
            logger.error("没有成功运行任何方法,评测失败")
    
    def _generate_paper_table(self, all_metrics: Dict[str, Dict[str, float]]):
        """
        生成论文LaTeX表格
        
        Args:
            all_metrics: {run_name: {metric: score}}
        """
        logger.info("\n=== 论文表格(LaTeX格式) ===")
        
        # 表头
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
        
        # 数据行
        method_order = ["Dense-only", "Sparse-only", "KG-CLIR (Ours)"]
        for method in method_order:
            if method not in all_metrics:
                continue
            
            metrics = all_metrics[method]
            ndcg = metrics.get("ndcg@10", 0.0)
            mrr = metrics.get("mrr", 0.0)
            recall = metrics.get("recall@50", 0.0)
            
            # 标注最佳结果
            best_mark = r" \textbf{*}" if method == "KG-CLIR (Ours)" else ""
            
            latex += f"{method}{best_mark} & {ndcg:.3f} & {mrr:.3f} & {recall:.3f} \\\\\n"
        
        # 表尾
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        
        print(latex)
        
        # 保存到文件
        table_file = self.output_dir / "paper_table.tex"
        with open(table_file, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        logger.info(f"\nLaTeX表格已保存: {table_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="完整CLIR评测流程")
    
    # 输入文件
    parser.add_argument("--corpus-dir", type=str,
                       default="data/cleaned",
                       help="语料目录")
    parser.add_argument("--queries", type=str,
                       default="data/eval/clir_queries.jsonl",
                       help="查询文件")
    parser.add_argument("--qrels", type=str,
                       default="data/eval/qrels.tsv",
                       help="相关性标注文件")
    
    # 索引目录
    parser.add_argument("--dense-index", type=str,
                       default="artifacts/faiss_labse",
                       help="Dense索引目录")
    parser.add_argument("--sparse-index", type=str,
                       default="artifacts/whoosh_bm25",
                       help="Sparse索引目录")
    
    # 输出
    parser.add_argument("--output-dir", type=str,
                       default="artifacts/eval_results",
                       help="输出目录")
    
    # 参数
    parser.add_argument("--use-kg", action="store_true",
                       help="是否使用KG增强")
    parser.add_argument("--top-k", type=int, default=100,
                       help="返回结果数")
    
    args = parser.parse_args()
    
    # 创建评测流程
    pipeline = CLIREvaluationPipeline(
        corpus_dir=Path(args.corpus_dir),
        queries_file=Path(args.queries),
        qrels_file=Path(args.qrels),
        output_dir=Path(args.output_dir),
        dense_index_dir=Path(args.dense_index) if args.dense_index else None,
        sparse_index_dir=Path(args.sparse_index) if args.sparse_index else None,
        use_kg=args.use_kg,
        top_k=args.top_k
    )
    
    # 运行评测
    pipeline.run_evaluation()
    
    logger.info("\n✅ 评测完成!")


if __name__ == "__main__":
    main()
