#!/usr/bin/env python3
"""
消融实验 (Ablation Study)

系统化评测不同组件对检索性能的影响：
1. Dense-only
2. Sparse-only  
3. KG-only
4. Dense+Sparse
5. Dense+KG
6. Sparse+KG
7. Dense+Sparse+KG (Full)

生成完整的对比实验结果和LaTeX表格
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config
from retrieval.dense.dense_search import DenseSearcher
from retrieval.sparse.sparse_search import SparseSearcher
from retrieval.kg_expansion.entity_linking import EntityLinker
from retrieval.kg_expansion.hop_expand import HopExpander
from retrieval.kg_expansion.kg_path_score import KGPathScorer
from retrieval.rerank.fusion_rerank import FusionReranker
from retrieval.eval.metrics import compute_ndcg, compute_mrr, compute_recall

logger = get_logger(__name__)


class AblationExperiment:
    """消融实验管理器"""
    
    # 实验配置：每个配置指定 (alpha, beta, gamma)
    EXPERIMENT_CONFIGS = {
        "Dense-only": (1.0, 0.0, 0.0),
        "Sparse-only": (0.0, 1.0, 0.0),
        "KG-only": (0.0, 0.0, 1.0),
        "Dense+Sparse": (0.6, 0.4, 0.0),
        "Dense+KG": (0.6, 0.0, 0.4),
        "Sparse+KG": (0.0, 0.6, 0.4),
        "Full (Ours)": (0.4, 0.3, 0.3),  # 论文提出的完整系统
    }
    
    def __init__(
        self,
        dense_index_dir: Path,
        sparse_index_dir: Path,
        queries_file: Path,
        qrels_file: Path,
        output_dir: Path,
        top_k: int = 100,
        neo4j_uri: str = None,
        neo4j_user: str = None,
        neo4j_password: str = None
    ):
        """
        初始化消融实验
        
        Args:
            dense_index_dir: Dense索引目录
            sparse_index_dir: Sparse索引目录
            queries_file: 查询文件 (JSONL)
            qrels_file: 相关性标注 (TSV)
            output_dir: 输出目录
            top_k: 返回结果数
            neo4j_uri: Neo4j连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
        """
        logger.info("初始化消融实验...")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        
        # 加载查询和标注
        self.queries = self._load_queries(queries_file)
        self.qrels = self._load_qrels(qrels_file)
        logger.info(f"加载 {len(self.queries)} 个查询, {len(self.qrels)} 条标注")
        
        # 初始化检索器
        logger.info("初始化检索模块...")
        self.dense_searcher = DenseSearcher(index_dir=dense_index_dir)
        self.sparse_searcher = SparseSearcher(index_dir=sparse_index_dir)
        
        # KG模块
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
        
        logger.info("消融实验初始化完成!")
    
    def _load_queries(self, queries_file: Path) -> Dict[str, Dict]:
        """加载查询"""
        queries = {}
        with open(queries_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    q = json.loads(line)
                    queries[q["qid"]] = q
        return queries
    
    def _load_qrels(self, qrels_file: Path) -> Dict[str, Dict[str, int]]:
        """加载相关性标注"""
        qrels = defaultdict(dict)
        with open(qrels_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        qid, doc_id, rel = parts[0], parts[1], int(parts[2])
                        qrels[qid][doc_id] = rel
        return dict(qrels)
    
    def _retrieve_single_query(
        self,
        query_text: str,
        lang: str,
        alpha: float,
        beta: float,
        gamma: float
    ) -> List[Dict]:
        """
        执行单个查询的检索
        
        Args:
            query_text: 查询文本
            lang: 查询语言
            alpha: Dense权重
            beta: Sparse权重
            gamma: KG权重
        
        Returns:
            检索结果列表
        """
        # Dense检索
        dense_results = []
        if alpha > 0:
            dense_results = self.dense_searcher.search(
                query=query_text,
                top_k=self.top_k
            )
        
        # Sparse检索
        sparse_results = []
        if beta > 0:
            sparse_results = self.sparse_searcher.search(
                query=query_text,
                lang=lang if lang != "auto" else None,
                top_k=self.top_k
            )
        
        # KG增强
        kg_results = []
        if gamma > 0:
            try:
                # 实体链接
                linked_entities = self.entity_linker.link_entities(query_text, lang=lang)
                
                if linked_entities:
                    entity_ids = [e["entity_id"] for e in linked_entities]
                    
                    # 图谱扩展
                    expanded_nodes = self.hop_expander.expand_multi_entities(
                        entity_ids=entity_ids,
                        max_hops=2
                    )
                    
                    # 路径评分
                    kg_results = self.kg_scorer.score_documents(
                        query_entities=entity_ids,
                        expanded_graph=expanded_nodes,
                        corpus_map=self.dense_searcher.corpus_map
                    )
            except Exception as e:
                logger.warning(f"KG增强失败: {e}")
        
        # 融合排序
        reranker = FusionReranker(alpha=alpha, beta=beta, gamma=gamma)
        final_results = reranker.rerank(
            dense_results=dense_results,
            sparse_results=sparse_results,
            kg_results=kg_results,
            top_k=self.top_k,
            method="weighted_sum"
        )
        
        return final_results
    
    def run_single_config(
        self,
        config_name: str,
        alpha: float,
        beta: float,
        gamma: float
    ) -> Dict:
        """
        运行单个配置的实验
        
        Args:
            config_name: 配置名称
            alpha: Dense权重
            beta: Sparse权重
            gamma: KG权重
        
        Returns:
            实验结果
        """
        logger.info(f"运行实验: {config_name} (α={alpha}, β={beta}, γ={gamma})")
        start_time = time.time()
        
        # 存储所有查询的检索结果
        all_results = {}
        
        # 逐个查询检索
        for i, (qid, query_info) in enumerate(self.queries.items(), 1):
            query_text = query_info.get("query", "")
            lang = query_info.get("lang", "auto")
            
            if i % 10 == 0:
                logger.info(f"  处理进度: {i}/{len(self.queries)}")
            
            try:
                results = self._retrieve_single_query(
                    query_text=query_text,
                    lang=lang,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )
                all_results[qid] = results
            except Exception as e:
                logger.error(f"查询 {qid} 失败: {e}")
                all_results[qid] = []
        
        # 计算评测指标
        metrics = self._compute_metrics(all_results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"  {config_name} 完成 (耗时: {elapsed_time:.1f}s)")
        logger.info(f"  nDCG@10={metrics['ndcg@10']:.4f}, MRR={metrics['mrr']:.4f}, Recall@50={metrics['recall@50']:.4f}")
        
        return {
            "config_name": config_name,
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "metrics": metrics,
            "elapsed_time": elapsed_time,
            "num_queries": len(all_results)
        }
    
    def _compute_metrics(self, all_results: Dict[str, List[Dict]]) -> Dict[str, float]:
        """
        计算评测指标
        
        Args:
            all_results: {qid: [检索结果]}
        
        Returns:
            评测指标
        """
        ndcg_scores = []
        mrr_scores = []
        recall_scores = []
        
        for qid, results in all_results.items():
            if qid not in self.qrels:
                continue
            
            qrel = self.qrels[qid]
            
            # 提取doc_id和得分
            ranked_docs = [(r["doc_id"], r.get("final_score", 0.0)) for r in results]
            
            # nDCG@10
            ndcg = compute_ndcg(ranked_docs[:10], qrel)
            ndcg_scores.append(ndcg)
            
            # MRR
            mrr = compute_mrr(ranked_docs, qrel)
            mrr_scores.append(mrr)
            
            # Recall@50
            recall = compute_recall(ranked_docs[:50], qrel)
            recall_scores.append(recall)
        
        return {
            "ndcg@10": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
            "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
            "recall@50": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
            "num_evaluated": len(ndcg_scores)
        }
    
    def run_all_experiments(self) -> List[Dict]:
        """
        运行所有消融实验
        
        Returns:
            所有实验结果
        """
        logger.info(f"开始消融实验: {len(self.EXPERIMENT_CONFIGS)} 个配置")
        
        results = []
        for config_name, (alpha, beta, gamma) in self.EXPERIMENT_CONFIGS.items():
            result = self.run_single_config(config_name, alpha, beta, gamma)
            results.append(result)
        
        logger.info("所有实验完成!")
        return results
    
    def save_results(self, results: List[Dict]):
        """保存实验结果"""
        # JSON格式
        json_file = self.output_dir / "ablation_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存: {json_file}")
        
        # 生成LaTeX表格
        latex_file = self.output_dir / "ablation_table.tex"
        self._generate_latex_table(results, latex_file)
        logger.info(f"LaTeX表格已生成: {latex_file}")
        
        # 生成Markdown表格
        md_file = self.output_dir / "ablation_results.md"
        self._generate_markdown_table(results, md_file)
        logger.info(f"Markdown表格已生成: {md_file}")
    
    def _generate_latex_table(self, results: List[Dict], output_file: Path):
        """生成LaTeX表格"""
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{消融实验结果对比 (Ablation Study Results)}",
            "\\label{tab:ablation}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "\\textbf{Configuration} & \\textbf{Dense} & \\textbf{Sparse} & \\textbf{KG} & \\textbf{nDCG@10} & \\textbf{MRR} & \\textbf{Recall@50} \\\\",
            "\\midrule"
        ]
        
        for r in results:
            config = r["config_name"]
            weights = r["weights"]
            metrics = r["metrics"]
            
            # 标记最佳结果
            alpha_str = f"${weights['alpha']:.1f}$" if weights["alpha"] > 0 else "-"
            beta_str = f"${weights['beta']:.1f}$" if weights["beta"] > 0 else "-"
            gamma_str = f"${weights['gamma']:.1f}$" if weights["gamma"] > 0 else "-"
            
            line = f"{config} & {alpha_str} & {beta_str} & {gamma_str} & "
            line += f"{metrics['ndcg@10']:.4f} & {metrics['mrr']:.4f} & {metrics['recall@50']:.4f} \\\\"
            lines.append(line)
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def _generate_markdown_table(self, results: List[Dict], output_file: Path):
        """生成Markdown表格"""
        lines = [
            "# 消融实验结果 (Ablation Study Results)",
            "",
            "| Configuration | Dense (α) | Sparse (β) | KG (γ) | nDCG@10 | MRR | Recall@50 |",
            "|---------------|-----------|------------|--------|---------|-----|-----------|"
        ]
        
        # 找出最佳结果
        best_ndcg = max(r["metrics"]["ndcg@10"] for r in results)
        best_mrr = max(r["metrics"]["mrr"] for r in results)
        best_recall = max(r["metrics"]["recall@50"] for r in results)
        
        for r in results:
            config = r["config_name"]
            weights = r["weights"]
            metrics = r["metrics"]
            
            alpha_str = f"{weights['alpha']:.1f}" if weights["alpha"] > 0 else "-"
            beta_str = f"{weights['beta']:.1f}" if weights["beta"] > 0 else "-"
            gamma_str = f"{weights['gamma']:.1f}" if weights["gamma"] > 0 else "-"
            
            # 标记最佳结果
            ndcg_str = f"**{metrics['ndcg@10']:.4f}**" if metrics["ndcg@10"] == best_ndcg else f"{metrics['ndcg@10']:.4f}"
            mrr_str = f"**{metrics['mrr']:.4f}**" if metrics["mrr"] == best_mrr else f"{metrics['mrr']:.4f}"
            recall_str = f"**{metrics['recall@50']:.4f}**" if metrics["recall@50"] == best_recall else f"{metrics['recall@50']:.4f}"
            
            line = f"| {config} | {alpha_str} | {beta_str} | {gamma_str} | {ndcg_str} | {mrr_str} | {recall_str} |"
            lines.append(line)
        
        lines.extend([
            "",
            "## 分析",
            "",
            f"- **最佳nDCG@10**: {best_ndcg:.4f}",
            f"- **最佳MRR**: {best_mrr:.4f}",
            f"- **最佳Recall@50**: {best_recall:.4f}",
            "",
            "### 关键发现",
            "",
            "1. **组件贡献**: 比较单独组件 vs 组合配置",
            "2. **互补性**: Dense+Sparse vs 单独使用",
            "3. **KG增益**: 加入KG后的性能提升",
            "4. **最优配置**: Full system (Ours) 的综合表现"
        ])
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def close(self):
        """关闭所有连接"""
        if self.dense_searcher:
            self.dense_searcher.close()
        if self.sparse_searcher:
            self.sparse_searcher.close()
        if self.entity_linker:
            self.entity_linker.close()
        if self.hop_expander:
            self.hop_expander.close()
        logger.info("资源已释放")


def main():
    parser = argparse.ArgumentParser(
        description="消融实验 - 系统化评测各组件贡献",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=Path("artifacts/faiss_labse"),
        help="Dense索引目录"
    )
    parser.add_argument(
        "--sparse-index",
        type=Path,
        default=Path("artifacts/whoosh_bm25"),
        help="Sparse索引目录"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/eval/clir_queries.jsonl"),
        help="查询文件 (JSONL)"
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=Path("data/eval/qrels.tsv"),
        help="相关性标注文件 (TSV)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/ablation_results"),
        help="输出目录"
    )
    
    # 可选参数
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="返回结果数"
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        help="Neo4j URI (默认从config读取)"
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        help="Neo4j用户名 (默认从config读取)"
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        help="Neo4j密码 (默认从config读取)"
    )
    
    args = parser.parse_args()
    
    # 初始化实验
    try:
        experiment = AblationExperiment(
            dense_index_dir=args.dense_index,
            sparse_index_dir=args.sparse_index,
            queries_file=args.queries,
            qrels_file=args.qrels,
            output_dir=args.output_dir,
            top_k=args.top_k,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password
        )
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return 1
    
    try:
        # 运行所有实验
        results = experiment.run_all_experiments()
        
        # 保存结果
        experiment.save_results(results)
        
        # 打印摘要
        print("\n" + "="*80)
        print("消融实验完成!")
        print("="*80)
        print(f"\n实验配置数: {len(results)}")
        print(f"查询数: {results[0]['num_queries']}")
        print(f"\n结果文件:")
        print(f"  - {args.output_dir / 'ablation_results.json'}")
        print(f"  - {args.output_dir / 'ablation_table.tex'}")
        print(f"  - {args.output_dir / 'ablation_results.md'}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("用户中断")
        return 130
    except Exception as e:
        logger.error(f"实验失败: {e}", exc_info=True)
        return 1
    finally:
        experiment.close()


if __name__ == "__main__":
    sys.exit(main())
