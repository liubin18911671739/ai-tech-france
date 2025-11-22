#!/usr/bin/env python3
"""
端到端KG-CLIR检索脚本

整合Dense、Sparse、KG三路检索，完成跨语言信息检索
支持单个查询和批量查询
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config

# 导入检索模块
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
        top_k_per_source: int = 100
    ):
        """
        初始化KG-CLIR系统
        
        Args:
            dense_index_dir: Dense索引目录
            sparse_index_dir: Sparse索引目录
            use_kg: 是否使用KG增强
            alpha: Dense权重
            beta: Sparse权重
            gamma: KG权重
            neo4j_uri: Neo4j连接URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            max_hops: 最大跳数
            top_k_per_source: 每个检索源返回的最大结果数
        """
        logger.info("初始化KG-CLIR系统...")
        
        # 检索器
        self.dense_searcher = DenseSearcher(index_dir=dense_index_dir)
        logger.info("✓ Dense检索器加载完成")
        
        self.sparse_searcher = SparseSearcher(index_dir=sparse_index_dir)
        logger.info("✓ Sparse检索器加载完成")
        
        # KG模块
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
            logger.info("✓ KG增强模块加载完成")
        else:
            self.entity_linker = None
            self.hop_expander = None
            self.kg_scorer = None
            logger.info("○ KG增强已禁用")
        
        # 融合器
        self.reranker = FusionReranker(
            alpha=alpha,
            beta=beta,
            gamma=gamma if use_kg else 0.0
        )
        logger.info(f"✓ 融合器配置: α={self.reranker.alpha:.2f}, β={self.reranker.beta:.2f}, γ={self.reranker.gamma:.2f}")
        
        self.max_hops = max_hops
        self.top_k_per_source = top_k_per_source
        
        logger.info("KG-CLIR系统初始化完成!")
    
    def search(
        self,
        query: str,
        lang: str = "auto",
        top_k: int = 10,
        explain: bool = False
    ) -> List[Dict]:
        """
        执行端到端检索
        
        Args:
            query: 查询文本
            lang: 查询语言 (fr/zh/en/auto)
            top_k: 返回结果数
            explain: 是否返回解释信息
        
        Returns:
            检索结果列表
        """
        start_time = time.time()
        logger.info(f"查询: '{query}' (语言: {lang})")
        
        # Step 1: Dense检索
        logger.info("Step 1/5: Dense检索...")
        dense_results = self.dense_searcher.search(
            query=query,
            top_k=self.top_k_per_source
        )
        logger.info(f"  Dense检索返回 {len(dense_results)} 个结果")
        
        # Step 2: Sparse检索
        logger.info("Step 2/5: Sparse检索...")
        sparse_results = self.sparse_searcher.search(
            query=query,
            lang=lang if lang != "auto" else None,
            top_k=self.top_k_per_source
        )
        logger.info(f"  Sparse检索返回 {len(sparse_results)} 个结果")
        
        # Step 3: KG增强
        kg_results = []
        if self.use_kg and self.entity_linker:
            logger.info("Step 3/5: KG实体链接...")
            linked_entities = self.entity_linker.link_entities(query, lang=lang)
            logger.info(f"  链接到 {len(linked_entities)} 个实体")
            
            if linked_entities:
                logger.info("Step 4/5: KG邻域扩展...")
                entity_ids = [e["entity_id"] for e in linked_entities]
                expanded_nodes = self.hop_expander.expand_multi_entities(
                    entity_ids=entity_ids,
                    max_hops=self.max_hops
                )
                logger.info(f"  扩展到 {len(expanded_nodes)} 个节点")
                
                logger.info("Step 5/5: KG路径评分...")
                kg_results = self.kg_scorer.score_documents(
                    query_entities=entity_ids,
                    expanded_graph=expanded_nodes,
                    corpus_map=self.dense_searcher.corpus_map
                )
                logger.info(f"  KG评分返回 {len(kg_results)} 个结果")
            else:
                logger.info("Step 3-5/5: 未链接到实体，跳过KG增强")
        else:
            logger.info("Step 3-5/5: KG增强已禁用")
        
        # Step 6: 融合排序
        logger.info("融合排序...")
        final_results = self.reranker.rerank(
            dense_results=dense_results,
            sparse_results=sparse_results,
            kg_results=kg_results,
            top_k=top_k,
            method="weighted_sum"
        )
        
        # 添加解释信息
        if explain:
            for result in final_results:
                result["query"] = query
                result["query_lang"] = lang
                result["fusion_config"] = {
                    "alpha": self.reranker.alpha,
                    "beta": self.reranker.beta,
                    "gamma": self.reranker.gamma
                }
        
        elapsed_time = time.time() - start_time
        logger.info(f"检索完成! 返回 {len(final_results)} 个结果 (耗时: {elapsed_time:.2f}s)")
        
        return final_results
    
    def batch_search(
        self,
        queries: List[Dict],
        top_k: int = 10,
        explain: bool = False
    ) -> Dict[str, List[Dict]]:
        """
        批量检索
        
        Args:
            queries: 查询列表 [{"qid": "q1", "query": "...", "lang": "fr"}, ...]
            top_k: 每个查询返回结果数
            explain: 是否返回解释信息
        
        Returns:
            {qid: [结果列表]}
        """
        logger.info(f"批量检索: {len(queries)} 个查询")
        
        results = {}
        for i, q in enumerate(queries, 1):
            qid = q.get("qid", f"q{i}")
            query_text = q.get("query", "")
            lang = q.get("lang", "auto")
            
            logger.info(f"[{i}/{len(queries)}] 处理查询: {qid}")
            
            try:
                search_results = self.search(
                    query=query_text,
                    lang=lang,
                    top_k=top_k,
                    explain=explain
                )
                results[qid] = search_results
            except Exception as e:
                logger.error(f"查询 {qid} 失败: {e}")
                results[qid] = []
        
        logger.info(f"批量检索完成: {len(results)}/{len(queries)} 成功")
        return results
    
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
        logger.info("KG-CLIR系统已关闭")


def print_results(results: List[Dict], top_k: int = 10):
    """打印检索结果"""
    print(f"\n{'='*80}")
    print(f"检索结果 (Top {min(top_k, len(results))})")
    print(f"{'='*80}\n")
    
    for i, result in enumerate(results[:top_k], 1):
        print(f"[{i}] doc_id: {result.get('doc_id', 'N/A')}")
        print(f"    Score: {result.get('final_score', 0.0):.4f}")
        
        if "title" in result:
            print(f"    Title: {result['title'][:100]}")
        
        if "content" in result:
            content = result["content"][:200].replace("\n", " ")
            print(f"    Content: {content}...")
        
        # 融合得分解释
        contrib = result.get("score_contributions", {})
        if contrib:
            print(f"    Contributions: Dense={contrib.get('dense', 0):.4f}, "
                  f"Sparse={contrib.get('sparse', 0):.4f}, "
                  f"KG={contrib.get('kg', 0):.4f}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="端到端KG-CLIR检索",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 检索参数
    parser.add_argument(
        "--query",
        type=str,
        help="查询文本 (单个查询模式)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="auto",
        choices=["fr", "zh", "en", "auto"],
        help="查询语言"
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        help="查询文件 (批量查询模式, JSONL格式)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="返回结果数"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="返回详细解释信息"
    )
    
    # 索引路径
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
    
    # KG配置
    parser.add_argument(
        "--use-kg",
        action="store_true",
        default=True,
        help="启用KG增强"
    )
    parser.add_argument(
        "--no-kg",
        action="store_false",
        dest="use_kg",
        help="禁用KG增强"
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=2,
        help="KG最大跳数"
    )
    
    # 融合权重
    parser.add_argument(
        "--alpha",
        type=float,
        help="Dense权重 (默认从config读取)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        help="Sparse权重 (默认从config读取)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="KG权重 (默认从config读取)"
    )
    
    # Neo4j连接
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
    
    # 输出
    parser.add_argument(
        "--output",
        type=Path,
        help="输出文件 (JSON格式)"
    )
    
    args = parser.parse_args()
    
    # 验证输入
    if not args.query and not args.queries_file:
        parser.error("必须提供 --query 或 --queries-file")
    
    # 初始化系统
    try:
        system = KGCLIRSystem(
            dense_index_dir=args.dense_index,
            sparse_index_dir=args.sparse_index,
            use_kg=args.use_kg,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            max_hops=args.max_hops
        )
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        return 1
    
    try:
        # 单个查询模式
        if args.query:
            logger.info(f"单个查询模式: '{args.query}'")
            results = system.search(
                query=args.query,
                lang=args.lang,
                top_k=args.top_k,
                explain=args.explain
            )
            
            # 打印结果
            print_results(results, top_k=args.top_k)
            
            # 保存结果
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"结果已保存到: {args.output}")
        
        # 批量查询模式
        elif args.queries_file:
            logger.info(f"批量查询模式: {args.queries_file}")
            
            # 读取查询
            queries = []
            with open(args.queries_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        queries.append(json.loads(line))
            
            logger.info(f"加载了 {len(queries)} 个查询")
            
            # 批量检索
            all_results = system.batch_search(
                queries=queries,
                top_k=args.top_k,
                explain=args.explain
            )
            
            # 保存结果
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)
                logger.info(f"批量结果已保存到: {args.output}")
            else:
                # 打印部分结果
                for qid, results in list(all_results.items())[:3]:
                    print(f"\n查询ID: {qid}")
                    print_results(results, top_k=5)
        
        logger.info("检索完成!")
        return 0
    
    except KeyboardInterrupt:
        logger.info("用户中断")
        return 130
    except Exception as e:
        logger.error(f"检索失败: {e}", exc_info=True)
        return 1
    finally:
        system.close()


if __name__ == "__main__":
    sys.exit(main())
