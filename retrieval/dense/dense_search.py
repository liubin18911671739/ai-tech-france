"""
Dense检索 - 基于FAISS的向量检索

使用预构建的FAISS索引进行快速向量相似度检索
支持跨语言查询
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config
from retrieval.dense.labse_encoder import LaBSEEncoder
from retrieval.dense.build_faiss import FAISSIndexBuilder

logger = get_logger(__name__)


class DenseSearcher:
    """Dense检索器"""
    
    def __init__(
        self,
        index_dir: Path = None,
        encoder: LaBSEEncoder = None
    ):
        """
        初始化Dense检索器
        
        Args:
            index_dir: FAISS索引目录
            encoder: LaBSE编码器
        """
        self.encoder = encoder or LaBSEEncoder()
        self.index_builder = None
        self.corpus_map = {}  # doc_id -> 文档内容
        
        if index_dir:
            self.load_index(index_dir)
        
        logger.info("Dense检索器初始化完成")
    
    def load_index(self, index_dir: Path):
        """
        加载FAISS索引
        
        Args:
            index_dir: 索引目录
        """
        logger.info(f"加载FAISS索引: {index_dir}")
        self.index_builder = FAISSIndexBuilder.load(index_dir)
        logger.info(f"索引加载完成: {self.index_builder.index.ntotal} 个向量")
    
    def load_corpus(self, corpus_files: List[Path]):
        """
        加载语料库(用于返回完整文档信息)
        
        Args:
            corpus_files: 语料文件列表
        """
        logger.info("加载语料库...")
        for corpus_file in corpus_files:
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    doc_id = doc.get("doc_id")
                    if doc_id:
                        self.corpus_map[doc_id] = doc
        
        logger.info(f"语料库加载完成: {len(self.corpus_map)} 篇文档")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        检索
        
        Args:
            query: 查询字符串
            top_k: 返回top-k结果
            return_scores: 是否返回分数
            
        Returns:
            检索结果列表 [{"doc_id": ..., "score": ..., "rank": ...}, ...]
        """
        if not self.index_builder or not self.index_builder.index:
            raise ValueError("索引未加载,请先调用load_index()")
        
        # 编码查询
        logger.debug(f"编码查询: {query}")
        query_emb = self.encoder.encode(query)
        
        # 搜索
        scores, indices = self.index_builder.index.search(
            query_emb.reshape(1, -1),
            top_k
        )
        
        # 构建结果
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:  # FAISS返回-1表示无效
                continue
            
            doc_id = self.index_builder.doc_ids[idx]
            
            result = {
                "doc_id": doc_id,
                "rank": rank,
                "score": float(score) if return_scores else None
            }
            
            # 添加文档内容(如果已加载)
            if doc_id in self.corpus_map:
                doc = self.corpus_map[doc_id]
                result["title"] = doc.get("title", "")
                result["content"] = doc.get("content", "")
                result["lang"] = doc.get("lang", "")
            
            results.append(result)
        
        logger.debug(f"检索完成: 返回 {len(results)} 个结果")
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Dict]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: 每个查询返回top-k
            
        Returns:
            结果列表的列表
        """
        logger.info(f"批量检索: {len(queries)} 个查询")
        
        # 批量编码
        query_embs = self.encoder.encode(queries)
        
        # 批量搜索
        scores, indices = self.index_builder.index.search(query_embs, top_k)
        
        # 构建结果
        all_results = []
        for query_idx in range(len(queries)):
            results = []
            for rank, (idx, score) in enumerate(
                zip(indices[query_idx], scores[query_idx]), 1
            ):
                if idx == -1:
                    continue
                
                doc_id = self.index_builder.doc_ids[idx]
                
                result = {
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": float(score)
                }
                
                if doc_id in self.corpus_map:
                    doc = self.corpus_map[doc_id]
                    result["title"] = doc.get("title", "")
                    result["lang"] = doc.get("lang", "")
                
                results.append(result)
            
            all_results.append(results)
        
        logger.info(f"批量检索完成")
        
        return all_results
    
    def get_statistics(self) -> Dict:
        """获取索引统计信息"""
        if not self.index_builder:
            return {}
        
        stats = {
            "index_type": self.index_builder.metadata.get("index_type"),
            "num_docs": self.index_builder.index.ntotal,
            "dimension": self.index_builder.metadata.get("dimension"),
            "model": self.index_builder.metadata.get("model"),
            "corpus_loaded": len(self.corpus_map)
        }
        
        return stats


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dense检索")
    parser.add_argument("--index", type=str, required=True,
                       help="FAISS索引目录")
    parser.add_argument("--corpus", type=str, nargs="+",
                       help="语料文件(可选,用于返回完整文档)")
    parser.add_argument("--query", type=str,
                       help="单个查询")
    parser.add_argument("--queries-file", type=str,
                       help="查询文件(每行一个查询)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="返回top-k结果")
    parser.add_argument("--output", type=str,
                       help="输出结果文件")
    
    args = parser.parse_args()
    
    # 初始化检索器
    searcher = DenseSearcher(index_dir=Path(args.index))
    
    # 加载语料
    if args.corpus:
        corpus_files = [Path(f) for f in args.corpus]
        searcher.load_corpus(corpus_files)
    
    # 显示统计
    stats = searcher.get_statistics()
    logger.info(f"索引统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 单查询
    if args.query:
        logger.info(f"查询: {args.query}")
        results = searcher.search(args.query, top_k=args.top_k)
        
        print(f"\n检索结果 (Top-{args.top_k}):")
        print("=" * 80)
        for result in results:
            print(f"\nRank {result['rank']}: {result['doc_id']} (Score: {result['score']:.4f})")
            if "title" in result:
                print(f"  标题: {result['title']}")
                print(f"  语言: {result.get('lang', 'N/A')}")
                print(f"  内容: {result.get('content', '')[:100]}...")
        
        # 保存结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存: {output_path}")
    
    # 批量查询
    elif args.queries_file:
        queries_file = Path(args.queries_file)
        logger.info(f"加载查询文件: {queries_file}")
        
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 如果是JSON格式
                    try:
                        query_obj = json.loads(line)
                        queries.append(query_obj.get("query", line))
                    except:
                        queries.append(line)
        
        logger.info(f"加载 {len(queries)} 个查询")
        
        # 批量检索
        all_results = searcher.batch_search(queries, top_k=args.top_k)
        
        # 显示结果
        for i, (query, results) in enumerate(zip(queries, all_results), 1):
            print(f"\n查询 {i}: {query}")
            print("-" * 80)
            for result in results[:3]:  # 只显示前3个
                print(f"  {result['rank']}. {result['doc_id']} ({result['score']:.4f})")
        
        # 保存结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for query, results in zip(queries, all_results):
                    f.write(json.dumps({
                        "query": query,
                        "results": results
                    }, ensure_ascii=False) + '\n')
            logger.info(f"结果已保存: {output_path}")
    
    else:
        # 交互模式
        logger.info("进入交互模式 (输入'quit'退出)")
        while True:
            query = input("\n请输入查询: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            results = searcher.search(query, top_k=args.top_k)
            
            print(f"\n检索结果 (Top-{len(results)}):")
            for result in results:
                print(f"  {result['rank']}. {result['doc_id']} ({result['score']:.4f})")
                if "title" in result:
                    print(f"     {result['title']} [{result.get('lang', 'N/A')}]")


if __name__ == "__main__":
    main()
