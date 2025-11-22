"""
Whoosh BM25检索器

基于Whoosh实现稀疏检索
支持单个查询和批量查询
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from whoosh import index
from whoosh.qparser import MultifieldParser, QueryParser, OrGroup
from whoosh.scoring import BM25F
from whoosh.query import Term, Or

from logger import get_logger
from config import config

logger = get_logger(__name__)


class SparseSearcher:
    """Whoosh稀疏检索器"""
    
    def __init__(
        self,
        index_dir: Path = None,
        k1: float = 1.2,
        b: float = 0.75
    ):
        """
        初始化检索器
        
        Args:
            index_dir: 索引目录
            k1: BM25 k1参数 (term frequency饱和参数)
            b: BM25 b参数 (文档长度归一化)
        """
        self.index_dir = index_dir or config.WHOOSH_INDEX_DIR
        self.index_dir = Path(self.index_dir)
        
        self.k1 = k1
        self.b = b
        
        self.idx = None
        self.searcher = None
        self.schema = None
        
        self._load_index()
    
    def _load_index(self):
        """加载索引"""
        if not index.exists_in(str(self.index_dir)):
            raise ValueError(f"索引不存在: {self.index_dir}")
        
        self.idx = index.open_dir(str(self.index_dir))
        self.schema = self.idx.schema
        
        # 创建searcher with BM25F scoring
        self.searcher = self.idx.searcher(weighting=BM25F(K1=self.k1, B=self.b))
        
        logger.info(f"索引已加载: {self.idx.doc_count_all()} 篇文档")
        logger.info(f"BM25参数: K1={self.k1}, B={self.b}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        fields: List[str] = None,
        lang_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        检索单个查询
        
        Args:
            query: 查询文本
            top_k: 返回Top-K结果
            fields: 查询字段列表 (默认["title", "content"])
            lang_filter: 语言过滤 (可选,如"fr","zh","en")
            
        Returns:
            结果列表 [{"doc_id": ..., "score": ..., "title": ..., "content": ...}, ...]
        """
        if not query.strip():
            logger.warning("空查询,返回空结果")
            return []
        
        # 默认查询字段
        if fields is None:
            fields = ["title", "content"]
        
        try:
            # 创建查询解析器
            # 使用OrGroup允许部分匹配
            parser = MultifieldParser(fields, schema=self.schema, group=OrGroup)
            parsed_query = parser.parse(query)
            
            # 添加语言过滤
            if lang_filter:
                from whoosh.query import And, Term
                lang_term = Term("lang", lang_filter)
                parsed_query = And([parsed_query, lang_term])
            
            # 执行搜索
            results = self.searcher.search(parsed_query, limit=top_k)
            
            # 格式化结果
            formatted_results = []
            for i, hit in enumerate(results):
                result = {
                    "rank": i + 1,
                    "doc_id": hit.get("doc_id", ""),
                    "score": float(hit.score),
                    "title": hit.get("title", ""),
                    "content": hit.get("content", ""),
                    "lang": hit.get("lang", ""),
                    "concepts": hit.get("concepts", "").split(",") if hit.get("concepts") else []
                }
                formatted_results.append(result)
            
            logger.info(f"查询: '{query}' | 结果数: {len(formatted_results)}")
            return formatted_results
        
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def batch_search(
        self,
        queries: List[Union[str, Dict]],
        top_k: int = 10,
        fields: List[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        批量检索
        
        Args:
            queries: 查询列表,可以是:
                     - List[str]: ["query1", "query2", ...]
                     - List[Dict]: [{"qid": "q1", "query": "..."}, ...]
            top_k: 每个查询返回Top-K结果
            fields: 查询字段列表
            
        Returns:
            结果字典 {query_id: [results], ...}
        """
        logger.info(f"批量检索: {len(queries)} 个查询")
        
        results_dict = {}
        
        for i, q in enumerate(queries, 1):
            # 处理不同格式的查询
            if isinstance(q, str):
                qid = f"q{i}"
                query_text = q
                lang_filter = None
            elif isinstance(q, dict):
                qid = q.get("qid", f"q{i}")
                query_text = q.get("query", "")
                lang_filter = q.get("lang", None)
            else:
                logger.warning(f"不支持的查询格式: {type(q)}")
                continue
            
            # 执行搜索
            results = self.search(
                query=query_text,
                top_k=top_k,
                fields=fields,
                lang_filter=lang_filter
            )
            
            results_dict[qid] = results
        
        logger.info(f"批量检索完成: {len(results_dict)} 个查询")
        return results_dict
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        根据doc_id获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            文档字典或None
        """
        try:
            # 使用doc_id字段精确查询
            parser = QueryParser("doc_id", schema=self.schema)
            query = parser.parse(doc_id)
            
            results = self.searcher.search(query, limit=1)
            
            if len(results) == 0:
                return None
            
            hit = results[0]
            doc = {
                "doc_id": hit.get("doc_id", ""),
                "title": hit.get("title", ""),
                "content": hit.get("content", ""),
                "lang": hit.get("lang", ""),
                "concepts": hit.get("concepts", "").split(",") if hit.get("concepts") else []
            }
            
            return doc
        
        except Exception as e:
            logger.error(f"获取文档失败 ({doc_id}): {e}")
            return None
    
    def export_results(
        self,
        results_dict: Dict[str, List[Dict]],
        output_path: Path,
        format: str = "jsonl"
    ):
        """
        导出检索结果
        
        Args:
            results_dict: 批量检索结果
            output_path: 输出路径
            format: 输出格式 ("jsonl" 或 "trec")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for qid, results in results_dict.items():
                    for result in results:
                        obj = {
                            "qid": qid,
                            "doc_id": result["doc_id"],
                            "rank": result["rank"],
                            "score": result["score"]
                        }
                        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            
            logger.info(f"结果已导出(JSONL): {output_path}")
        
        elif format == "trec":
            with open(output_path, 'w', encoding='utf-8') as f:
                for qid, results in results_dict.items():
                    for result in results:
                        # TREC格式: qid Q0 doc_id rank score run_name
                        line = f"{qid} Q0 {result['doc_id']} {result['rank']} {result['score']:.6f} BM25\n"
                        f.write(line)
            
            logger.info(f"结果已导出(TREC): {output_path}")
        
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def get_statistics(self) -> Dict:
        """获取索引统计信息"""
        stats = {
            "index_dir": str(self.index_dir),
            "num_docs": self.idx.doc_count_all(),
            "fields": list(self.schema.names()),
            "bm25_k1": self.k1,
            "bm25_b": self.b
        }
        return stats
    
    def close(self):
        """关闭检索器"""
        if self.searcher:
            self.searcher.close()
        logger.info("检索器已关闭")
    
    def __enter__(self):
        """上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.close()


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Whoosh BM25检索")
    parser.add_argument("--index", type=str, required=True,
                       help="索引目录")
    parser.add_argument("--query", type=str,
                       help="单个查询")
    parser.add_argument("--queries", type=str,
                       help="查询文件(JSONL)")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Top-K结果数")
    parser.add_argument("--output", type=str,
                       help="结果输出路径")
    parser.add_argument("--format", type=str, default="jsonl",
                       choices=["jsonl", "trec"],
                       help="输出格式")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式模式")
    
    args = parser.parse_args()
    
    # 创建检索器
    searcher = SparseSearcher(index_dir=Path(args.index))
    
    # 显示统计
    stats = searcher.get_statistics()
    logger.info("索引统计:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # 单个查询
    if args.query:
        logger.info(f"\n查询: {args.query}")
        results = searcher.search(args.query, top_k=args.top_k)
        
        for result in results:
            print(f"{result['rank']}. {result['doc_id']} (Score: {result['score']:.4f})")
            print(f"   Title: {result['title'][:80]}")
            print(f"   Lang: {result['lang']}")
            print()
    
    # 批量查询
    elif args.queries:
        queries_file = Path(args.queries)
        
        # 加载查询
        queries = []
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                q = json.loads(line.strip())
                queries.append(q)
        
        logger.info(f"加载查询: {len(queries)} 个")
        
        # 批量检索
        results_dict = searcher.batch_search(queries, top_k=args.top_k)
        
        # 导出结果
        if args.output:
            searcher.export_results(
                results_dict,
                Path(args.output),
                format=args.format
            )
        else:
            # 显示结果统计
            for qid, results in list(results_dict.items())[:5]:  # 显示前5个
                print(f"\n查询ID: {qid}")
                for result in results[:3]:  # 每个查询显示Top-3
                    print(f"  {result['rank']}. {result['doc_id']} (Score: {result['score']:.4f})")
    
    # 交互式模式
    elif args.interactive:
        logger.info("\n进入交互式检索模式 (输入'quit'退出)")
        
        while True:
            try:
                query = input("\n请输入查询: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                results = searcher.search(query, top_k=args.top_k)
                
                if len(results) == 0:
                    print("无结果")
                    continue
                
                print(f"\n找到 {len(results)} 个结果:")
                for result in results:
                    print(f"\n{result['rank']}. {result['doc_id']} (Score: {result['score']:.4f})")
                    print(f"   Title: {result['title'][:100]}")
                    print(f"   Lang: {result['lang']}")
                    print(f"   Preview: {result['content'][:150]}...")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"错误: {e}")
    
    else:
        parser.print_help()
    
    # 关闭检索器
    searcher.close()


if __name__ == "__main__":
    main()
