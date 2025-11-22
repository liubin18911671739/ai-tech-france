"""
构建Whoosh索引 - Sparse检索核心

使用Whoosh实现BM25检索
支持多语言文本索引
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from whoosh import index
from whoosh.fields import Schema, TEXT, ID, KEYWORD
from whoosh.analysis import StemmingAnalyzer, SimpleAnalyzer
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

from logger import get_logger
from config import config

logger = get_logger(__name__)


class WhooshIndexBuilder:
    """Whoosh索引构建器"""
    
    def __init__(self, index_dir: Path = None):
        """
        初始化Whoosh索引构建器
        
        Args:
            index_dir: 索引目录
        """
        self.index_dir = index_dir or config.WHOOSH_INDEX_DIR
        self.index_dir = Path(self.index_dir)
        
        self.schema = self._create_schema()
        self.idx = None
        
        logger.info(f"Whoosh索引构建器初始化: {self.index_dir}")
    
    def _create_schema(self) -> Schema:
        """
        创建索引Schema
        
        Returns:
            Whoosh Schema对象
        """
        # 使用简单分析器,适配多语言
        # 对于更好的效果,可以为不同语言使用不同的analyzer
        
        schema = Schema(
            doc_id=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=SimpleAnalyzer()),
            content=TEXT(stored=True, analyzer=SimpleAnalyzer()),
            lang=ID(stored=True),
            concepts=KEYWORD(stored=True, commas=True, scorable=True)
        )
        
        logger.info("Schema创建完成")
        return schema
    
    def create_index(self, force_new: bool = False):
        """
        创建索引
        
        Args:
            force_new: 是否强制创建新索引
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if force_new and index.exists_in(str(self.index_dir)):
            logger.warning("删除现有索引...")
            import shutil
            shutil.rmtree(str(self.index_dir))
            self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if index.exists_in(str(self.index_dir)):
            logger.info("打开现有索引")
            self.idx = index.open_dir(str(self.index_dir))
        else:
            logger.info("创建新索引")
            self.idx = index.create_in(str(self.index_dir), self.schema)
    
    def add_documents(
        self,
        corpus: List[Dict],
        batch_size: int = 100
    ):
        """
        添加文档到索引
        
        Args:
            corpus: 文档列表 [{"doc_id": ..., "title": ..., "content": ...}, ...]
            batch_size: 批处理大小
        """
        if not self.idx:
            raise ValueError("索引未创建,请先调用create_index()")
        
        logger.info(f"开始添加文档: {len(corpus)} 篇")
        
        writer = self.idx.writer()
        
        for i, doc in enumerate(corpus, 1):
            try:
                # 准备文档字段
                doc_id = doc.get("doc_id", f"doc_{i}")
                title = doc.get("title", "")
                content = doc.get("content", "")
                lang = doc.get("lang", "unknown")
                concepts = doc.get("concepts", [])
                
                # 概念转为逗号分隔字符串
                if isinstance(concepts, list):
                    concepts_str = ",".join(concepts)
                else:
                    concepts_str = str(concepts)
                
                # 添加文档
                writer.add_document(
                    doc_id=doc_id,
                    title=title,
                    content=content,
                    lang=lang,
                    concepts=concepts_str
                )
                
                # 批量提交
                if i % batch_size == 0:
                    writer.commit()
                    logger.info(f"已添加 {i}/{len(corpus)} 篇文档")
                    writer = self.idx.writer()
            
            except Exception as e:
                logger.error(f"添加文档失败 (doc {i}): {e}")
                continue
        
        # 最终提交
        writer.commit()
        logger.info(f"文档添加完成: 总计 {len(corpus)} 篇")
    
    def build_from_files(
        self,
        corpus_files: List[Path],
        force_new: bool = True
    ):
        """
        从文件构建索引
        
        Args:
            corpus_files: 语料文件列表
            force_new: 是否强制创建新索引
        """
        logger.info(f"从 {len(corpus_files)} 个文件构建索引")
        
        # 创建索引
        self.create_index(force_new=force_new)
        
        # 加载所有文档
        all_corpus = []
        for corpus_file in corpus_files:
            logger.info(f"加载: {corpus_file}")
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line.strip())
                        all_corpus.append(doc)
                    except Exception as e:
                        logger.error(f"解析文档失败: {e}")
                        continue
        
        logger.info(f"总文档数: {len(all_corpus)}")
        
        # 添加到索引
        self.add_documents(all_corpus)
        
        # 保存统计信息
        self._save_metadata(len(all_corpus), corpus_files)
    
    def _save_metadata(self, num_docs: int, corpus_files: List[Path]):
        """保存索引元数据"""
        metadata = {
            "num_docs": num_docs,
            "corpus_files": [str(f) for f in corpus_files],
            "schema_fields": list(self.schema.names()),
            "scoring": "BM25F"
        }
        
        meta_path = self.index_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"元数据已保存: {meta_path}")
    
    def get_statistics(self) -> Dict:
        """获取索引统计信息"""
        if not self.idx:
            return {}
        
        stats = {
            "index_dir": str(self.index_dir),
            "num_docs": self.idx.doc_count_all(),
            "fields": list(self.schema.names()),
            "latest_generation": self.idx.latest_generation()
        }
        
        return stats
    
    @classmethod
    def open_index(cls, index_dir: Path) -> 'WhooshIndexBuilder':
        """
        打开现有索引
        
        Args:
            index_dir: 索引目录
            
        Returns:
            WhooshIndexBuilder实例
        """
        builder = cls(index_dir=index_dir)
        
        if not index.exists_in(str(index_dir)):
            raise ValueError(f"索引不存在: {index_dir}")
        
        builder.idx = index.open_dir(str(index_dir))
        logger.info(f"索引已打开: {builder.idx.doc_count_all()} 篇文档")
        
        return builder


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建Whoosh索引")
    parser.add_argument("--corpus", type=str, nargs="+", required=True,
                       help="语料文件路径(可多个)")
    parser.add_argument("--output", type=str, required=True,
                       help="索引输出目录")
    parser.add_argument("--force-new", action="store_true",
                       help="强制创建新索引")
    parser.add_argument("--test", action="store_true",
                       help="构建后测试索引")
    
    args = parser.parse_args()
    
    # 构建索引
    corpus_files = [Path(f) for f in args.corpus]
    output_dir = Path(args.output)
    
    builder = WhooshIndexBuilder(index_dir=output_dir)
    builder.build_from_files(corpus_files, force_new=args.force_new)
    
    # 显示统计
    stats = builder.get_statistics()
    logger.info("索引统计:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # 测试索引
    if args.test:
        logger.info("\n测试索引...")
        
        # 打开索引进行搜索
        with builder.idx.searcher(weighting=BM25F()) as searcher:
            # 测试查询
            test_queries = ["语法", "grammaire", "verb"]
            
            parser = MultifieldParser(["title", "content"], schema=builder.schema)
            
            for q in test_queries:
                query = parser.parse(q)
                results = searcher.search(query, limit=3)
                
                logger.info(f"\n查询: {q}")
                logger.info(f"结果数: {len(results)}")
                for i, hit in enumerate(results, 1):
                    logger.info(f"  {i}. {hit['doc_id']} (Score: {hit.score:.3f})")


if __name__ == "__main__":
    main()
