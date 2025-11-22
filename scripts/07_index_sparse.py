"""
一键构建Whoosh索引

自动发现清洗后的语料并构建BM25索引
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.sparse.build_whoosh import WhooshIndexBuilder
from logger import get_logger
from config import config

logger = get_logger(__name__)


def discover_corpus_files(data_dir: Path) -> list:
    """
    自动发现清洗后的语料文件
    
    Args:
        data_dir: 数据目录
        
    Returns:
        语料文件列表
    """
    data_dir = Path(data_dir)
    
    # 查找所有清洗后的语料
    patterns = [
        "corpus_*_cleaned.jsonl",  # 标准格式
        "corpus_*.jsonl"            # 备用格式
    ]
    
    corpus_files = []
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        corpus_files.extend(files)
    
    # 去重
    corpus_files = list(set(corpus_files))
    corpus_files.sort()
    
    return corpus_files


def build_sparse_index(
    corpus_files: list,
    output_dir: Path,
    force_new: bool = True
):
    """
    构建Whoosh索引
    
    Args:
        corpus_files: 语料文件列表
        output_dir: 索引输出目录
        force_new: 是否强制创建新索引
    """
    logger.info("=" * 60)
    logger.info("开始构建Whoosh稀疏索引")
    logger.info("=" * 60)
    
    # 检查语料文件
    if not corpus_files:
        logger.error("未找到语料文件,请先运行 01_clean_corpus.py")
        return
    
    logger.info(f"发现 {len(corpus_files)} 个语料文件:")
    for f in corpus_files:
        logger.info(f"  - {f}")
    
    # 创建索引构建器
    builder = WhooshIndexBuilder(index_dir=output_dir)
    
    # 构建索引
    logger.info(f"\n索引输出目录: {output_dir}")
    builder.build_from_files(corpus_files, force_new=force_new)
    
    # 显示统计
    stats = builder.get_statistics()
    logger.info("\n索引统计:")
    logger.info(f"  文档数: {stats['num_docs']}")
    logger.info(f"  字段: {', '.join(stats['fields'])}")
    logger.info(f"  索引目录: {stats['index_dir']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Whoosh索引构建完成 ✓")
    logger.info("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="一键构建Whoosh稀疏索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python scripts/07_index_sparse.py
  
  # 指定数据目录和输出目录
  python scripts/07_index_sparse.py --data-dir ./data --output-dir ./whoosh_index
  
  # 增量更新索引(不删除现有索引)
  python scripts/07_index_sparse.py --no-force-new
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"数据目录 (默认: {config.DATA_DIR})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"索引输出目录 (默认: {config.WHOOSH_INDEX_DIR})"
    )
    
    parser.add_argument(
        "--corpus-files",
        type=str,
        nargs="+",
        help="指定语料文件(可选,默认自动发现)"
    )
    
    parser.add_argument(
        "--no-force-new",
        action="store_true",
        help="不强制创建新索引(增量更新)"
    )
    
    args = parser.parse_args()
    
    # 确定数据目录和输出目录
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else config.WHOOSH_INDEX_DIR
    
    # 确定语料文件
    if args.corpus_files:
        corpus_files = [Path(f) for f in args.corpus_files]
    else:
        corpus_files = discover_corpus_files(data_dir)
    
    # 构建索引
    build_sparse_index(
        corpus_files=corpus_files,
        output_dir=output_dir,
        force_new=not args.no_force_new
    )
    
    logger.info("\n下一步:")
    logger.info(f"  运行检索测试:")
    logger.info(f"    python retrieval/sparse/sparse_search.py \\")
    logger.info(f"      --index {output_dir} \\")
    logger.info(f"      --interactive")


if __name__ == "__main__":
    main()
