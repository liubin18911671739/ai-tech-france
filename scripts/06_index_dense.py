"""
脚本06: 构建Dense索引

功能:
- 读取清洗后的语料
- 使用LaBSE编码
- 构建FAISS索引
- 保存索引文件
"""
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config
from retrieval.dense.build_faiss import FAISSIndexBuilder

logger = get_logger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建Dense索引(FAISS)")
    parser.add_argument("--corpus-dir", type=str, 
                       default=str(config.CLEANED_DIR),
                       help="清洗后的语料目录")
    parser.add_argument("--corpus-files", type=str, nargs="+",
                       help="指定语料文件(可选,默认使用所有cleaned文件)")
    parser.add_argument("--output", type=str,
                       default=str(config.FAISS_INDEX_DIR),
                       help="索引输出目录")
    parser.add_argument("--index-type", type=str, default="IVF",
                       choices=["Flat", "IVF", "IVFPQ"],
                       help="索引类型")
    parser.add_argument("--nlist", type=int, default=100,
                       help="IVF聚类数(建议为文档数的4倍根号)")
    parser.add_argument("--nprobe", type=int, default=10,
                       help="搜索时探测的聚类数")
    parser.add_argument("--langs", type=str, nargs="+",
                       default=["fr", "zh", "en"],
                       help="要索引的语言")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始构建Dense索引(FAISS)")
    logger.info("=" * 60)
    
    # 确定语料文件
    corpus_dir = Path(args.corpus_dir)
    
    if args.corpus_files:
        corpus_files = [Path(f) for f in args.corpus_files]
    else:
        # 自动查找清洗后的语料
        corpus_files = []
        for lang in args.langs:
            pattern = f"corpus_{lang}_cleaned.jsonl"
            files = list(corpus_dir.glob(pattern))
            if files:
                corpus_files.extend(files)
                logger.info(f"找到 {lang} 语料: {files[0]}")
            else:
                logger.warning(f"未找到 {lang} 语料: {pattern}")
        
        if not corpus_files:
            logger.error(f"未找到任何语料文件! 请检查目录: {corpus_dir}")
            return
    
    logger.info(f"将索引 {len(corpus_files)} 个语料文件")
    for f in corpus_files:
        logger.info(f"  - {f.name}")
    
    # 自动调整nlist
    # 经验法则: nlist ≈ sqrt(N) * 4, 其中N是文档总数
    # 这里假设每个文件100个文档作为估计
    estimated_docs = len(corpus_files) * 100
    if args.nlist == 100 and estimated_docs > 1000:
        suggested_nlist = int((estimated_docs ** 0.5) * 4)
        logger.info(f"估计文档数: {estimated_docs}, 建议nlist: {suggested_nlist}")
        args.nlist = min(suggested_nlist, 1000)  # 上限1000
    
    # 构建索引
    logger.info(f"索引配置: type={args.index_type}, nlist={args.nlist}, nprobe={args.nprobe}")
    
    builder = FAISSIndexBuilder(
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe
    )
    
    try:
        # 从文件构建
        builder.build_from_files(corpus_files)
        
        # 保存索引
        output_dir = Path(args.output)
        builder.save(output_dir)
        
        logger.info("=" * 60)
        logger.info("✅ Dense索引构建完成!")
        logger.info("=" * 60)
        logger.info(f"索引类型: {builder.index_type}")
        logger.info(f"文档总数: {builder.index.ntotal}")
        logger.info(f"向量维度: {builder.metadata['dimension']}")
        logger.info(f"保存路径: {output_dir}")
        logger.info("")
        logger.info("测试索引:")
        logger.info(f"  python retrieval/dense/dense_search.py \\")
        logger.info(f"    --index {output_dir} \\")
        logger.info(f"    --corpus {' '.join(str(f) for f in corpus_files)} \\")
        logger.info(f"    --query \"法语语法学习\"")
    
    except Exception as e:
        logger.error(f"索引构建失败: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
