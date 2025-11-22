#!/usr/bin/env python3
"""完整CLIR评测流程 (CLI 包装)"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from retrieval.eval.pipeline import CLIREvaluationPipeline

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="完整CLIR评测流程")
    parser.add_argument("--corpus-dir", type=str, default="data/cleaned", help="语料目录")
    parser.add_argument("--queries", type=str, default="data/eval/clir_queries.jsonl", help="查询文件")
    parser.add_argument("--qrels", type=str, default="data/eval/qrels.tsv", help="相关性标注文件")
    parser.add_argument("--dense-index", type=str, default="artifacts/faiss_labse", help="Dense索引目录")
    parser.add_argument("--sparse-index", type=str, default="artifacts/whoosh_bm25", help="Sparse索引目录")
    parser.add_argument("--output-dir", type=str, default="artifacts/eval_results", help="输出目录")
    parser.add_argument("--use-kg", action="store_true", help="是否使用KG增强")
    parser.add_argument("--top-k", type=int, default=100, help="返回结果数")

    args = parser.parse_args()

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

    pipeline.run_evaluation()
    logger.info("\n✅ 评测完成!")


if __name__ == "__main__":
    main()
