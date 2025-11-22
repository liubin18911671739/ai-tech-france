#!/usr/bin/env python3
"""一键运行 MVP 流水线: 构建索引 + 运行检索 + 可选评测"""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config
from retrieval.dense.build_faiss import FAISSIndexBuilder
from retrieval.sparse.build_whoosh import WhooshIndexBuilder
from retrieval.pipelines.kg_clir import KGCLIRSystem, print_results
from retrieval.eval.pipeline import CLIREvaluationPipeline

logger = get_logger(__name__)


def discover_corpus_files() -> list[Path]:
    files = sorted(config.CLEANED_DIR.glob("corpus_*_cleaned.jsonl"))
    if not files:
        logger.error(f"未在 {config.CLEANED_DIR} 找到清洗后的语料")
    return files


def build_dense_index(corpus_files: list[Path], output_dir: Path) -> None:
    logger.info("\n=== 构建 FAISS 索引 ===")
    builder = FAISSIndexBuilder(index_type="Flat", nlist=1, nprobe=1)
    builder.build_from_files(corpus_files)
    output_dir.mkdir(parents=True, exist_ok=True)
    builder.save(output_dir)
    logger.info(f"FAISS 索引保存在: {output_dir}")


def build_sparse_index(corpus_files: list[Path], output_dir: Path) -> None:
    logger.info("\n=== 构建 Whoosh 索引 ===")
    builder = WhooshIndexBuilder(index_dir=output_dir)
    builder.build_from_files(corpus_files, force_new=True)
    logger.info(f"Whoosh 索引保存在: {output_dir}")


def run_demo_queries(dense_dir: Path, sparse_dir: Path, queries_file: Path, top_k: int = 3) -> None:
    if not queries_file.exists():
        logger.warning(f"查询文件不存在: {queries_file}")
        return

    system = KGCLIRSystem(
        dense_index_dir=dense_dir,
        sparse_index_dir=sparse_dir,
        use_kg=True,
        top_k_per_source=20
    )

    corpus_files = discover_corpus_files()
    if corpus_files:
        system.dense_searcher.load_corpus(corpus_files)

    logger.info("\n=== 运行示例查询 ===")
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            results = system.search(
                query=data.get("query", ""),
                lang=data.get("lang", "auto"),
                top_k=top_k,
                explain=True
            )
            print_results(results, top_k=top_k)
    system.close()


def main():
    parser = argparse.ArgumentParser(description="MVP 管线: 构建索引并运行示例")
    parser.add_argument("--dense-dir", type=Path, default=Path("artifacts/faiss_labse"), help="FAISS 输出目录")
    parser.add_argument("--sparse-dir", type=Path, default=Path("artifacts/whoosh_bm25"), help="Whoosh 输出目录")
    parser.add_argument("--queries", type=Path, default=config.EVAL_DIR / "clir_queries.jsonl", help="演示查询文件")
    parser.add_argument("--run-eval", action="store_true", help="完成索引后顺便运行评测")

    args = parser.parse_args()

    corpus_files = discover_corpus_files()
    if not corpus_files:
        return 1

    build_dense_index(corpus_files, args.dense_dir)
    build_sparse_index(corpus_files, args.sparse_dir)

    run_demo_queries(args.dense_dir, args.sparse_dir, args.queries)

    if args.run_eval:
        pipeline = CLIREvaluationPipeline(
            corpus_dir=config.CLEANED_DIR,
            queries_file=args.queries,
            qrels_file=config.EVAL_DIR / "qrels.tsv",
            output_dir=Path("artifacts/mvp_eval"),
            dense_index_dir=args.dense_dir,
            sparse_index_dir=args.sparse_dir,
            use_kg=True,
            top_k=20
        )
        pipeline.run_evaluation()

    logger.info("\nMVP 流水线完成 ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
