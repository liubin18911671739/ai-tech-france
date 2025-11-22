#!/usr/bin/env python3
"""端到端KG-CLIR检索脚本"""
import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from retrieval.pipelines.kg_clir import KGCLIRSystem, print_results

logger = get_logger(__name__)


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
