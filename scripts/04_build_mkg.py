#!/usr/bin/env python3
"""
构建多语种知识图谱

完整的图谱构建流程:
1. 加载实体和关系
2. 构建节点和边
3. 导入Neo4j
4. 生成统计报告
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from kg.neo4j_import.build_nodes_rels import GraphBuilder
from kg.neo4j_import.import_to_neo4j import Neo4jImporter
from kg.stats.graph_stats import GraphStatistics

logger = get_logger(__name__)


def main():
    """主流程"""
    parser = argparse.ArgumentParser(description="构建多语种知识图谱")
    
    # 输入文件
    parser.add_argument("--entities-fr", type=str,
                       help="法语实体文件")
    parser.add_argument("--entities-zh", type=str,
                       help="中文实体文件")
    parser.add_argument("--entities-en", type=str,
                       help="英文实体文件")
    parser.add_argument("--relations-fr", type=str,
                       help="法语关系文件")
    parser.add_argument("--relations-zh", type=str,
                       help="中文关系文件")
    parser.add_argument("--relations-en", type=str,
                       help="英文关系文件")
    parser.add_argument("--alignment", type=str,
                       help="对齐文件")
    
    # 输出
    parser.add_argument("--output-dir", type=str,
                       default="data/kg",
                       help="输出目录")
    
    # Mock模式
    parser.add_argument("--mock", action="store_true",
                       help="使用Mock数据")
    
    # Neo4j导入
    parser.add_argument("--import-neo4j", action="store_true",
                       help="导入到Neo4j")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("开始构建多语种知识图谱")
    logger.info("=" * 60)
    
    # ============================================================
    # Step 1: 构建节点和关系
    # ============================================================
    logger.info("\n[Step 1/4] 构建节点和关系")
    
    builder = GraphBuilder()
    
    if args.mock:
        logger.info("使用Mock数据")
        builder.build_mock_graph()
    else:
        # 从实体构建节点
        if args.entities_fr:
            builder.build_from_entities(Path(args.entities_fr), "fr")
        if args.entities_zh:
            builder.build_from_entities(Path(args.entities_zh), "zh")
        if args.entities_en:
            builder.build_from_entities(Path(args.entities_en), "en")
        
        # 从关系构建边
        if args.relations_fr:
            builder.build_from_relations(Path(args.relations_fr))
        if args.relations_zh:
            builder.build_from_relations(Path(args.relations_zh))
        if args.relations_en:
            builder.build_from_relations(Path(args.relations_en))
        
        # 添加对齐关系
        if args.alignment:
            builder.add_alignment_relations(Path(args.alignment))
    
    # 导出节点和关系
    nodes_file = output_dir / "nodes.jsonl"
    relations_file = output_dir / "relations.jsonl"
    
    builder.export_nodes(nodes_file)
    builder.export_relations(relations_file)
    
    # 统计
    stats = builder.get_statistics()
    logger.info(f"\n构建完成:")
    logger.info(f"  节点数: {stats['total_concepts']}")
    logger.info(f"  关系数: {stats['total_relations']}")
    logger.info(f"  按语言: {stats['concepts_by_lang']}")
    
    # ============================================================
    # Step 2: 导入Neo4j (可选)
    # ============================================================
    if args.import_neo4j:
        logger.info("\n[Step 2/4] 导入Neo4j")
        
        try:
            importer = Neo4jImporter()
            
            # 清空数据库
            logger.info("清空现有数据...")
            importer.clear_database()
            
            # 创建约束
            logger.info("创建约束...")
            importer.create_constraints()
            
            # 导入节点
            logger.info("导入节点...")
            concepts = []
            with open(nodes_file, 'r', encoding='utf-8') as f:
                import json
                for line in f:
                    concepts.append(json.loads(line.strip()))
            
            importer.import_concepts(concepts)
            
            # 导入关系
            logger.info("导入关系...")
            relations = []
            with open(relations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    relations.append(json.loads(line.strip()))
            
            importer.import_relations(relations)
            
            # 统计
            neo4j_stats = importer.get_statistics()
            logger.info(f"\nNeo4j导入完成:")
            logger.info(f"  节点数: {neo4j_stats['node_count']}")
            logger.info(f"  关系数: {neo4j_stats['relationship_count']}")
            
            importer.close()
        
        except Exception as e:
            logger.error(f"Neo4j导入失败: {e}")
            logger.warning("跳过Neo4j导入,继续后续步骤")
    else:
        logger.info("\n[Step 2/4] 跳过Neo4j导入")
    
    # ============================================================
    # Step 3: 生成统计报告
    # ============================================================
    logger.info("\n[Step 3/4] 生成统计报告")
    
    stats_analyzer = GraphStatistics()
    stats_analyzer.load_graph(nodes_file, relations_file)
    
    report_file = output_dir.parent / "artifacts" / "graph_stats.md"
    stats_analyzer.export_report(report_file)
    
    logger.info(f"统计报告已生成: {report_file}")
    
    # ============================================================
    # Step 4: 总结
    # ============================================================
    logger.info("\n[Step 4/4] 构建完成")
    logger.info("=" * 60)
    logger.info("知识图谱构建完成!")
    logger.info("=" * 60)
    logger.info(f"\n输出文件:")
    logger.info(f"  - 节点文件: {nodes_file}")
    logger.info(f"  - 关系文件: {relations_file}")
    logger.info(f"  - 统计报告: {report_file}")
    
    if args.import_neo4j:
        logger.info(f"\nNeo4j已导入,可通过以下方式访问:")
        logger.info(f"  - Neo4j Browser: http://localhost:7474")
        logger.info(f"  - Cypher查询: MATCH (n) RETURN n LIMIT 25")
    
    logger.info("\n下一步:")
    if not args.import_neo4j:
        logger.info("  1. 导入Neo4j: python scripts/04_build_mkg.py --import-neo4j --mock")
    logger.info("  2. 训练对齐: python scripts/05_train_alignment.py")
    logger.info("  3. 运行检索: python scripts/08_run_kg_clir.py")


if __name__ == "__main__":
    main()
