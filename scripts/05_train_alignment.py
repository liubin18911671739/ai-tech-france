#!/usr/bin/env python3
"""
训练跨语言对齐模型

完整的对齐训练流程:
1. 加载图谱数据
2. 加载种子对齐
3. 训练MTransE模型
4. 保存模型
5. 测试对齐预测
"""
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from kg.alignment.train_alignment import AlignmentTrainer

logger = get_logger(__name__)


def create_seed_alignment(output_file: Path):
    """
    创建种子对齐数据(Mock)
    
    Args:
        output_file: 输出文件
    """
    logger.info(f"创建Mock种子对齐: {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Mock对齐对
    alignments = [
        ("apprentissage automatique", "机器学习"),
        ("machine learning", "机器学习"),
        ("apprentissage profond", "深度学习"),
        ("deep learning", "深度学习"),
        ("réseaux de neurones", "神经网络"),
        ("neural networks", "神经网络"),
        ("réseaux convolutifs", "卷积神经网络"),
        ("convolutional neural networks", "卷积神经网络"),
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 种子对齐数据\n")
        f.write("# 格式: entity1\tentity2\n\n")
        
        for e1, e2 in alignments:
            f.write(f"{e1}\t{e2}\n")
    
    logger.info(f"创建 {len(alignments)} 对种子对齐")


def main():
    """主流程"""
    parser = argparse.ArgumentParser(description="训练跨语言对齐模型")
    
    # 输入文件
    parser.add_argument("--nodes", type=str,
                       default="data/kg/nodes.jsonl",
                       help="节点文件")
    parser.add_argument("--relations", type=str,
                       default="data/kg/relations.jsonl",
                       help="关系文件")
    parser.add_argument("--alignment", type=str,
                       default="data/seeds/seed_align.tsv",
                       help="种子对齐文件")
    
    # 输出
    parser.add_argument("--output", type=str,
                       default="artifacts/alignment_model",
                       help="输出目录")
    
    # 训练参数
    parser.add_argument("--embedding-dim", type=int, default=100,
                       help="嵌入维度")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="批大小")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="学习率")
    parser.add_argument("--margin", type=float, default=1.0,
                       help="Margin参数")
    
    # Mock模式
    parser.add_argument("--create-seeds", action="store_true",
                       help="创建Mock种子对齐")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("开始训练跨语言对齐模型")
    logger.info("=" * 60)
    
    # ============================================================
    # Step 0: 创建种子对齐 (可选)
    # ============================================================
    alignment_file = Path(args.alignment)
    
    if args.create_seeds or not alignment_file.exists():
        logger.info("\n[Step 0/5] 创建种子对齐")
        create_seed_alignment(alignment_file)
    
    # ============================================================
    # Step 1: 检查输入文件
    # ============================================================
    logger.info("\n[Step 1/5] 检查输入文件")
    
    nodes_file = Path(args.nodes)
    relations_file = Path(args.relations)
    
    if not nodes_file.exists():
        logger.error(f"节点文件不存在: {nodes_file}")
        logger.info("请先运行: python scripts/04_build_mkg.py --mock")
        return
    
    if not relations_file.exists():
        logger.error(f"关系文件不存在: {relations_file}")
        logger.info("请先运行: python scripts/04_build_mkg.py --mock")
        return
    
    logger.info(f"节点文件: {nodes_file} ✓")
    logger.info(f"关系文件: {relations_file} ✓")
    logger.info(f"对齐文件: {alignment_file} ✓")
    
    # ============================================================
    # Step 2: 创建训练器
    # ============================================================
    logger.info("\n[Step 2/5] 初始化训练器")
    
    trainer = AlignmentTrainer(
        embedding_dim=args.embedding_dim,
        margin=args.margin,
        learning_rate=args.learning_rate
    )
    
    logger.info(f"配置:")
    logger.info(f"  - 嵌入维度: {args.embedding_dim}")
    logger.info(f"  - 训练轮数: {args.epochs}")
    logger.info(f"  - 批大小: {args.batch_size}")
    logger.info(f"  - 学习率: {args.learning_rate}")
    logger.info(f"  - Margin: {args.margin}")
    
    # ============================================================
    # Step 3: 加载数据
    # ============================================================
    logger.info("\n[Step 3/5] 加载图谱数据")
    
    triples = trainer.load_graph(
        nodes_file=nodes_file,
        relations_file=relations_file
    )
    
    seed_alignments = trainer.load_seed_alignment(
        alignment_file=alignment_file
    )
    
    if not triples:
        logger.error("没有加载到三元组,无法训练")
        return
    
    if not seed_alignments:
        logger.warning("没有种子对齐,将仅使用结构信息训练")
    
    # ============================================================
    # Step 4: 训练模型
    # ============================================================
    logger.info("\n[Step 4/5] 训练模型")
    logger.info(f"开始训练 {args.epochs} 轮...")
    
    losses = trainer.train(
        triples=triples,
        seed_alignments=seed_alignments,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    logger.info(f"\n训练完成!")
    logger.info(f"  - 初始损失: {losses[0]:.4f}")
    logger.info(f"  - 最终损失: {losses[-1]:.4f}")
    logger.info(f"  - 损失下降: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # ============================================================
    # Step 5: 保存模型
    # ============================================================
    logger.info("\n[Step 5/5] 保存模型")
    
    output_dir = Path(args.output)
    trainer.save_model(output_dir)
    
    logger.info(f"模型已保存到: {output_dir}")
    
    # ============================================================
    # Step 6: 测试对齐预测
    # ============================================================
    logger.info("\n[Bonus] 测试对齐预测")
    
    # 测试实体
    test_cases = [
        ("fr_CONCEPT_000000", "zh"),  # 法语 -> 中文
        ("fr_CONCEPT_000001", "en"),  # 法语 -> 英文
        ("zh_CONCEPT_000000", "en"),  # 中文 -> 英文
    ]
    
    for source_entity, target_lang in test_cases:
        if source_entity not in trainer.entity2id:
            continue
        
        predictions = trainer.predict_alignment(
            source_entity=source_entity,
            target_lang=target_lang,
            top_k=3
        )
        
        logger.info(f"\n{source_entity} -> {target_lang}:")
        for target, similarity in predictions:
            # 查找目标实体名称
            target_name = "未知"
            for entity_str, entity_id in trainer.entity2id.items():
                if trainer.entity2id.get(target) == entity_id:
                    # 从节点文件查找名称
                    import json
                    with open(nodes_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            obj = json.loads(line.strip())
                            if obj["id"] == target:
                                target_name = obj["name"]
                                break
                    break
            
            logger.info(f"  {target} ({target_name}): {similarity:.4f}")
    
    # ============================================================
    # 总结
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("对齐训练完成!")
    logger.info("=" * 60)
    logger.info(f"\n输出:")
    logger.info(f"  - 实体嵌入: {output_dir}/entity_embeddings.npy")
    logger.info(f"  - 关系嵌入: {output_dir}/relation_embeddings.npy")
    logger.info(f"  - 实体映射: {output_dir}/entity2id.json")
    logger.info(f"  - 关系映射: {output_dir}/relation2id.json")
    
    logger.info("\n下一步:")
    logger.info("  1. 运行KG-CLIR检索: python scripts/08_run_kg_clir.py")
    logger.info("  2. 运行评测: python scripts/09_eval_clir.py --use-kg")


if __name__ == "__main__":
    main()
