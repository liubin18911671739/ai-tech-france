"""
训练跨语言对齐模型

使用MTransE算法训练实体对齐
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kg.alignment.mtrans_e import MTransE
from logger import get_logger

logger = get_logger(__name__)


class AlignmentTrainer:
    """对齐训练器"""
    
    def __init__(
        self,
        embedding_dim: int = 100,
        margin: float = 1.0,
        learning_rate: float = 0.01
    ):
        """
        初始化训练器
        
        Args:
            embedding_dim: 嵌入维度
            margin: Margin损失参数
            learning_rate: 学习率
        """
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.learning_rate = learning_rate
        
        self.model = None
        self.entity2id = {}
        self.relation2id = {}
        
        logger.info(f"对齐训练器初始化: dim={embedding_dim}, margin={margin}, lr={learning_rate}")
    
    def load_graph(
        self,
        nodes_file: Path,
        relations_file: Path
    ):
        """
        加载图谱数据
        
        Args:
            nodes_file: 节点文件
            relations_file: 关系文件
        """
        logger.info("加载图谱数据")
        
        # 加载节点
        entities = []
        with open(nodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                entity_id = obj["id"]
                entities.append(entity_id)
                self.entity2id[entity_id] = len(self.entity2id)
        
        logger.info(f"加载 {len(entities)} 个实体")
        
        # 加载关系
        relations = []
        triples = []
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                source = obj["source"]
                target = obj["target"]
                rel_type = obj["type"]
                
                if rel_type not in self.relation2id:
                    self.relation2id[rel_type] = len(self.relation2id)
                
                if source in self.entity2id and target in self.entity2id:
                    triple = (
                        self.entity2id[source],
                        self.relation2id[rel_type],
                        self.entity2id[target]
                    )
                    triples.append(triple)
        
        logger.info(f"加载 {len(triples)} 个三元组, {len(self.relation2id)} 种关系")
        
        return triples
    
    def load_seed_alignment(
        self,
        alignment_file: Path
    ) -> List[Tuple[str, str]]:
        """
        加载种子对齐
        
        Args:
            alignment_file: 对齐文件(TSV)
            
        Returns:
            对齐对列表 [(entity1, entity2)]
        """
        logger.info(f"加载种子对齐: {alignment_file}")
        
        alignments = []
        
        if not alignment_file.exists():
            logger.warning(f"对齐文件不存在: {alignment_file}")
            return alignments
        
        with open(alignment_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                entity1 = parts[0]
                entity2 = parts[1]
                
                alignments.append((entity1, entity2))
        
        logger.info(f"加载 {len(alignments)} 对种子对齐")
        return alignments
    
    def train(
        self,
        triples: List[Tuple[int, int, int]],
        seed_alignments: List[Tuple[str, str]],
        epochs: int = 100,
        batch_size: int = 128
    ):
        """
        训练对齐模型
        
        Args:
            triples: 三元组列表
            seed_alignments: 种子对齐
            epochs: 训练轮数
            batch_size: 批大小
        """
        logger.info(f"开始训练: epochs={epochs}, batch_size={batch_size}")
        
        # 初始化模型
        self.model = MTransE(
            num_entities=len(self.entity2id),
            num_relations=len(self.relation2id),
            embedding_dim=self.embedding_dim,
            margin=self.margin
        )
        
        # 转换种子对齐为ID
        seed_pairs = []
        for e1, e2 in seed_alignments:
            if e1 in self.entity2id and e2 in self.entity2id:
                seed_pairs.append((
                    self.entity2id[e1],
                    self.entity2id[e2]
                ))
        
        logger.info(f"有效种子对齐: {len(seed_pairs)}")
        
        # 训练
        losses = []
        
        for epoch in range(epochs):
            # 随机打乱
            np.random.shuffle(triples)
            
            epoch_loss = 0.0
            num_batches = 0
            
            # 批次训练
            for i in range(0, len(triples), batch_size):
                batch = triples[i:i+batch_size]
                
                # 转换为numpy数组
                batch_array = np.array(batch)
                heads = batch_array[:, 0]
                relations = batch_array[:, 1]
                tails = batch_array[:, 2]
                
                # 训练步
                loss = self.model.train_step(
                    heads=heads,
                    relations=relations,
                    tails=tails,
                    learning_rate=self.learning_rate
                )
                
                epoch_loss += loss
                num_batches += 1
            
            # 对齐损失
            if seed_pairs:
                alignment_loss = self._compute_alignment_loss(seed_pairs)
                epoch_loss += alignment_loss
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("训练完成")
        return losses
    
    def _compute_alignment_loss(
        self,
        seed_pairs: List[Tuple[int, int]]
    ) -> float:
        """
        计算对齐损失
        
        Args:
            seed_pairs: 种子对齐对
            
        Returns:
            对齐损失
        """
        loss = 0.0
        
        for e1, e2 in seed_pairs:
            # 获取嵌入
            emb1 = self.model.entity_embeddings[e1]
            emb2 = self.model.entity_embeddings[e2]
            
            # L2距离
            dist = np.linalg.norm(emb1 - emb2)
            loss += dist
        
        return loss / len(seed_pairs)
    
    def predict_alignment(
        self,
        source_entity: str,
        target_lang: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        预测对齐
        
        Args:
            source_entity: 源实体ID
            target_lang: 目标语言
            top_k: 返回Top-K
            
        Returns:
            [(target_entity, similarity)]
        """
        if source_entity not in self.entity2id:
            logger.warning(f"实体不存在: {source_entity}")
            return []
        
        source_id = self.entity2id[source_entity]
        source_emb = self.model.entity_embeddings[source_id]
        
        # 计算与目标语言实体的相似度
        candidates = []
        
        for entity, entity_id in self.entity2id.items():
            # 过滤目标语言
            if not entity.startswith(f"{target_lang}_"):
                continue
            
            target_emb = self.model.entity_embeddings[entity_id]
            
            # 余弦相似度
            similarity = np.dot(source_emb, target_emb) / (
                np.linalg.norm(source_emb) * np.linalg.norm(target_emb)
            )
            
            candidates.append((entity, similarity))
        
        # 排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def save_model(self, output_dir: Path):
        """
        保存模型
        
        Args:
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存嵌入
        np.save(
            output_dir / "entity_embeddings.npy",
            self.model.entity_embeddings
        )
        np.save(
            output_dir / "relation_embeddings.npy",
            self.model.relation_embeddings
        )
        
        # 保存映射
        with open(output_dir / "entity2id.json", 'w', encoding='utf-8') as f:
            json.dump(self.entity2id, f, ensure_ascii=False, indent=2)
        
        with open(output_dir / "relation2id.json", 'w', encoding='utf-8') as f:
            json.dump(self.relation2id, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存: {output_dir}")
    
    def load_model(self, model_dir: Path):
        """
        加载模型
        
        Args:
            model_dir: 模型目录
        """
        model_dir = Path(model_dir)
        
        # 加载嵌入
        entity_emb = np.load(model_dir / "entity_embeddings.npy")
        relation_emb = np.load(model_dir / "relation_embeddings.npy")
        
        # 加载映射
        with open(model_dir / "entity2id.json", 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        
        with open(model_dir / "relation2id.json", 'r', encoding='utf-8') as f:
            self.relation2id = json.load(f)
        
        # 重建模型
        self.model = MTransE(
            num_entities=len(self.entity2id),
            num_relations=len(self.relation2id),
            embedding_dim=entity_emb.shape[1],
            margin=self.margin
        )
        
        self.model.entity_embeddings = entity_emb
        self.model.relation_embeddings = relation_emb
        
        logger.info(f"模型已加载: {model_dir}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练跨语言对齐模型")
    parser.add_argument("--nodes", type=str,
                       default="data/kg/nodes.jsonl",
                       help="节点文件")
    parser.add_argument("--relations", type=str,
                       default="data/kg/relations.jsonl",
                       help="关系文件")
    parser.add_argument("--alignment", type=str,
                       default="data/seeds/seed_align.tsv",
                       help="种子对齐文件")
    parser.add_argument("--output", type=str,
                       default="artifacts/alignment_model",
                       help="输出目录")
    parser.add_argument("--embedding-dim", type=int, default=100,
                       help="嵌入维度")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="批大小")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                       help="学习率")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = AlignmentTrainer(
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate
    )
    
    # 加载图谱
    triples = trainer.load_graph(
        nodes_file=Path(args.nodes),
        relations_file=Path(args.relations)
    )
    
    # 加载种子对齐
    seed_alignments = trainer.load_seed_alignment(
        alignment_file=Path(args.alignment)
    )
    
    # 训练
    losses = trainer.train(
        triples=triples,
        seed_alignments=seed_alignments,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # 保存
    trainer.save_model(Path(args.output))
    
    # 测试对齐预测
    logger.info("\n=== 测试对齐预测 ===")
    test_entities = ["fr_CONCEPT_000001", "zh_CONCEPT_000001"]
    
    for entity in test_entities:
        if entity in trainer.entity2id:
            predictions = trainer.predict_alignment(entity, "en", top_k=3)
            logger.info(f"\n{entity} 的对齐预测:")
            for target, similarity in predictions:
                logger.info(f"  {target}: {similarity:.4f}")


if __name__ == "__main__":
    main()
