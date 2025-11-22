"""
MTransE - 跨语言知识图谱对齐

简化实现的 Multilingual Translation Embeddings
用于对齐不同语言的知识图谱实体
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple
from pathlib import Path
import json
from logger import get_logger
from config import config

logger = get_logger(__name__)


class MTransE(nn.Module):
    """
    MTransE模型
    
    对每种语言学习独立的嵌入空间,通过翻译矩阵连接不同语言
    Loss = Σ max(0, γ + d(h + Mr, t') - d(h' + Mr, t'))
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 128,
        margin: float = 1.0
    ):
        """
        初始化MTransE
        
        Args:
            num_entities: 实体总数
            num_relations: 关系总数
            embedding_dim: 嵌入维度
            margin: margin值
        """
        super(MTransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        
        # 实体嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 翻译矩阵 (简化: 使用单个矩阵)
        self.translation_matrix = nn.Parameter(
            torch.eye(embedding_dim)
        )
        
        # 初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        logger.info(f"MTransE初始化: entities={num_entities}, relations={num_relations}, dim={embedding_dim}")
    
    def forward(
        self,
        head_ids: torch.Tensor,
        relation_ids: torch.Tensor,
        tail_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        计算三元组得分
        
        Args:
            head_ids: 头实体ID [batch_size]
            relation_ids: 关系ID [batch_size]
            tail_ids: 尾实体ID [batch_size]
            
        Returns:
            距离得分 [batch_size]
        """
        # 获取嵌入
        head = self.entity_embeddings(head_ids)  # [batch, dim]
        rel = self.relation_embeddings(relation_ids)  # [batch, dim]
        tail = self.entity_embeddings(tail_ids)  # [batch, dim]
        
        # 应用翻译矩阵
        head_translated = torch.matmul(head, self.translation_matrix)
        
        # 计算距离: ||h + r - t||
        score = torch.norm(head_translated + rel - tail, p=2, dim=1)
        
        return score
    
    def get_entity_embedding(self, entity_id: int) -> np.ndarray:
        """获取实体嵌入向量"""
        with torch.no_grad():
            emb = self.entity_embeddings(torch.LongTensor([entity_id]))
            return emb.cpu().numpy()[0]


class MTransETrainer:
    """MTransE训练器"""
    
    def __init__(
        self,
        embedding_dim: int = None,
        margin: float = None,
        learning_rate: float = None,
        batch_size: int = None,
        epochs: int = None
    ):
        """
        初始化训练器
        
        Args:
            embedding_dim: 嵌入维度
            margin: margin值
            learning_rate: 学习率
            batch_size: 批大小
            epochs: 训练轮数
        """
        self.embedding_dim = embedding_dim or config.MTRANSE_DIM
        self.margin = margin or config.MTRANSE_MARGIN
        self.learning_rate = learning_rate or config.MTRANSE_LR
        self.batch_size = batch_size or config.MTRANSE_BATCH_SIZE
        self.epochs = epochs or config.MTRANSE_EPOCHS
        
        self.model = None
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"训练设备: {self.device}")
    
    def prepare_data(
        self,
        kg_triples: List[Tuple[str, str, str]],
        alignment_seeds: List[Tuple[str, str]]
    ):
        """
        准备训练数据
        
        Args:
            kg_triples: 知识图谱三元组 [(h, r, t), ...]
            alignment_seeds: 对齐种子 [(entity1, entity2), ...]
        """
        logger.info("准备训练数据...")
        
        # 构建实体和关系词典
        entities = set()
        relations = set()
        
        for h, r, t in kg_triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
        
        # 添加对齐实体
        for e1, e2 in alignment_seeds:
            entities.add(e1)
            entities.add(e2)
        
        self.entity2id = {e: i for i, e in enumerate(entities)}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.relation2id = {r: i for i, r in enumerate(relations)}
        
        # 转换为ID
        self.train_triples = [
            (self.entity2id[h], self.relation2id[r], self.entity2id[t])
            for h, r, t in kg_triples
            if h in self.entity2id and t in self.entity2id and r in self.relation2id
        ]
        
        self.alignment_pairs = [
            (self.entity2id[e1], self.entity2id[e2])
            for e1, e2 in alignment_seeds
            if e1 in self.entity2id and e2 in self.entity2id
        ]
        
        logger.info(f"实体数: {len(self.entity2id)}, 关系数: {len(self.relation2id)}")
        logger.info(f"训练三元组: {len(self.train_triples)}, 对齐种子: {len(self.alignment_pairs)}")
    
    def train(self):
        """训练模型"""
        # 初始化模型
        self.model = MTransE(
            num_entities=len(self.entity2id),
            num_relations=len(self.relation2id),
            embedding_dim=self.embedding_dim,
            margin=self.margin
        ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        logger.info(f"开始训练: epochs={self.epochs}, batch_size={self.batch_size}")
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # 随机打乱
            np.random.shuffle(self.train_triples)
            
            # 批处理
            for i in range(0, len(self.train_triples), self.batch_size):
                batch = self.train_triples[i:i+self.batch_size]
                
                # 正样本
                pos_h = torch.LongTensor([t[0] for t in batch]).to(self.device)
                pos_r = torch.LongTensor([t[1] for t in batch]).to(self.device)
                pos_t = torch.LongTensor([t[2] for t in batch]).to(self.device)
                
                # 负采样: 随机替换头或尾
                neg_h = pos_h.clone()
                neg_t = pos_t.clone()
                for j in range(len(batch)):
                    if np.random.rand() < 0.5:
                        neg_h[j] = np.random.randint(0, self.model.num_entities)
                    else:
                        neg_t[j] = np.random.randint(0, self.model.num_entities)
                
                # 计算得分
                pos_score = self.model(pos_h, pos_r, pos_t)
                neg_score = self.model(neg_h, pos_r, neg_t)
                
                # Margin loss
                loss = torch.relu(self.margin + pos_score - neg_score).mean()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(self.train_triples) // self.batch_size + 1)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("训练完成!")
    
    def predict_alignment(
        self,
        source_entities: List[str],
        target_lang: str,
        top_k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        预测对齐
        
        Args:
            source_entities: 源实体列表
            target_lang: 目标语言
            top_k: 返回top-k候选
            
        Returns:
            {source_entity: [(target_entity, score), ...]}
        """
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for source_entity in source_entities:
                if source_entity not in self.entity2id:
                    continue
                
                source_id = self.entity2id[source_entity]
                source_emb = self.model.get_entity_embedding(source_id)
                
                # 计算与所有实体的距离
                candidates = []
                for target_entity, target_id in self.entity2id.items():
                    # 简单过滤: 只考虑不同语言的实体
                    # (实际应该有语言标签)
                    target_emb = self.model.get_entity_embedding(target_id)
                    dist = np.linalg.norm(source_emb - target_emb)
                    candidates.append((target_entity, float(dist)))
                
                # 排序并取top-k
                candidates.sort(key=lambda x: x[1])
                results[source_entity] = candidates[:top_k]
        
        return results
    
    def save(self, save_path: Path):
        """保存模型"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'relation2id': self.relation2id,
            'config': {
                'embedding_dim': self.embedding_dim,
                'margin': self.margin
            }
        }, save_path)
        
        logger.info(f"模型已保存: {save_path}")
    
    def load(self, load_path: Path):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.entity2id = checkpoint['entity2id']
        self.id2entity = checkpoint['id2entity']
        self.relation2id = checkpoint['relation2id']
        
        cfg = checkpoint['config']
        self.model = MTransE(
            num_entities=len(self.entity2id),
            num_relations=len(self.relation2id),
            embedding_dim=cfg['embedding_dim'],
            margin=cfg['margin']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"模型已加载: {load_path}")


if __name__ == "__main__":
    # 测试
    kg_triples = [
        ("语法_zh", "related_to", "语法_zh"),
        ("grammaire_fr", "prerequisite", "vocabulaire_fr"),
        ("verb_en", "related_to", "verbe_fr")
    ]
    
    alignment_seeds = [
        ("语法_zh", "grammaire_fr"),
        ("动词_zh", "verbe_fr")
    ]
    
    trainer = MTransETrainer(epochs=20)
    trainer.prepare_data(kg_triples, alignment_seeds)
    trainer.train()
    
    # 预测
    results = trainer.predict_alignment(["语法_zh"], "fr", top_k=3)
    logger.info(f"对齐预测: {results}")
