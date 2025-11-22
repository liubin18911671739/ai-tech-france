"""
知识图谱路径评分

对扩展的图谱路径进行评分和排序
支持多种评分策略
"""
import math
from typing import List, Dict, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config

logger = get_logger(__name__)


class KGPathScorer:
    """知识图谱路径评分器"""
    
    def __init__(
        self,
        depth_penalty: float = 0.5,
        weight_importance: float = 0.8,
        relation_weights: Dict[str, float] = None
    ):
        """
        初始化评分器
        
        Args:
            depth_penalty: 深度惩罚系数(越大,深层路径得分越低)
            weight_importance: 边权重重要性(0-1)
            relation_weights: 关系类型权重映射
        """
        self.depth_penalty = depth_penalty
        self.weight_importance = weight_importance
        
        # 默认关系权重
        self.relation_weights = relation_weights or {
            "PREREQUISITE": 1.0,     # 前置关系最重要
            "RELATED_TO": 0.8,       # 相关关系次之
            "PART_OF": 0.7,          # 部分关系
            "IS_A": 0.6,             # 类型关系
            "EQUIVALENT": 0.9        # 等价关系(对齐)
        }
        
        logger.info("KG路径评分器初始化完成")
    
    def score_path(
        self,
        path: Dict,
        method: str = "combined"
    ) -> float:
        """
        对单条路径评分
        
        Args:
            path: 路径字典 {"length": ..., "relations": [...], "weights": [...]}
            method: 评分方法 ("depth", "weight", "relation", "combined")
            
        Returns:
            路径得分
        """
        if method == "depth":
            return self._score_by_depth(path)
        elif method == "weight":
            return self._score_by_weight(path)
        elif method == "relation":
            return self._score_by_relation(path)
        elif method == "combined":
            return self._score_combined(path)
        else:
            logger.warning(f"未知评分方法: {method}, 使用combined")
            return self._score_combined(path)
    
    def _score_by_depth(self, path: Dict) -> float:
        """
        基于路径长度评分(越短越好)
        
        Score = exp(-depth_penalty * length)
        """
        length = path.get("length", len(path.get("relations", [])))
        score = math.exp(-self.depth_penalty * length)
        return score
    
    def _score_by_weight(self, path: Dict) -> float:
        """
        基于边权重评分(边权重平均值)
        """
        weights = path.get("weights", [])
        if not weights:
            # 如果没有权重信息,尝试从edges获取
            edges = path.get("edges", [])
            weights = [e.get("weight", 1.0) for e in edges]
        
        if not weights:
            return 0.5  # 默认中等得分
        
        # 平均权重
        avg_weight = sum(weights) / len(weights)
        return avg_weight
    
    def _score_by_relation(self, path: Dict) -> float:
        """
        基于关系类型评分
        """
        relations = path.get("relations", [])
        if not relations:
            return 0.5
        
        # 计算关系权重平均值
        relation_scores = []
        for rel in relations:
            weight = self.relation_weights.get(rel, 0.5)  # 未知关系默认0.5
            relation_scores.append(weight)
        
        avg_score = sum(relation_scores) / len(relation_scores)
        return avg_score
    
    def _score_combined(self, path: Dict) -> float:
        """
        组合评分(综合深度、权重、关系类型)
        
        Score = depth_score * (weight_importance * weight_score + (1 - weight_importance) * relation_score)
        """
        depth_score = self._score_by_depth(path)
        weight_score = self._score_by_weight(path)
        relation_score = self._score_by_relation(path)
        
        # 组合得分
        combined_score = depth_score * (
            self.weight_importance * weight_score +
            (1 - self.weight_importance) * relation_score
        )
        
        return combined_score
    
    def score_paths(
        self,
        paths: List[Dict],
        method: str = "combined"
    ) -> List[Dict]:
        """
        批量评分并排序
        
        Args:
            paths: 路径列表
            method: 评分方法
            
        Returns:
            添加了score字段的路径列表(按得分降序)
        """
        logger.info(f"开始评分: {len(paths)} 条路径")
        
        # 计算得分
        for path in paths:
            path["score"] = self.score_path(path, method=method)
        
        # 按得分降序排序
        paths.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"评分完成, 最高分: {paths[0]['score']:.4f}, 最低分: {paths[-1]['score']:.4f}")
        return paths
    
    def score_nodes_from_paths(
        self,
        paths: List[Dict],
        aggregation: str = "max"
    ) -> Dict[str, float]:
        """
        从路径得分聚合节点得分
        
        Args:
            paths: 已评分的路径列表
            aggregation: 聚合方法 ("max", "avg", "sum")
            
        Returns:
            节点得分字典 {node_id: score}
        """
        node_scores = {}
        node_counts = {}
        
        for path in paths:
            path_score = path.get("score", 0.0)
            node_ids = path.get("node_ids", []) or path.get("nodes", [])
            
            for node_id in node_ids:
                if node_id not in node_scores:
                    node_scores[node_id] = []
                    node_counts[node_id] = 0
                
                node_scores[node_id].append(path_score)
                node_counts[node_id] += 1
        
        # 聚合
        aggregated = {}
        for node_id, scores in node_scores.items():
            if aggregation == "max":
                aggregated[node_id] = max(scores)
            elif aggregation == "avg":
                aggregated[node_id] = sum(scores) / len(scores)
            elif aggregation == "sum":
                aggregated[node_id] = sum(scores)
            else:
                logger.warning(f"未知聚合方法: {aggregation}, 使用max")
                aggregated[node_id] = max(scores)
        
        logger.info(f"节点评分完成: {len(aggregated)} 个节点")
        return aggregated
    
    def rerank_documents(
        self,
        documents: List[Dict],
        kg_node_scores: Dict[str, float],
        alpha: float = 0.5
    ) -> List[Dict]:
        """
        使用KG得分重排文档
        
        Args:
            documents: 文档列表 [{"doc_id": ..., "score": ..., "concepts": [...]}, ...]
            kg_node_scores: 节点得分
            alpha: KG得分权重 (最终得分 = alpha * kg_score + (1-alpha) * original_score)
            
        Returns:
            重排后的文档列表
        """
        logger.info(f"开始KG重排: {len(documents)} 个文档")
        
        for doc in documents:
            original_score = doc.get("score", 0.0)
            concepts = doc.get("concepts", [])
            
            # 计算文档的KG得分(概念得分的最大值或平均值)
            if concepts:
                kg_scores = [kg_node_scores.get(c, 0.0) for c in concepts]
                kg_score = max(kg_scores) if kg_scores else 0.0
            else:
                kg_score = 0.0
            
            # 组合得分
            final_score = alpha * kg_score + (1 - alpha) * original_score
            
            doc["kg_score"] = kg_score
            doc["original_score"] = original_score
            doc["final_score"] = final_score
        
        # 按最终得分降序排序
        documents.sort(key=lambda x: x["final_score"], reverse=True)
        
        logger.info(f"KG重排完成")
        return documents
    
    def explain_score(self, path: Dict) -> Dict:
        """
        解释路径得分
        
        Args:
            path: 路径
            
        Returns:
            得分解释
        """
        explanation = {
            "total_score": path.get("score", 0.0),
            "depth_score": self._score_by_depth(path),
            "weight_score": self._score_by_weight(path),
            "relation_score": self._score_by_relation(path),
            "length": path.get("length", 0),
            "relations": path.get("relations", [])
        }
        
        return explanation


def main():
    """测试函数"""
    # 模拟路径数据
    test_paths = [
        {
            "node_ids": ["concept1", "concept2", "concept3"],
            "relations": ["PREREQUISITE", "RELATED_TO"],
            "weights": [1.0, 0.8],
            "length": 2
        },
        {
            "node_ids": ["concept1", "concept4"],
            "relations": ["RELATED_TO"],
            "weights": [0.6],
            "length": 1
        },
        {
            "node_ids": ["concept1", "concept5", "concept6", "concept7"],
            "relations": ["IS_A", "PART_OF", "RELATED_TO"],
            "weights": [0.7, 0.6, 0.5],
            "length": 3
        }
    ]
    
    # 创建评分器
    scorer = KGPathScorer()
    
    # 评分
    scored_paths = scorer.score_paths(test_paths, method="combined")
    
    # 输出结果
    logger.info("\n路径评分结果:")
    for i, path in enumerate(scored_paths, 1):
        logger.info(f"\nPath {i}:")
        logger.info(f"  Length: {path['length']}")
        logger.info(f"  Relations: {path['relations']}")
        logger.info(f"  Score: {path['score']:.4f}")
        
        # 详细解释
        explanation = scorer.explain_score(path)
        logger.info(f"  - Depth score: {explanation['depth_score']:.4f}")
        logger.info(f"  - Weight score: {explanation['weight_score']:.4f}")
        logger.info(f"  - Relation score: {explanation['relation_score']:.4f}")
    
    # 节点聚合得分
    node_scores = scorer.score_nodes_from_paths(scored_paths, aggregation="max")
    logger.info(f"\n节点得分:")
    for node_id, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {node_id}: {score:.4f}")


if __name__ == "__main__":
    main()
