#!/usr/bin/env python3
"""
学习路径推荐

基于prerequisite关系和学习者掌握度推荐学习路径
使用拓扑排序确保prerequisite顺序
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque

from logger import get_logger

logger = get_logger(__name__)


class PathRecommender:
    """学习路径推荐器"""
    
    def __init__(self):
        """初始化推荐器"""
        self.graph = defaultdict(list)      # {concept: [prerequisites]}
        self.reverse_graph = defaultdict(list)  # {concept: [dependents]}
        self.all_concepts = set()
        
        logger.info("学习路径推荐器初始化")
    
    def load_kg(self, kg_relations: List[Dict]):
        """
        加载知识图谱关系
        
        Args:
            kg_relations: [{
                "source": concept_id,
                "target": concept_id,
                "type": "PREREQUISITE"|"RELATED_TO"|...,
                "weight": float
            }]
        """
        self.graph.clear()
        self.reverse_graph.clear()
        self.all_concepts.clear()
        
        for rel in kg_relations:
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            
            self.all_concepts.add(source)
            self.all_concepts.add(target)
            
            if rel_type == "PREREQUISITE":
                # target需要source作为前置
                self.graph[target].append(source)
                self.reverse_graph[source].append(target)
        
        logger.info(f"知识图谱加载完成: {len(self.all_concepts)} 个概念, "
                   f"{sum(len(v) for v in self.graph.values())} 个前置关系")
    
    def topological_sort(
        self,
        target_concepts: Set[str],
        mastery_scores: Dict[str, float],
        mastery_threshold: float = 0.7
    ) -> List[str]:
        """
        拓扑排序生成学习路径
        
        Args:
            target_concepts: 目标概念集合
            mastery_scores: 当前掌握度 {concept_id: score}
            mastery_threshold: 认为已掌握的阈值
            
        Returns:
            有序学习路径 [concept_id]
        """
        # 找到所有需要学习的概念 (包括prerequisites)
        to_learn = set()
        queue = deque(target_concepts)
        visited = set()
        
        while queue:
            concept = queue.popleft()
            if concept in visited:
                continue
            visited.add(concept)
            
            # 如果未掌握，加入学习列表
            if mastery_scores.get(concept, 0) < mastery_threshold:
                to_learn.add(concept)
                
                # 添加prerequisites
                for prereq in self.graph.get(concept, []):
                    if prereq not in visited:
                        queue.append(prereq)
        
        if not to_learn:
            return []
        
        # 计算入度 (仅考虑to_learn中的概念)
        in_degree = {c: 0 for c in to_learn}
        local_graph = defaultdict(list)
        
        for concept in to_learn:
            for prereq in self.graph.get(concept, []):
                if prereq in to_learn:
                    in_degree[concept] += 1
                    local_graph[prereq].append(concept)
        
        # 拓扑排序 (Kahn算法)
        queue = deque([c for c in to_learn if in_degree[c] == 0])
        result = []
        
        while queue:
            # 按掌握度排序 (掌握度低的优先)
            queue = deque(sorted(
                queue,
                key=lambda c: mastery_scores.get(c, 0)
            ))
            
            concept = queue.popleft()
            result.append(concept)
            
            # 更新后续概念的入度
            for dependent in local_graph[concept]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # 检测循环依赖
        if len(result) < len(to_learn):
            logger.warning(f"检测到循环依赖: {len(to_learn) - len(result)} 个概念无法排序")
            # 添加剩余概念
            remaining = to_learn - set(result)
            result.extend(sorted(remaining, key=lambda c: mastery_scores.get(c, 0)))
        
        return result
    
    def recommend_path(
        self,
        goal_concept: str,
        mastery_scores: Dict[str, float],
        max_length: int = 10,
        mastery_threshold: float = 0.7
    ) -> List[Dict]:
        """
        推荐学习路径
        
        Args:
            goal_concept: 目标概念
            mastery_scores: 当前掌握度
            max_length: 最大路径长度
            mastery_threshold: 掌握阈值
            
        Returns:
            学习路径 [{
                "concept_id": str,
                "current_mastery": float,
                "step": int,
                "reason": str
            }]
        """
        if goal_concept not in self.all_concepts:
            logger.warning(f"目标概念不存在: {goal_concept}")
            return []
        
        # 拓扑排序
        path_concepts = self.topological_sort(
            {goal_concept},
            mastery_scores,
            mastery_threshold
        )
        
        # 限制长度
        path_concepts = path_concepts[:max_length]
        
        # 构建详细路径
        path = []
        for i, concept in enumerate(path_concepts):
            mastery = mastery_scores.get(concept, 0)
            
            # 判断原因
            if i == len(path_concepts) - 1:
                reason = "目标概念"
            elif mastery < 0.3:
                reason = "基础薄弱，需要重点学习"
            elif mastery < mastery_threshold:
                reason = "需要巩固提升"
            else:
                reason = "复习巩固"
            
            path.append({
                "concept_id": concept,
                "current_mastery": mastery,
                "step": i + 1,
                "reason": reason,
                "prerequisites": self.graph.get(concept, [])
            })
        
        logger.info(f"学习路径生成: {goal_concept}, 共{len(path)}步")
        return path
    
    def recommend_batch(
        self,
        weak_concepts: List[Tuple[str, float]],
        mastery_scores: Dict[str, float],
        max_total: int = 20
    ) -> List[Dict]:
        """
        批量推荐 (针对多个薄弱概念)
        
        Args:
            weak_concepts: [(concept_id, score)] 薄弱概念列表
            mastery_scores: 掌握度
            max_total: 最大推荐数
            
        Returns:
            学习路径
        """
        target_set = {cid for cid, _ in weak_concepts}
        
        path_concepts = self.topological_sort(
            target_set,
            mastery_scores,
            mastery_threshold=0.7
        )
        
        path_concepts = path_concepts[:max_total]
        
        path = []
        for i, concept in enumerate(path_concepts):
            mastery = mastery_scores.get(concept, 0)
            
            path.append({
                "concept_id": concept,
                "current_mastery": mastery,
                "step": i + 1,
                "reason": "薄弱概念补强",
                "prerequisites": self.graph.get(concept, [])
            })
        
        return path


def demo():
    """演示路径推荐"""
    recommender = PathRecommender()
    
    # 模拟知识图谱
    kg_relations = [
        {"source": "alphabet", "target": "pronunciation", "type": "PREREQUISITE", "weight": 1.0},
        {"source": "pronunciation", "target": "vocabulary", "type": "PREREQUISITE", "weight": 1.0},
        {"source": "vocabulary", "target": "grammar", "type": "PREREQUISITE", "weight": 0.8},
        {"source": "grammar", "target": "sentence", "type": "PREREQUISITE", "weight": 1.0},
        {"source": "vocabulary", "target": "reading", "type": "PREREQUISITE", "weight": 0.6},
        {"source": "grammar", "target": "writing", "type": "PREREQUISITE", "weight": 0.9},
    ]
    
    recommender.load_kg(kg_relations)
    
    # 模拟掌握度
    mastery_scores = {
        "alphabet": 0.9,
        "pronunciation": 0.7,
        "vocabulary": 0.4,
        "grammar": 0.2,
        "sentence": 0.1,
        "reading": 0.3,
        "writing": 0.0
    }
    
    # 推荐路径
    path = recommender.recommend_path(
        goal_concept="writing",
        mastery_scores=mastery_scores,
        max_length=8
    )
    
    print("\n=== 学习路径推荐演示 ===")
    print(f"\n目标: writing")
    print(f"当前掌握度: {mastery_scores.get('writing', 0):.2f}")
    print(f"\n推荐学习路径 (共{len(path)}步):")
    
    for step in path:
        print(f"\nStep {step['step']}: {step['concept_id']}")
        print(f"  当前掌握度: {step['current_mastery']:.2f}")
        print(f"  原因: {step['reason']}")
        if step['prerequisites']:
            print(f"  前置概念: {', '.join(step['prerequisites'])}")


if __name__ == "__main__":
    demo()
