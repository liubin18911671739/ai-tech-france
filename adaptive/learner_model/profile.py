#!/usr/bin/env python3
"""
学习者画像

整合学习行为数据，构建完整的学习者画像
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from logger import get_logger
from .mastery import MasteryEstimator

logger = get_logger(__name__)


class LearnerProfile:
    """学习者画像"""
    
    def __init__(self, learner_id: str):
        """
        初始化学习者画像
        
        Args:
            learner_id: 学习者ID
        """
        self.learner_id = learner_id
        self.mastery_estimator = MasteryEstimator()
        
        # 学习数据
        self.concept_events = defaultdict(list)  # {concept_id: [events]}
        self.query_history = []                  # 查询历史
        self.resource_views = []                 # 资源浏览
        
        # 统计数据
        self.total_study_time = 0                # 总学习时长(秒)
        self.session_count = 0                   # 学习会话数
        self.last_active_time = None             # 最后活跃时间
        
        logger.info(f"学习者画像创建: {learner_id}")
    
    def add_event(
        self,
        concept_id: str,
        event_type: str,
        timestamp: float,
        success: Optional[bool] = None,
        duration: int = 0,
        metadata: Optional[Dict] = None
    ):
        """
        添加学习事件
        
        Args:
            concept_id: 概念ID
            event_type: 事件类型 (view/practice/test)
            timestamp: 时间戳
            success: 成功与否 (练习/测试)
            duration: 持续时间(秒)
            metadata: 额外元数据
        """
        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "success": success,
            "duration": duration,
            "metadata": metadata or {}
        }
        
        self.concept_events[concept_id].append(event)
        self.total_study_time += duration
        self.last_active_time = timestamp
    
    def add_query(
        self,
        query: str,
        lang: str,
        timestamp: float,
        results: List[str],
        clicked_docs: List[str]
    ):
        """添加查询历史"""
        self.query_history.append({
            "query": query,
            "lang": lang,
            "timestamp": timestamp,
            "results": results,
            "clicked_docs": clicked_docs
        })
    
    def add_resource_view(
        self,
        doc_id: str,
        concepts: List[str],
        timestamp: float,
        duration: int
    ):
        """添加资源浏览记录"""
        self.resource_views.append({
            "doc_id": doc_id,
            "concepts": concepts,
            "timestamp": timestamp,
            "duration": duration
        })
        
        # 同步到概念事件
        for concept in concepts:
            self.add_event(
                concept_id=concept,
                event_type="view",
                timestamp=timestamp,
                duration=duration
            )
    
    def get_mastery_profile(self, current_time: float) -> Dict[str, float]:
        """
        获取概念掌握度画像
        
        Returns:
            {concept_id: mastery_score}
        """
        return self.mastery_estimator.batch_estimate(
            self.concept_events,
            current_time
        )
    
    def get_weak_concepts(
        self,
        current_time: float,
        threshold: float = 0.5,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        识别薄弱概念
        
        Args:
            current_time: 当前时间
            threshold: 薄弱阈值
            top_k: 返回前k个
            
        Returns:
            [(concept_id, mastery_score)]
        """
        mastery_profile = self.get_mastery_profile(current_time)
        
        # 筛选薄弱概念
        weak = [
            (cid, score)
            for cid, score in mastery_profile.items()
            if score < threshold
        ]
        
        # 按掌握度排序 (最弱的在前)
        weak.sort(key=lambda x: x[1])
        
        return weak[:top_k]
    
    def get_learning_preferences(self) -> Dict:
        """
        分析学习偏好
        
        Returns:
            学习偏好统计
        """
        # 语言偏好
        lang_counts = defaultdict(int)
        for query in self.query_history:
            lang_counts[query["lang"]] += 1
        
        # 学习时段
        hour_counts = defaultdict(int)
        for events in self.concept_events.values():
            for event in events:
                hour = datetime.fromtimestamp(event["timestamp"]).hour
                hour_counts[hour] += 1
        
        # 学习风格 (实践 vs 理论)
        practice_count = sum(
            1 for events in self.concept_events.values()
            for e in events if e["event_type"] in ["practice", "test"]
        )
        view_count = sum(
            1 for events in self.concept_events.values()
            for e in events if e["event_type"] == "view"
        )
        
        return {
            "preferred_languages": dict(lang_counts),
            "active_hours": dict(hour_counts),
            "learning_style": {
                "practice_ratio": practice_count / max(practice_count + view_count, 1),
                "view_ratio": view_count / max(practice_count + view_count, 1)
            },
            "total_concepts": len(self.concept_events),
            "avg_study_time_per_concept": self.total_study_time / max(len(self.concept_events), 1)
        }
    
    def get_summary(self, current_time: float) -> Dict:
        """
        获取完整画像摘要
        
        Returns:
            学习者画像摘要
        """
        mastery_profile = self.get_mastery_profile(current_time)
        weak_concepts = self.get_weak_concepts(current_time, top_k=5)
        preferences = self.get_learning_preferences()
        
        # 掌握度统计
        mastery_levels = defaultdict(int)
        for score in mastery_profile.values():
            level = self.mastery_estimator.get_mastery_level(score)
            mastery_levels[level] += 1
        
        return {
            "learner_id": self.learner_id,
            "summary": {
                "total_concepts": len(self.concept_events),
                "total_queries": len(self.query_history),
                "total_resources_viewed": len(self.resource_views),
                "total_study_time_hours": self.total_study_time / 3600,
                "session_count": self.session_count,
                "last_active": datetime.fromtimestamp(self.last_active_time).isoformat() if self.last_active_time else None
            },
            "mastery_distribution": dict(mastery_levels),
            "weak_concepts": [
                {"concept_id": cid, "mastery": score}
                for cid, score in weak_concepts
            ],
            "preferences": preferences,
            "avg_mastery": sum(mastery_profile.values()) / max(len(mastery_profile), 1)
        }
    
    def save(self, output_path: Path):
        """保存画像到文件"""
        summary = self.get_summary(datetime.now().timestamp())
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"学习者画像已保存: {output_path}")
    
    @classmethod
    def load_from_logs(
        cls,
        learner_id: str,
        log_file: Path
    ) -> "LearnerProfile":
        """
        从日志文件加载学习数据
        
        Args:
            learner_id: 学习者ID
            log_file: 日志文件路径
            
        Returns:
            学习者画像实例
        """
        profile = cls(learner_id)
        
        if not log_file.exists():
            logger.warning(f"日志文件不存在: {log_file}")
            return profile
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    
                    if entry.get("learner_id") != learner_id:
                        continue
                    
                    event_type = entry.get("event_type")
                    
                    if event_type in ["view", "practice", "test"]:
                        profile.add_event(
                            concept_id=entry["concept_id"],
                            event_type=event_type,
                            timestamp=entry["timestamp"],
                            success=entry.get("success"),
                            duration=entry.get("duration", 0)
                        )
                    
                    elif event_type == "query":
                        profile.add_query(
                            query=entry["query"],
                            lang=entry["lang"],
                            timestamp=entry["timestamp"],
                            results=entry.get("results", []),
                            clicked_docs=entry.get("clicked_docs", [])
                        )
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"学习数据加载完成: {len(profile.concept_events)} 个概念")
        return profile


def demo():
    """演示学习者画像"""
    import time
    
    # 创建画像
    profile = LearnerProfile("learner_001")
    
    current_time = time.time()
    
    # 模拟学习活动
    profile.add_resource_view(
        doc_id="doc_001",
        concepts=["grammaire", "syntaxe"],
        timestamp=current_time - 86400 * 7,
        duration=300
    )
    
    profile.add_event(
        concept_id="grammaire",
        event_type="practice",
        timestamp=current_time - 86400 * 5,
        success=False,
        duration=180
    )
    
    profile.add_event(
        concept_id="grammaire",
        event_type="practice",
        timestamp=current_time - 86400 * 3,
        success=True,
        duration=240
    )
    
    profile.add_query(
        query="法语语法",
        lang="zh",
        timestamp=current_time - 86400 * 2,
        results=["doc_001", "doc_002"],
        clicked_docs=["doc_001"]
    )
    
    # 生成画像摘要
    summary = profile.get_summary(current_time)
    
    print("\n=== 学习者画像演示 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
