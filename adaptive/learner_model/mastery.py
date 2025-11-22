#!/usr/bin/env python3
"""
概念掌握度评估模型

基于学习行为数据估计学习者对各概念的掌握程度
使用改进的BKT (Bayesian Knowledge Tracing) 模型
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple
import math
from collections import defaultdict

from logger import get_logger

logger = get_logger(__name__)


class MasteryEstimator:
    """概念掌握度评估器"""
    
    def __init__(
        self,
        p_init: float = 0.1,      # 初始掌握概率
        p_learn: float = 0.3,     # 学习率
        p_guess: float = 0.2,     # 猜测概率
        p_slip: float = 0.1,      # 失误概率
        decay_rate: float = 0.05  # 遗忘率 (每天)
    ):
        """
        初始化评估器
        
        Args:
            p_init: 初始掌握概率
            p_learn: 每次学习后的掌握提升概率
            p_guess: 未掌握时猜对的概率
            p_slip: 已掌握时答错的概率
            decay_rate: 遗忘率 (随时间衰减)
        """
        self.p_init = p_init
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip
        self.decay_rate = decay_rate
        
        logger.info(f"掌握度评估器初始化: p_init={p_init}, p_learn={p_learn}")
    
    def estimate_mastery(
        self,
        concept_id: str,
        learning_events: List[Dict],
        current_time: float
    ) -> float:
        """
        估计概念掌握度
        
        Args:
            concept_id: 概念ID
            learning_events: 学习事件列表 [{
                "timestamp": 时间戳,
                "event_type": "view"|"practice"|"test",
                "success": True/False (练习/测试结果),
                "duration": 持续时间(秒)
            }]
            current_time: 当前时间戳
            
        Returns:
            掌握概率 [0, 1]
        """
        if not learning_events:
            return self.p_init
        
        # 按时间排序
        events = sorted(learning_events, key=lambda x: x["timestamp"])
        
        # BKT更新
        p_mastery = self.p_init
        
        for event in events:
            event_type = event.get("event_type", "view")
            success = event.get("success", None)
            
            if event_type == "view":
                # 仅浏览，轻微提升
                p_mastery = self._update_mastery_view(p_mastery)
            
            elif event_type in ["practice", "test"] and success is not None:
                # 练习或测试，根据结果更新
                p_mastery = self._update_mastery_performance(
                    p_mastery, success
                )
        
        # 时间衰减
        if events:
            last_time = events[-1]["timestamp"]
            days_elapsed = (current_time - last_time) / 86400  # 转换为天
            p_mastery = self._apply_decay(p_mastery, days_elapsed)
        
        return p_mastery
    
    def _update_mastery_view(self, p_mastery: float) -> float:
        """更新掌握度 (仅浏览)"""
        # 轻微学习效果
        return p_mastery + (1 - p_mastery) * self.p_learn * 0.3
    
    def _update_mastery_performance(
        self,
        p_mastery: float,
        success: bool
    ) -> float:
        """根据练习/测试结果更新掌握度 (BKT更新)"""
        if success:
            # 答对：贝叶斯更新
            p_correct_if_mastered = 1 - self.p_slip
            p_correct_if_not = self.p_guess
            
            p_correct = (
                p_mastery * p_correct_if_mastered +
                (1 - p_mastery) * p_correct_if_not
            )
            
            # 后验概率
            p_mastery_new = (
                p_mastery * p_correct_if_mastered / p_correct
            )
        else:
            # 答错：贝叶斯更新
            p_wrong_if_mastered = self.p_slip
            p_wrong_if_not = 1 - self.p_guess
            
            p_wrong = (
                p_mastery * p_wrong_if_mastered +
                (1 - p_mastery) * p_wrong_if_not
            )
            
            # 后验概率
            p_mastery_new = (
                p_mastery * p_wrong_if_mastered / p_wrong
            )
        
        # 学习效果
        p_mastery_final = p_mastery_new + (1 - p_mastery_new) * self.p_learn
        
        return min(p_mastery_final, 0.99)  # 上限
    
    def _apply_decay(self, p_mastery: float, days: float) -> float:
        """应用遗忘衰减"""
        # 指数衰减: P(t) = P0 * exp(-λt) + (1-exp(-λt)) * P_init
        decay_factor = math.exp(-self.decay_rate * days)
        return p_mastery * decay_factor + self.p_init * (1 - decay_factor)
    
    def batch_estimate(
        self,
        learner_data: Dict[str, List[Dict]],
        current_time: float
    ) -> Dict[str, float]:
        """
        批量估计多个概念的掌握度
        
        Args:
            learner_data: {concept_id: [events]}
            current_time: 当前时间戳
            
        Returns:
            {concept_id: mastery_score}
        """
        results = {}
        for concept_id, events in learner_data.items():
            results[concept_id] = self.estimate_mastery(
                concept_id, events, current_time
            )
        
        logger.info(f"批量评估完成: {len(results)} 个概念")
        return results
    
    def get_mastery_level(self, score: float) -> str:
        """获取掌握等级"""
        if score >= 0.8:
            return "mastered"  # 已掌握
        elif score >= 0.5:
            return "familiar"  # 熟悉
        elif score >= 0.2:
            return "learning"  # 学习中
        else:
            return "novice"    # 初学


def demo():
    """演示掌握度评估"""
    import time
    
    estimator = MasteryEstimator()
    
    # 模拟学习事件
    current_time = time.time()
    
    events_concept1 = [
        {
            "timestamp": current_time - 86400 * 7,  # 7天前
            "event_type": "view",
            "duration": 300
        },
        {
            "timestamp": current_time - 86400 * 5,  # 5天前
            "event_type": "practice",
            "success": False,
            "duration": 180
        },
        {
            "timestamp": current_time - 86400 * 3,  # 3天前
            "event_type": "practice",
            "success": True,
            "duration": 240
        },
        {
            "timestamp": current_time - 86400 * 1,  # 1天前
            "event_type": "test",
            "success": True,
            "duration": 120
        }
    ]
    
    events_concept2 = [
        {
            "timestamp": current_time - 86400 * 2,  # 2天前
            "event_type": "view",
            "duration": 200
        }
    ]
    
    # 评估掌握度
    mastery1 = estimator.estimate_mastery("grammaire", events_concept1, current_time)
    mastery2 = estimator.estimate_mastery("verbe", events_concept2, current_time)
    
    print("\n=== 掌握度评估演示 ===")
    print(f"\n概念1 (grammaire):")
    print(f"  学习事件数: {len(events_concept1)}")
    print(f"  掌握度: {mastery1:.4f}")
    print(f"  等级: {estimator.get_mastery_level(mastery1)}")
    
    print(f"\n概念2 (verbe):")
    print(f"  学习事件数: {len(events_concept2)}")
    print(f"  掌握度: {mastery2:.4f}")
    print(f"  等级: {estimator.get_mastery_level(mastery2)}")
    
    # 批量评估
    learner_data = {
        "grammaire": events_concept1,
        "verbe": events_concept2
    }
    
    batch_results = estimator.batch_estimate(learner_data, current_time)
    print(f"\n批量评估结果:")
    for concept, score in batch_results.items():
        level = estimator.get_mastery_level(score)
        print(f"  {concept}: {score:.4f} ({level})")


if __name__ == "__main__":
    demo()
