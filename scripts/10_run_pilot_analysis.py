#!/usr/bin/env python3
"""
运行试点学习分析

分析学习者数据，生成学习画像和推荐学习路径
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from adaptive.learner_model.mastery import MasteryEstimator
from adaptive.learner_model.profile import LearnerProfile
from adaptive.path_reco.recommend_path import PathRecommender

logger = get_logger(__name__)


class PilotAnalyzer:
    """试点学习分析器"""
    
    def __init__(
        self,
        kg_file: Path,
        output_dir: Path
    ):
        """
        初始化分析器
        
        Args:
            kg_file: 知识图谱关系文件
            output_dir: 输出目录
        """
        self.kg_file = kg_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.mastery_estimator = MasteryEstimator()
        self.recommender = PathRecommender()
        
        logger.info(f"试点分析器初始化: {output_dir}")
    
    def load_knowledge_graph(self):
        """加载知识图谱"""
        if not self.kg_file.exists():
            logger.warning(f"知识图谱文件不存在: {self.kg_file}")
            logger.info("使用模拟知识图谱")
            self._create_mock_kg()
            return
        
        with open(self.kg_file, "r", encoding="utf-8") as f:
            relations = [json.loads(line) for line in f]
        
        self.recommender.load_kg(relations)
        logger.info(f"知识图谱加载: {len(relations)} 个关系")
    
    def _create_mock_kg(self):
        """创建模拟知识图谱 (用于演示)"""
        mock_relations = [
            # 法语学习路径
            {"source": "alphabet", "target": "pronunciation", "type": "PREREQUISITE", "weight": 1.0},
            {"source": "pronunciation", "target": "vocabulary_basic", "type": "PREREQUISITE", "weight": 1.0},
            {"source": "vocabulary_basic", "target": "grammar_basic", "type": "PREREQUISITE", "weight": 0.9},
            {"source": "grammar_basic", "target": "sentence_structure", "type": "PREREQUISITE", "weight": 1.0},
            {"source": "vocabulary_basic", "target": "vocabulary_advanced", "type": "PREREQUISITE", "weight": 0.8},
            {"source": "grammar_basic", "target": "grammar_advanced", "type": "PREREQUISITE", "weight": 0.9},
            {"source": "sentence_structure", "target": "reading_comprehension", "type": "PREREQUISITE", "weight": 0.8},
            {"source": "grammar_advanced", "target": "writing_composition", "type": "PREREQUISITE", "weight": 0.9},
            {"source": "vocabulary_advanced", "target": "writing_composition", "type": "PREREQUISITE", "weight": 0.7},
            
            # 语法细分
            {"source": "grammar_basic", "target": "verb_conjugation", "type": "PREREQUISITE", "weight": 0.9},
            {"source": "verb_conjugation", "target": "tense_usage", "type": "PREREQUISITE", "weight": 1.0},
            {"source": "grammar_basic", "target": "article_usage", "type": "PREREQUISITE", "weight": 0.8},
            {"source": "sentence_structure", "target": "complex_sentences", "type": "PREREQUISITE", "weight": 0.9},
        ]
        
        self.recommender.load_kg(mock_relations)
        
        # 保存模拟数据
        mock_file = self.output_dir / "mock_kg_relations.jsonl"
        with open(mock_file, "w", encoding="utf-8") as f:
            for rel in mock_relations:
                f.write(json.dumps(rel, ensure_ascii=False) + "\n")
        
        logger.info(f"模拟知识图谱已创建: {mock_file}")
    
    def generate_mock_learner_data(self, learner_id: str) -> LearnerProfile:
        """生成模拟学习者数据 (用于演示)"""
        profile = LearnerProfile(learner_id)
        
        current_time = time.time()
        
        # 模拟30天学习活动
        concepts = [
            "alphabet", "pronunciation", "vocabulary_basic",
            "grammar_basic", "verb_conjugation", "article_usage",
            "sentence_structure", "vocabulary_advanced"
        ]
        
        for i, concept in enumerate(concepts):
            days_ago = 30 - i * 3
            
            # 浏览
            profile.add_event(
                concept_id=concept,
                event_type="view",
                timestamp=current_time - 86400 * days_ago,
                duration=300 + i * 50
            )
            
            # 练习
            if i < 6:  # 前6个概念有练习
                for j in range(2 + i // 2):
                    success = j > 0 or i < 3  # 第一次可能失败
                    profile.add_event(
                        concept_id=concept,
                        event_type="practice",
                        timestamp=current_time - 86400 * (days_ago - j - 1),
                        success=success,
                        duration=180
                    )
            
            # 测试
            if i < 4:  # 前4个概念有测试
                profile.add_event(
                    concept_id=concept,
                    event_type="test",
                    timestamp=current_time - 86400 * (days_ago - 5),
                    success=True,
                    duration=120
                )
        
        # 添加查询历史
        queries = [
            ("法语发音规则", "zh"),
            ("French grammar basics", "en"),
            ("动词变位", "zh"),
            ("article usage", "en")
        ]
        
        for i, (query, lang) in enumerate(queries):
            profile.add_query(
                query=query,
                lang=lang,
                timestamp=current_time - 86400 * (20 - i * 5),
                results=[f"doc_{i}01", f"doc_{i}02"],
                clicked_docs=[f"doc_{i}01"]
            )
        
        logger.info(f"模拟学习数据生成: {learner_id}, {len(profile.concept_events)} 个概念")
        return profile
    
    def analyze_learner(
        self,
        learner_id: str,
        log_file: Optional[Path] = None
    ) -> Dict:
        """
        分析学习者
        
        Args:
            learner_id: 学习者ID
            log_file: 学习日志文件 (可选，否则使用模拟数据)
            
        Returns:
            分析结果
        """
        logger.info(f"开始分析学习者: {learner_id}")
        
        # 加载或生成学习数据
        if log_file and log_file.exists():
            profile = LearnerProfile.load_from_logs(learner_id, log_file)
        else:
            logger.info("使用模拟学习数据")
            profile = self.generate_mock_learner_data(learner_id)
        
        current_time = time.time()
        
        # 生成学习画像
        summary = profile.get_summary(current_time)
        
        # 识别薄弱概念
        weak_concepts = profile.get_weak_concepts(current_time, threshold=0.6, top_k=5)
        
        # 推荐学习路径
        if weak_concepts:
            path = self.recommender.recommend_batch(
                weak_concepts,
                profile.get_mastery_profile(current_time),
                max_total=10
            )
        else:
            path = []
            logger.info("没有薄弱概念，学习状态良好！")
        
        # 整合结果
        analysis_result = {
            "learner_id": learner_id,
            "analysis_time": datetime.now().isoformat(),
            "profile_summary": summary,
            "weak_concepts": [
                {"concept_id": cid, "mastery": score}
                for cid, score in weak_concepts
            ],
            "recommended_path": path,
            "recommendations": self._generate_recommendations(summary, weak_concepts, path)
        }
        
        return analysis_result
    
    def _generate_recommendations(
        self,
        summary: Dict,
        weak_concepts: List[Tuple[str, float]],
        path: List[Dict]
    ) -> Dict:
        """生成个性化建议"""
        recommendations = {
            "overall": [],
            "study_tips": [],
            "next_steps": []
        }
        
        # 整体建议
        avg_mastery = summary.get("avg_mastery", 0)
        if avg_mastery < 0.3:
            recommendations["overall"].append("您还处于学习初期，建议先打好基础，从简单概念开始")
        elif avg_mastery < 0.6:
            recommendations["overall"].append("您已经掌握了一些基础知识，继续保持学习节奏")
        else:
            recommendations["overall"].append("您的学习进展良好，可以尝试更高级的内容")
        
        # 学习风格建议
        preferences = summary.get("preferences", {})
        learning_style = preferences.get("learning_style", {})
        practice_ratio = learning_style.get("practice_ratio", 0)
        
        if practice_ratio < 0.3:
            recommendations["study_tips"].append("建议增加练习和测试，实践是掌握知识的关键")
        elif practice_ratio > 0.7:
            recommendations["study_tips"].append("练习很充足，可以适当增加理论学习和阅读")
        
        # 下一步建议
        if path:
            first_concept = path[0]["concept_id"]
            recommendations["next_steps"].append(
                f"建议下一步学习: {first_concept} (当前掌握度: {path[0]['current_mastery']:.2f})"
            )
            
            if len(path) > 1:
                recommendations["next_steps"].append(
                    f"后续计划: {' → '.join([p['concept_id'] for p in path[1:4]])}"
                )
        
        # 时间建议
        total_time = summary["summary"].get("total_study_time_hours", 0)
        if total_time < 10:
            recommendations["study_tips"].append("建议每周学习3-5小时，保持稳定的学习节奏")
        
        return recommendations
    
    def run_analysis(
        self,
        learner_ids: List[str],
        log_files: Optional[List[Path]] = None
    ):
        """
        运行批量分析
        
        Args:
            learner_ids: 学习者ID列表
            log_files: 日志文件列表 (可选)
        """
        logger.info(f"开始试点分析: {len(learner_ids)} 个学习者")
        
        # 加载知识图谱
        self.load_knowledge_graph()
        
        # 分析每个学习者
        all_results = []
        
        for i, learner_id in enumerate(learner_ids):
            log_file = log_files[i] if log_files and i < len(log_files) else None
            
            try:
                result = self.analyze_learner(learner_id, log_file)
                all_results.append(result)
                
                # 保存个人报告
                output_file = self.output_dir / f"learner_{learner_id}_report.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"分析完成: {learner_id} → {output_file}")
                
            except Exception as e:
                logger.error(f"分析失败 {learner_id}: {e}")
                continue
        
        # 生成汇总报告
        self._generate_summary_report(all_results)
        
        logger.info(f"试点分析完成: {len(all_results)}/{len(learner_ids)} 个学习者")
    
    def _generate_summary_report(self, results: List[Dict]):
        """生成汇总报告"""
        summary_file = self.output_dir / "pilot_summary_report.json"
        
        # 统计数据
        total_learners = len(results)
        avg_concepts = sum(
            r["profile_summary"]["summary"]["total_concepts"]
            for r in results
        ) / max(total_learners, 1)
        
        avg_mastery = sum(
            r["profile_summary"].get("avg_mastery", 0)
            for r in results
        ) / max(total_learners, 1)
        
        # 常见薄弱概念
        weak_concept_counts = {}
        for r in results:
            for wc in r["weak_concepts"]:
                cid = wc["concept_id"]
                weak_concept_counts[cid] = weak_concept_counts.get(cid, 0) + 1
        
        common_weak = sorted(
            weak_concept_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "total_learners": total_learners,
            "statistics": {
                "avg_concepts_studied": avg_concepts,
                "avg_mastery_score": avg_mastery,
                "common_weak_concepts": [
                    {"concept_id": cid, "learner_count": count}
                    for cid, count in common_weak
                ]
            },
            "individual_results": [
                {
                    "learner_id": r["learner_id"],
                    "total_concepts": r["profile_summary"]["summary"]["total_concepts"],
                    "avg_mastery": r["profile_summary"].get("avg_mastery", 0),
                    "weak_concept_count": len(r["weak_concepts"])
                }
                for r in results
            ]
        }
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"汇总报告已生成: {summary_file}")
        
        # 打印摘要
        print("\n" + "="*60)
        print("试点学习分析报告")
        print("="*60)
        print(f"\n总学习者数: {total_learners}")
        print(f"平均学习概念数: {avg_concepts:.1f}")
        print(f"平均掌握度: {avg_mastery:.3f}")
        print(f"\n常见薄弱概念 (Top 5):")
        for cid, count in common_weak[:5]:
            print(f"  - {cid}: {count} 个学习者")
        print("\n" + "="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行试点学习分析")
    parser.add_argument(
        "--kg-file",
        type=Path,
        default=Path("data/kg/relations.jsonl"),
        help="知识图谱关系文件"
    )
    parser.add_argument(
        "--learner-ids",
        type=str,
        nargs="+",
        default=["learner_001", "learner_002", "learner_003"],
        help="学习者ID列表"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="学习日志目录 (可选，否则使用模拟数据)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pilot_analysis"),
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = PilotAnalyzer(
        kg_file=args.kg_file,
        output_dir=args.output_dir
    )
    
    # 准备日志文件
    log_files = None
    if args.log_dir and args.log_dir.exists():
        log_files = [
            args.log_dir / f"{lid}.log"
            for lid in args.learner_ids
        ]
    
    # 运行分析
    analyzer.run_analysis(
        learner_ids=args.learner_ids,
        log_files=log_files
    )
    
    print(f"\n✅ 分析完成！结果已保存到: {args.output_dir}")
    print(f"\n查看个人报告: {args.output_dir}/learner_*_report.json")
    print(f"查看汇总报告: {args.output_dir}/pilot_summary_report.json")


if __name__ == "__main__":
    main()
