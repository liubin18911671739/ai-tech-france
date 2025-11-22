# 自适应学习分析系统

本模块提供完整的自适应学习支持，包括学习者画像构建、概念掌握度评估和个性化学习路径推荐。

## 🎯 核心功能

### 1. 概念掌握度评估 (`mastery.py`)

基于改进的 **BKT (Bayesian Knowledge Tracing)** 模型，估计学习者对各概念的掌握程度。

**核心算法:**
- **初始化**: P(mastery) = p_init (默认 0.1)
- **学习更新**: P(t+1) = P(t) + (1-P(t)) × p_learn
- **贝叶斯更新**: 根据练习/测试结果调整概率
- **时间衰减**: P(t) = P₀ × exp(-λt) + P_init × (1-exp(-λt))

**掌握等级:**
- `mastered` (≥0.8): 已掌握
- `familiar` (≥0.5): 熟悉
- `learning` (≥0.2): 学习中
- `novice` (<0.2): 初学

### 2. 学习者画像 (`profile.py`)

整合学习行为数据，构建完整的学习者画像。

**数据收集:**
- 概念学习事件 (浏览/练习/测试)
- 查询历史
- 资源浏览记录
- 学习时长统计

**画像维度:**
- 概念掌握度分布
- 薄弱概念识别
- 学习偏好分析 (语言偏好/学习时段/学习风格)
- 学习进度追踪

### 3. 学习路径推荐 (`recommend_path.py`)

基于知识图谱和学习者掌握度，推荐个性化学习路径。

**核心算法:**
- **拓扑排序**: 确保 prerequisite 顺序
- **掌握度优先**: 薄弱概念优先学习
- **路径优化**: 考虑前置关系和学习成本

**推荐策略:**
- 单目标推荐: 针对特定概念生成学习路径
- 批量推荐: 针对多个薄弱概念统一规划
- 适应性调整: 根据学习进度动态更新

## 📊 使用示例

### 1. 运行完整分析

```bash
# 使用模拟数据
python scripts/10_run_pilot_analysis.py \
  --learner-ids learner_001 learner_002 learner_003 \
  --output-dir artifacts/pilot_analysis

# 使用真实日志数据
python scripts/10_run_pilot_analysis.py \
  --learner-ids user_123 user_456 \
  --log-dir data/learning_logs \
  --kg-file data/kg/relations.jsonl \
  --output-dir artifacts/analysis_results
```

### 2. 评估概念掌握度

```python
from adaptive.learner_model.mastery import MasteryEstimator
import time

estimator = MasteryEstimator()

# 学习事件
events = [
    {"timestamp": time.time() - 86400*7, "event_type": "view", "duration": 300},
    {"timestamp": time.time() - 86400*5, "event_type": "practice", "success": False, "duration": 180},
    {"timestamp": time.time() - 86400*3, "event_type": "practice", "success": True, "duration": 240},
    {"timestamp": time.time() - 86400*1, "event_type": "test", "success": True, "duration": 120}
]

# 评估掌握度
mastery = estimator.estimate_mastery("grammaire", events, time.time())
level = estimator.get_mastery_level(mastery)
print(f"掌握度: {mastery:.3f} ({level})")
```

### 3. 构建学习者画像

```python
from adaptive.learner_model.profile import LearnerProfile
import time

profile = LearnerProfile("user_001")

# 添加学习事件
profile.add_event(
    concept_id="grammaire",
    event_type="practice",
    timestamp=time.time(),
    success=True,
    duration=180
)

# 添加查询记录
profile.add_query(
    query="法语语法",
    lang="zh",
    timestamp=time.time(),
    results=["doc_001", "doc_002"],
    clicked_docs=["doc_001"]
)

# 生成画像摘要
summary = profile.get_summary(time.time())
print(json.dumps(summary, ensure_ascii=False, indent=2))

# 识别薄弱概念
weak = profile.get_weak_concepts(time.time(), threshold=0.5, top_k=5)
```

### 4. 推荐学习路径

```python
from adaptive.path_reco.recommend_path import PathRecommender

recommender = PathRecommender()

# 加载知识图谱
with open("data/kg/relations.jsonl") as f:
    relations = [json.loads(line) for line in f]
recommender.load_kg(relations)

# 推荐路径
mastery_scores = {
    "alphabet": 0.9,
    "pronunciation": 0.7,
    "vocabulary": 0.4,
    "grammar": 0.2
}

path = recommender.recommend_path(
    goal_concept="writing",
    mastery_scores=mastery_scores,
    max_length=10
)

print("推荐学习路径:")
for step in path:
    print(f"{step['step']}. {step['concept_id']} (掌握度: {step['current_mastery']:.2f})")
    print(f"   原因: {step['reason']}")
```

## 📁 输出格式

### 个人报告 (`learner_XXX_report.json`)

```json
{
  "learner_id": "learner_001",
  "analysis_time": "2025-11-22T15:21:38",
  "profile_summary": {
    "summary": {
      "total_concepts": 8,
      "total_queries": 4,
      "total_study_time_hours": 2.09
    },
    "mastery_distribution": {
      "learning": 4,
      "familiar": 2,
      "novice": 2
    },
    "weak_concepts": [...],
    "preferences": {
      "preferred_languages": {"zh": 2, "en": 2},
      "learning_style": {
        "practice_ratio": 0.73,
        "view_ratio": 0.27
      }
    },
    "avg_mastery": 0.393
  },
  "recommended_path": [
    {
      "concept_id": "alphabet",
      "current_mastery": 0.354,
      "step": 1,
      "reason": "基础薄弱，需要重点学习",
      "prerequisites": []
    }
  ],
  "recommendations": {
    "overall": ["您已经掌握了一些基础知识，继续保持学习节奏"],
    "study_tips": ["练习很充足，可以适当增加理论学习和阅读"],
    "next_steps": [
      "建议下一步学习: alphabet (当前掌握度: 0.35)",
      "后续计划: pronunciation → vocabulary_basic → grammar_basic"
    ]
  }
}
```

### 汇总报告 (`pilot_summary_report.json`)

```json
{
  "analysis_date": "2025-11-22T15:21:38",
  "total_learners": 3,
  "statistics": {
    "avg_concepts_studied": 8.0,
    "avg_mastery_score": 0.393,
    "common_weak_concepts": [
      {"concept_id": "sentence_structure", "learner_count": 3},
      {"concept_id": "vocabulary_advanced", "learner_count": 3}
    ]
  },
  "individual_results": [
    {
      "learner_id": "learner_001",
      "total_concepts": 8,
      "avg_mastery": 0.393,
      "weak_concept_count": 5
    }
  ]
}
```

## 🔧 参数调优

### 掌握度模型参数

```python
estimator = MasteryEstimator(
    p_init=0.1,       # 初始掌握概率 (0.05-0.15)
    p_learn=0.3,      # 学习率 (0.2-0.4)
    p_guess=0.2,      # 猜测概率 (0.1-0.3)
    p_slip=0.1,       # 失误概率 (0.05-0.15)
    decay_rate=0.05   # 遗忘率/天 (0.01-0.1)
)
```

### 路径推荐参数

```python
path = recommender.recommend_path(
    goal_concept="target",
    mastery_scores={...},
    max_length=10,           # 最大路径长度 (5-20)
    mastery_threshold=0.7    # 掌握阈值 (0.6-0.8)
)
```

## 📈 应用场景

1. **个性化推荐系统**
   - 根据学习者画像推荐资源
   - 动态调整学习路径
   - 提供个性化建议

2. **学习进度追踪**
   - 监控概念掌握度变化
   - 识别学习瓶颈
   - 预测学习成效

3. **教学辅助**
   - 教师了解学生整体情况
   - 识别共同薄弱点
   - 调整教学策略

4. **学习分析研究**
   - 分析学习行为模式
   - 评估教学效果
   - 优化课程设计

## 🎓 技术特点

1. **科学建模**: 基于认知科学的 BKT 模型
2. **实时更新**: 支持增量式学习数据更新
3. **可解释性**: 提供详细的推荐理由
4. **灵活扩展**: 易于集成新的学习数据源
5. **高效计算**: 批量处理优化

## 📚 参考文献

- Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User modeling and user-adapted interaction*, 4(4), 253-278.
- Piech, C., et al. (2015). Deep knowledge tracing. *Advances in neural information processing systems*, 28.

## 🔄 后续优化方向

1. **深度学习模型**: 使用 LSTM/Transformer 替代 BKT
2. **多维度建模**: 考虑学习时间、难度等因素
3. **协同过滤**: 利用其他学习者的数据
4. **强化学习**: 优化路径推荐策略
5. **实时反馈**: 在线学习和模型更新

---

**注**: 本模块为 Future Work，用于演示自适应学习功能，可根据实际需求进一步完善。
