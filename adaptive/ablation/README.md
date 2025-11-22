# 消融实验 (Ablation Study)

本模块提供系统化的消融实验，用于评测各检索组件对整体性能的贡献。

## 🎯 实验目的

消融实验通过逐步移除或组合不同组件，量化每个组件的贡献：

1. **单组件性能**: Dense-only、Sparse-only、KG-only
2. **两组件组合**: Dense+Sparse、Dense+KG、Sparse+KG
3. **完整系统**: Dense+Sparse+KG (论文提出的方法)

## 📊 实验配置

| Configuration | Dense (α) | Sparse (β) | KG (γ) | 说明 |
|---------------|-----------|------------|--------|------|
| Dense-only | 1.0 | 0.0 | 0.0 | 仅LaBSE向量检索 |
| Sparse-only | 0.0 | 1.0 | 0.0 | 仅BM25词汇匹配 |
| KG-only | 0.0 | 0.0 | 1.0 | 仅知识图谱路径评分 |
| Dense+Sparse | 0.6 | 0.4 | 0.0 | 向量+词汇混合 |
| Dense+KG | 0.6 | 0.0 | 0.4 | 向量+知识增强 |
| Sparse+KG | 0.0 | 0.6 | 0.4 | 词汇+知识增强 |
| **Full (Ours)** | **0.4** | **0.3** | **0.3** | **完整系统** ✨ |

## 🚀 使用方法

### 基本用法

```bash
# 运行完整消融实验
python adaptive/ablation/run_ablation.py \
  --dense-index artifacts/faiss_labse \
  --sparse-index artifacts/whoosh_bm25 \
  --queries data/eval/clir_queries.jsonl \
  --qrels data/eval/qrels.tsv \
  --output-dir artifacts/ablation_results
```

### 自定义参数

```bash
# 指定top-k和Neo4j连接
python adaptive/ablation/run_ablation.py \
  --dense-index artifacts/faiss_labse \
  --sparse-index artifacts/whoosh_bm25 \
  --queries data/eval/clir_queries.jsonl \
  --qrels data/eval/qrels.tsv \
  --output-dir artifacts/ablation_custom \
  --top-k 50 \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password password
```

### 使用默认路径

```bash
# 所有参数都有合理的默认值
python adaptive/ablation/run_ablation.py
```

## 📁 输出文件

运行完成后，会在输出目录生成三个文件：

### 1. `ablation_results.json`

完整的实验结果（JSON格式）：

```json
[
  {
    "config_name": "Dense-only",
    "weights": {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
    "metrics": {
      "ndcg@10": 0.6523,
      "mrr": 0.5814,
      "recall@50": 0.7123,
      "num_evaluated": 50
    },
    "elapsed_time": 45.2,
    "num_queries": 50
  },
  ...
]
```

### 2. `ablation_table.tex`

LaTeX格式表格（可直接用于论文）：

```latex
\begin{table}[htbp]
\centering
\caption{消融实验结果对比 (Ablation Study Results)}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{Dense} & \textbf{Sparse} & \textbf{KG} & \textbf{nDCG@10} & \textbf{MRR} & \textbf{Recall@50} \\
\midrule
Dense-only & $1.0$ & - & - & 0.6523 & 0.5814 & 0.7123 \\
Sparse-only & - & $1.0$ & - & 0.5987 & 0.5234 & 0.6890 \\
...
Full (Ours) & $0.4$ & $0.3$ & $0.3$ & 0.7580 & 0.6921 & 0.7856 \\
\bottomrule
\end{tabular}
\end{table}
```

### 3. `ablation_results.md`

Markdown格式表格（易读）：

```markdown
# 消融实验结果 (Ablation Study Results)

| Configuration | Dense (α) | Sparse (β) | KG (γ) | nDCG@10 | MRR | Recall@50 |
|---------------|-----------|------------|--------|---------|-----|-----------|
| Dense-only | 1.0 | - | - | 0.6523 | 0.5814 | 0.7123 |
| Sparse-only | - | 1.0 | - | 0.5987 | 0.5234 | 0.6890 |
| Full (Ours) | 0.4 | 0.3 | 0.3 | **0.7580** | **0.6921** | **0.7856** |

## 分析

- **最佳nDCG@10**: 0.7580
- **最佳MRR**: 0.6921
- **最佳Recall@50**: 0.7856
```

## 📈 预期结果分析

### 单组件性能

- **Dense-only**: 适合语义相似度检索，跨语言能力强
- **Sparse-only**: 适合词汇精确匹配，单语言效果好
- **KG-only**: 依赖实体链接质量，覆盖面较窄

### 组合效果

- **Dense+Sparse**: 互补效果明显，提升约10-15%
- **Dense+KG**: 知识增强对语义理解的提升
- **Sparse+KG**: 词汇+知识的协同效应

### 完整系统 (Ours)

- **三路融合**: 应显著优于任何两路组合
- **预期提升**: 相比最佳基线 +10-20%
- **论文贡献**: 证明KG增强的有效性

## 🔬 典型实验流程

### 1. 准备数据

```bash
# 确保索引已构建
ls artifacts/faiss_labse/
ls artifacts/whoosh_bm25/

# 确保评测数据存在
ls data/eval/clir_queries.jsonl
ls data/eval/qrels.tsv
```

### 2. 运行实验

```bash
# 运行完整消融实验（约30-60分钟）
python adaptive/ablation/run_ablation.py

# 实时查看日志
tail -f logs/app.log
```

### 3. 分析结果

```bash
# 查看JSON结果
cat artifacts/ablation_results/ablation_results.json | jq

# 查看Markdown表格
cat artifacts/ablation_results/ablation_results.md

# 复制LaTeX表格到论文
cat artifacts/ablation_results/ablation_table.tex
```

## 💡 高级用法

### 自定义实验配置

如需添加新的实验配置，修改 `run_ablation.py` 中的 `EXPERIMENT_CONFIGS`：

```python
EXPERIMENT_CONFIGS = {
    "Dense-only": (1.0, 0.0, 0.0),
    "Your-Config": (0.5, 0.3, 0.2),  # 自定义配置
    ...
}
```

### 并行执行

对于大规模查询集，可以修改代码启用多进程：

```python
from multiprocessing import Pool

def run_parallel(self):
    configs = list(self.EXPERIMENT_CONFIGS.items())
    with Pool(4) as pool:
        results = pool.starmap(
            self.run_single_config,
            [(name, *weights) for name, weights in configs]
        )
    return results
```

### 分阶段运行

如果实验耗时过长，可以分阶段运行：

```bash
# 只运行部分配置
python -c "
from adaptive.ablation.run_ablation import AblationExperiment
exp = AblationExperiment(...)
result1 = exp.run_single_config('Dense-only', 1.0, 0.0, 0.0)
result2 = exp.run_single_config('Sparse-only', 0.0, 1.0, 0.0)
"
```

## 📊 结果解读

### 指标说明

- **nDCG@10**: 前10个结果的排序质量（考虑相关性等级）
- **MRR**: 第一个相关文档的排名倒数（越高越好）
- **Recall@50**: 前50个结果中相关文档的召回率

### 论文写作建议

1. **Table**: 使用生成的LaTeX表格展示所有配置的性能
2. **Analysis**: 分析各组件的贡献和互补性
3. **Visualization**: 可以绘制柱状图或雷达图对比
4. **Statistical Test**: 对Full vs 最佳基线进行显著性检验

### 统计显著性检验

```python
from scipy import stats

# t-test
baseline_scores = [...]  # Dense+Sparse的nDCG@10得分
full_scores = [...]      # Full system的nDCG@10得分
t_stat, p_value = stats.ttest_rel(baseline_scores, full_scores)

print(f"p-value: {p_value:.4f}")
if p_value < 0.05:
    print("✓ 提升具有统计显著性 (p < 0.05)")
```

## ⚠️ 注意事项

1. **运行时间**: 完整实验可能需要30-60分钟（取决于查询数和索引大小）
2. **内存占用**: 确保有足够内存加载所有索引（建议 ≥16GB）
3. **Neo4j连接**: KG相关配置需要Neo4j正常运行
4. **结果稳定性**: 由于KG扩展的随机性，建议多次运行取平均

## 🔗 相关脚本

- `scripts/09_eval_clir.py`: 完整评测流程
- `retrieval/eval/metrics.py`: 评测指标计算
- `retrieval/rerank/fusion_rerank.py`: 融合排序逻辑

## 📚 参考文献

消融实验是验证系统各组件贡献的标准方法：

- Ablation studies help identify which components contribute most
- Each removed component shows its individual impact
- Combination studies reveal synergistic effects

---

**提示**: 首次运行建议使用较小的查询集进行测试，确认流程正常后再运行完整实验。
