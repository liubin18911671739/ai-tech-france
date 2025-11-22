# 端到端KG-CLIR检索使用指南

`scripts/08_run_kg_clir.py` 是完整的端到端检索脚本，整合了Dense、Sparse和KG三路检索。

## 🚀 快速开始

### 1. 单个查询

```bash
# 基本用法
python scripts/08_run_kg_clir.py \
  --query "法语语法学习" \
  --lang zh \
  --top-k 10

# 指定索引路径
python scripts/08_run_kg_clir.py \
  --query "grammaire française" \
  --lang fr \
  --top-k 20 \
  --dense-index artifacts/faiss_labse \
  --sparse-index artifacts/whoosh_bm25

# 禁用KG增强（仅使用Dense+Sparse）
python scripts/08_run_kg_clir.py \
  --query "French grammar" \
  --lang en \
  --no-kg \
  --top-k 15

# 显示详细解释信息
python scripts/08_run_kg_clir.py \
  --query "法语动词变位" \
  --lang zh \
  --explain \
  --output results.json
```

### 2. 批量查询

```bash
# 从文件读取查询列表
python scripts/08_run_kg_clir.py \
  --queries-file data/eval/clir_queries.jsonl \
  --top-k 10 \
  --output artifacts/batch_results.json

# 自定义查询文件格式（JSONL）
cat > my_queries.jsonl << EOF
{"qid": "q1", "query": "法语语法", "lang": "zh"}
{"qid": "q2", "query": "grammaire française", "lang": "fr"}
{"qid": "q3", "query": "French verbs", "lang": "en"}
EOF

python scripts/08_run_kg_clir.py \
  --queries-file my_queries.jsonl \
  --top-k 20 \
  --output my_results.json
```

## ⚙️ 参数说明

### 检索参数

- `--query TEXT`: 单个查询文本
- `--lang {fr,zh,en,auto}`: 查询语言，默认 `auto` 自动检测
- `--queries-file PATH`: 批量查询文件（JSONL格式）
- `--top-k INT`: 返回结果数，默认 10
- `--explain`: 返回详细解释信息（包括各路得分贡献）

### 索引路径

- `--dense-index PATH`: Dense索引目录，默认 `artifacts/faiss_labse`
- `--sparse-index PATH`: Sparse索引目录，默认 `artifacts/whoosh_bm25`

### KG配置

- `--use-kg`: 启用KG增强（默认启用）
- `--no-kg`: 禁用KG增强
- `--max-hops INT`: KG扩展最大跳数，默认 2
- `--neo4j-uri TEXT`: Neo4j连接URI（默认从config读取）
- `--neo4j-user TEXT`: Neo4j用户名（默认从config读取）
- `--neo4j-password TEXT`: Neo4j密码（默认从config读取）

### 融合权重

- `--alpha FLOAT`: Dense检索权重（默认从config读取，通常0.4-0.5）
- `--beta FLOAT`: Sparse检索权重（默认从config读取，通常0.3-0.4）
- `--gamma FLOAT`: KG增强权重（默认从config读取，通常0.2-0.3）

### 输出

- `--output PATH`: 输出文件路径（JSON格式）

## 📊 输出格式

### 单个查询输出

终端打印格式：
```
================================================================================
检索结果 (Top 10)
================================================================================

[1] doc_id: doc_123
    Score: 0.8532
    Title: La grammaire française pour débutants
    Content: Ce guide présente les règles essentielles...
    Contributions: Dense=0.4123, Sparse=0.3201, KG=0.1208

[2] doc_id: doc_456
    ...
```

JSON输出格式（使用 `--output`）：
```json
[
  {
    "doc_id": "doc_123",
    "title": "La grammaire française pour débutants",
    "content": "Ce guide présente...",
    "lang": "fr",
    "final_score": 0.8532,
    "rank": 1,
    "score_contributions": {
      "dense": 0.4123,
      "sparse": 0.3201,
      "kg": 0.1208
    },
    "query": "法语语法学习",
    "query_lang": "zh",
    "fusion_config": {
      "alpha": 0.4,
      "beta": 0.3,
      "gamma": 0.3
    }
  }
]
```

### 批量查询输出

JSON格式：
```json
{
  "q1": [
    {
      "doc_id": "doc_123",
      "final_score": 0.8532,
      ...
    }
  ],
  "q2": [
    ...
  ]
}
```

## 🎯 使用场景

### 场景1: 对比实验（有无KG）

```bash
# 基线：Dense + Sparse
python scripts/08_run_kg_clir.py \
  --query "法语语法" \
  --no-kg \
  --output baseline_results.json

# 完整系统：Dense + Sparse + KG
python scripts/08_run_kg_clir.py \
  --query "法语语法" \
  --use-kg \
  --output full_results.json

# 对比nDCG@10等指标
```

### 场景2: 权重调优

```bash
# 实验1: 偏重Dense
python scripts/08_run_kg_clir.py \
  --query "grammaire" \
  --alpha 0.6 --beta 0.2 --gamma 0.2

# 实验2: 偏重KG
python scripts/08_run_kg_clir.py \
  --query "grammaire" \
  --alpha 0.3 --beta 0.2 --gamma 0.5

# 实验3: 均衡配置
python scripts/08_run_kg_clir.py \
  --query "grammaire" \
  --alpha 0.4 --beta 0.3 --gamma 0.3
```

### 场景3: 跨语言检索

```bash
# 中文查询 → 法语文档
python scripts/08_run_kg_clir.py \
  --query "法语动词变位规则" \
  --lang zh

# 法语查询 → 多语种文档
python scripts/08_run_kg_clir.py \
  --query "conjugaison des verbes" \
  --lang fr

# 英语查询 → 多语种文档
python scripts/08_run_kg_clir.py \
  --query "French verb conjugation" \
  --lang en
```

### 场景4: 论文实验

```bash
# 批量评测（生成论文数据）
python scripts/08_run_kg_clir.py \
  --queries-file data/eval/clir_queries.jsonl \
  --top-k 100 \
  --output artifacts/retrieval_results.json

# 然后用评测脚本计算指标
python scripts/09_eval_clir.py \
  --results artifacts/retrieval_results.json \
  --qrels data/eval/qrels.tsv
```

## 🔧 故障排除

### 问题1: Neo4j连接失败

```bash
# 检查Neo4j是否启动
docker ps | grep neo4j

# 如果未启动
docker-compose up -d neo4j

# 或临时禁用KG
python scripts/08_run_kg_clir.py --query "test" --no-kg
```

### 问题2: 索引文件不存在

```bash
# 检查索引是否存在
ls artifacts/faiss_labse/
ls artifacts/whoosh_bm25/

# 如果不存在，先构建索引
python scripts/06_index_dense.py --corpus-dir data/cleaned
python scripts/07_index_sparse.py --corpus-dir data/cleaned
```

### 问题3: 内存不足

```bash
# 减少top-k
python scripts/08_run_kg_clir.py --query "test" --top-k 10

# 或使用更小的索引（如IVF而不是Flat）
python scripts/06_index_dense.py --index-type IVF
```

## 📈 性能优化

### 批量查询优化

```python
# 如果需要处理大量查询，可以修改代码启用多进程
# 在 KGCLIRSystem.batch_search() 中添加：

from multiprocessing import Pool

def batch_search_parallel(self, queries, top_k=10, workers=4):
    with Pool(workers) as pool:
        results = pool.starmap(
            self.search,
            [(q["query"], q.get("lang", "auto"), top_k) for q in queries]
        )
    return {q["qid"]: r for q, r in zip(queries, results)}
```

### 索引加载优化

```python
# 预加载所有组件，避免重复初始化
system = KGCLIRSystem(...)  # 只初始化一次

# 重复使用
for query in many_queries:
    results = system.search(query)
```

## 🔗 相关脚本

- `scripts/06_index_dense.py`: 构建Dense索引
- `scripts/07_index_sparse.py`: 构建Sparse索引
- `scripts/09_eval_clir.py`: 完整评测流程
- `retrieval/dense/dense_search.py`: Dense检索模块
- `retrieval/sparse/sparse_search.py`: Sparse检索模块
- `retrieval/kg_expansion/`: KG扩展模块
- `retrieval/rerank/fusion_rerank.py`: 融合排序模块

## 📚 API文档

详见各模块的docstring：
```python
from retrieval.dense.dense_search import DenseSearcher
help(DenseSearcher.search)

from retrieval.sparse.sparse_search import SparseSearcher
help(SparseSearcher.search)

from retrieval.rerank.fusion_rerank import FusionReranker
help(FusionReranker.rerank)
```

---

**提示**: 首次运行时需要下载LaBSE模型（约500MB），会自动缓存到 `~/.cache/huggingface/`
