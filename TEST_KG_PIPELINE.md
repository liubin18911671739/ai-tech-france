# 知识图谱完善测试清单

## 文件清单

本次实现完成了知识图谱完善的5个核心文件:

### 1. `kg/neo4j_import/build_nodes_rels.py` (520行)
**核心功能:**
- `GraphBuilder` 类 - 图谱构建器
  - `add_concept(name, lang, type, properties)` - 添加概念节点
  - `add_relation(source, target, type, properties)` - 添加关系
  - `build_from_entities(entities_file, lang)` - 从实体文件构建节点
  - `build_from_relations(relations_file)` - 从关系文件构建边
  - `add_alignment_relations(alignment_file)` - 添加对齐关系
  - `export_nodes(output_file)` - 导出节点(JSONL)
  - `export_relations(output_file)` - 导出关系(JSONL)
  - `get_statistics()` - 获取图谱统计
  - `build_mock_graph()` - 构建Mock图谱

**CLI用法:**
```bash
# Mock模式
python kg/neo4j_import/build_nodes_rels.py \
  --mock \
  --output-nodes data/kg/nodes.jsonl \
  --output-rels data/kg/relations.jsonl

# 真实数据
python kg/neo4j_import/build_nodes_rels.py \
  --entities-fr data/entities/entities_fr.jsonl \
  --entities-zh data/entities/entities_zh.jsonl \
  --relations-fr data/relations/relations_fr.jsonl \
  --alignment data/seeds/seed_align.tsv \
  --output-nodes data/kg/nodes.jsonl \
  --output-rels data/kg/relations.jsonl
```

### 2. `kg/alignment/train_alignment.py` (380行)
**核心功能:**
- `AlignmentTrainer` 类 - 对齐训练器
  - `load_graph(nodes_file, relations_file)` - 加载图谱数据
  - `load_seed_alignment(alignment_file)` - 加载种子对齐
  - `train(triples, seed_alignments, epochs, batch_size)` - 训练MTransE模型
  - `_compute_alignment_loss(seed_pairs)` - 计算对齐损失
  - `predict_alignment(source_entity, target_lang, top_k)` - 预测对齐
  - `save_model(output_dir)` - 保存模型
  - `load_model(model_dir)` - 加载模型

**训练流程:**
1. 加载图谱三元组
2. 加载种子对齐对
3. 初始化MTransE模型
4. 训练嵌入 (TransE损失 + 对齐损失)
5. 保存实体/关系嵌入

**CLI用法:**
```bash
python kg/alignment/train_alignment.py \
  --nodes data/kg/nodes.jsonl \
  --relations data/kg/relations.jsonl \
  --alignment data/seeds/seed_align.tsv \
  --output artifacts/alignment_model \
  --embedding-dim 100 \
  --epochs 100 \
  --batch-size 128 \
  --learning-rate 0.01
```

### 3. `kg/stats/graph_stats.py` (420行)
**核心功能:**
- `GraphStatistics` 类 - 图谱统计分析器
  - `load_graph(nodes_file, relations_file)` - 加载图谱
  - `compute_basic_stats()` - 基本统计(节点数、关系数、按语言/类型分布)
  - `compute_degree_stats()` - 度数统计(入度、出度、孤立节点)
  - `get_top_nodes(top_k)` - Top-K节点
  - `compute_connectivity_stats()` - 连通性统计(连通分量、BFS)
  - `compute_alignment_stats()` - 对齐统计(对齐数、覆盖率、语言对)
  - `generate_report()` - 生成Markdown报告
  - `export_report(output_file)` - 导出报告

**报告内容:**
1. 基本统计 (节点数、关系数、按语言/类型分布)
2. 度数统计 (平均入度、最大入度、孤立节点)
3. Top-10节点 (按总度数排序)
4. 连通性统计 (连通分量数、最大/最小分量)
5. 对齐统计 (对齐数、覆盖率、语言对分布)

**CLI用法:**
```bash
python kg/stats/graph_stats.py \
  --nodes data/kg/nodes.jsonl \
  --relations data/kg/relations.jsonl \
  --output artifacts/graph_stats.md
```

### 4. `scripts/04_build_mkg.py` (220行)
**核心功能:**
- 完整图谱构建流程
  - Step 1: 构建节点和关系 (GraphBuilder)
  - Step 2: 导入Neo4j (Neo4jImporter, 可选)
  - Step 3: 生成统计报告 (GraphStatistics)
  - Step 4: 总结

**CLI用法:**
```bash
# Mock模式 (快速测试)
python scripts/04_build_mkg.py --mock

# Mock + Neo4j导入
python scripts/04_build_mkg.py --mock --import-neo4j

# 真实数据
python scripts/04_build_mkg.py \
  --entities-fr data/entities/entities_fr.jsonl \
  --entities-zh data/entities/entities_zh.jsonl \
  --relations-fr data/relations/relations_fr.jsonl \
  --alignment data/seeds/seed_align.tsv \
  --import-neo4j
```

### 5. `scripts/05_train_alignment.py` (260行)
**核心功能:**
- 完整对齐训练流程
  - Step 0: 创建种子对齐 (可选)
  - Step 1: 检查输入文件
  - Step 2: 初始化训练器
  - Step 3: 加载数据
  - Step 4: 训练模型
  - Step 5: 保存模型
  - Bonus: 测试对齐预测

**CLI用法:**
```bash
# 基本训练
python scripts/05_train_alignment.py

# 自定义参数
python scripts/05_train_alignment.py \
  --nodes data/kg/nodes.jsonl \
  --relations data/kg/relations.jsonl \
  --alignment data/seeds/seed_align.tsv \
  --output artifacts/alignment_model \
  --embedding-dim 100 \
  --epochs 100 \
  --learning-rate 0.01

# 创建Mock种子对齐
python scripts/05_train_alignment.py --create-seeds
```

---

## 测试场景

### 场景1: 构建Mock图谱
```bash
# 快速构建测试图谱
python scripts/04_build_mkg.py --mock

# 预期输出:
# [Step 1/4] 构建节点和关系
# Mock图谱构建完成: 12 个节点, 18 条边
# 导出 12 个节点: data/kg/nodes.jsonl
# 导出 18 条关系: data/kg/relations.jsonl
#
# [Step 3/4] 生成统计报告
# 统计报告已生成: artifacts/graph_stats.md
```

### 场景2: 导入Neo4j
```bash
# 确保Neo4j运行
docker-compose up -d neo4j

# 构建并导入
python scripts/04_build_mkg.py --mock --import-neo4j

# 预期输出:
# [Step 2/4] 导入Neo4j
# 清空现有数据...
# 创建约束...
# 导入节点...
# 导入 12 个概念
# 导入关系...
# 导入 18 条关系
# Neo4j导入完成:
#   节点数: 12
#   关系数: 18

# 验证导入
# 打开 http://localhost:7474
# 运行查询: MATCH (n) RETURN n LIMIT 25
```

### 场景3: 训练对齐模型
```bash
# 先构建图谱
python scripts/04_build_mkg.py --mock

# 训练对齐
python scripts/05_train_alignment.py --create-seeds --epochs 50

# 预期输出:
# [Step 0/5] 创建种子对齐
# 创建 8 对种子对齐
#
# [Step 3/5] 加载图谱数据
# 加载 12 个实体
# 加载 18 个三元组, 3 种关系
# 加载 8 对种子对齐
#
# [Step 4/5] 训练模型
# Epoch 10/50, Loss: 1.2345
# Epoch 20/50, Loss: 0.9876
# ...
# 训练完成!
#   初始损失: 1.5432
#   最终损失: 0.7654
#   损失下降: 50.4%
#
# [Bonus] 测试对齐预测
# fr_CONCEPT_000000 -> zh:
#   zh_CONCEPT_000000 (机器学习): 0.8523
```

### 场景4: 生成图谱统计报告
```bash
# 生成报告
python kg/stats/graph_stats.py \
  --nodes data/kg/nodes.jsonl \
  --relations data/kg/relations.jsonl \
  --output artifacts/graph_stats.md

# 查看报告
cat artifacts/graph_stats.md

# 预期内容:
# # 知识图谱统计报告
# 
# ## 1. 基本统计
# - 总节点数: 12
# - 总关系数: 18
#
# ### 节点按语言分布
# | 语言 | 节点数 |
# |------|--------|
# | fr   | 4      |
# | zh   | 4      |
# | en   | 4      |
# ...
```

### 场景5: 端到端流程
```bash
# 完整流程
echo "Step 1: 构建图谱"
python scripts/04_build_mkg.py --mock --import-neo4j

echo "Step 2: 训练对齐"
python scripts/05_train_alignment.py --create-seeds --epochs 50

echo "Step 3: 查看统计"
cat artifacts/graph_stats.md

echo "完成! ✅"
```

---

## 数据格式

### 节点文件格式 (`nodes.jsonl`)
```json
{
  "id": "fr_CONCEPT_000000",
  "name": "apprentissage automatique",
  "lang": "fr",
  "type": "CONCEPT",
  "properties": {}
}
```

### 关系文件格式 (`relations.jsonl`)
```json
{
  "source": "fr_CONCEPT_000001",
  "target": "fr_CONCEPT_000000",
  "type": "IS_A",
  "properties": {}
}
```

### 种子对齐格式 (`seed_align.tsv`)
```tsv
# 种子对齐数据
# 格式: entity1	entity2

apprentissage automatique	机器学习
machine learning	机器学习
apprentissage profond	深度学习
```

### 对齐模型输出
```
artifacts/alignment_model/
├── entity_embeddings.npy      # 实体嵌入矩阵
├── relation_embeddings.npy    # 关系嵌入矩阵
├── entity2id.json             # 实体->ID映射
└── relation2id.json           # 关系->ID映射
```

---

## 与其他模块集成

### 1. 与Neo4j导入集成
```python
from kg.neo4j_import.build_nodes_rels import GraphBuilder
from kg.neo4j_import.import_to_neo4j import Neo4jImporter

# 构建图谱
builder = GraphBuilder()
builder.build_mock_graph()
builder.export_nodes("data/kg/nodes.jsonl")
builder.export_relations("data/kg/relations.jsonl")

# 导入Neo4j
importer = Neo4jImporter()
importer.import_from_files(
    concepts_file="data/kg/nodes.jsonl",
    relations_file="data/kg/relations.jsonl"
)
```

### 2. 与实体链接集成
```python
from kg.neo4j_import.build_nodes_rels import GraphBuilder
from retrieval.kg_expansion.entity_linking import EntityLinker

# 构建图谱并导入Neo4j
# ...

# 实体链接
linker = EntityLinker()
entities = [{"text": "apprentissage automatique", "type": "CONCEPT"}]
linked = linker.link_entities(entities, lang="fr")

# linked 返回图谱中的节点ID
```

### 3. 与对齐预测集成
```python
from kg.alignment.train_alignment import AlignmentTrainer

# 训练对齐模型
trainer = AlignmentTrainer()
# ... 训练 ...
trainer.save_model("artifacts/alignment_model")

# 预测对齐
predictions = trainer.predict_alignment(
    source_entity="fr_CONCEPT_000000",
    target_lang="zh",
    top_k=5
)
# 返回: [("zh_CONCEPT_000000", 0.85), ...]
```

---

## 常见问题

### Q1: 如何添加新的实体类型?
在`build_nodes_rels.py`的`add_concept`中支持任意类型:
```python
builder.add_concept("Python", "en", concept_type="TECHNOLOGY")
```

### Q2: 如何自定义关系类型?
在`add_relation`中使用自定义类型:
```python
builder.add_relation(source_id, target_id, relation_type="DEPENDS_ON")
```

### Q3: 对齐训练收敛慢怎么办?
调整训练参数:
```bash
python scripts/05_train_alignment.py \
  --epochs 200 \
  --learning-rate 0.001 \
  --batch-size 64
```

### Q4: 如何从真实NER结果构建图谱?
准备实体文件格式:
```json
{
  "doc_id": "doc_001",
  "entities": [
    {"text": "machine learning", "type": "CONCEPT"},
    {"text": "neural networks", "type": "CONCEPT"}
  ]
}
```

然后运行:
```bash
python scripts/04_build_mkg.py \
  --entities-en data/entities/entities_en.jsonl
```

---

## Mock图谱说明

Mock图谱包含:

**节点 (12个):**
- 法语: apprentissage automatique, apprentissage profond, réseaux de neurones, réseaux convolutifs
- 中文: 机器学习, 深度学习, 神经网络, 卷积神经网络
- 英语: machine learning, deep learning, neural networks, convolutional neural networks

**关系 (18条):**
- 层级关系 (IS_A): 深度学习 -> 机器学习, 卷积神经网络 -> 深度学习
- 相关关系 (RELATED_TO): 神经网络 <-> 机器学习
- 对齐关系 (ALIGNED_WITH): 跨语言对齐

**用途:**
- 快速测试图谱构建
- 验证Neo4j导入
- 训练对齐模型
- 测试KG增强检索

---

## 下一步

### 建议后续工作:

1. **运行完整图谱构建** (推荐)
   ```bash
   python scripts/04_build_mkg.py --mock --import-neo4j
   python scripts/05_train_alignment.py --create-seeds
   ```

2. **实现数据Pipeline** (任务组B)
   - `scripts/02_extract_entities.py` - 批量实体提取
   - `scripts/03_extract_relations.py` - 批量关系提取

3. **端到端检索脚本** (任务组C)
   - `scripts/08_run_kg_clir.py` - KG-CLIR检索

4. **运行评测** (立即可用)
   ```bash
   python scripts/09_eval_clir.py --use-kg
   ```

---

## 总结

✅ **已完成的功能:**
- 图谱节点和关系构建 (从实体/关系文件或Mock)
- 跨语言对齐训练 (MTransE算法)
- 图谱统计分析 (基本统计、度数、连通性、对齐)
- 完整构建流程脚本 (构建 + 导入 + 统计)
- 完整训练流程脚本 (数据加载 + 训练 + 测试)

🎯 **任务组A完成度: 100%**

📊 **图谱功能就绪:**
- 节点构建 ✅
- 关系构建 ✅
- Neo4j导入 ✅
- 对齐训练 ✅
- 统计分析 ✅

🚀 **可立即与KG增强检索集成!**
