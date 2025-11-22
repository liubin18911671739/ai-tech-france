# Dense检索测试指南

本指南用于测试刚完成的Dense检索功能。

## ✅ 已完成的文件

1. `retrieval/dense/build_faiss.py` - FAISS索引构建器
2. `retrieval/dense/dense_search.py` - Dense检索器
3. `scripts/06_index_dense.py` - 索引构建脚本

## 🧪 测试步骤

### 步骤1: 生成测试数据 (如果还没有)

```bash
# 生成Mock语料
python scripts/01_clean_corpus.py \
  --create-mock \
  --output data/raw \
  --mock-size 100

# 清洗语料
python scripts/01_clean_corpus.py \
  --input data/raw/corpus_fr.jsonl \
  --output data/cleaned \
  --lang fr

python scripts/01_clean_corpus.py \
  --input data/raw/corpus_zh.jsonl \
  --output data/cleaned \
  --lang zh
```

### 步骤2: 构建FAISS索引

```bash
# 使用脚本构建索引
python scripts/06_index_dense.py \
  --corpus-dir data/cleaned \
  --output models/faiss \
  --index-type IVF \
  --langs fr zh en

# 或者直接使用构建器
python retrieval/dense/build_faiss.py \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl data/cleaned/corpus_zh_cleaned.jsonl \
  --output models/faiss \
  --index-type Flat
```

**预期输出:**
```
FAISS索引构建器初始化: type=IVF, nlist=100
开始构建FAISS索引: 200 篇文档
使用LaBSE编码文档...
编码完成: shape=(200, 768)
向量维度: 768, 文档数: 200
使用IVF索引: nlist=100
训练IVF索引...
添加向量到索引...
索引构建完成! 总文档数: 200
FAISS索引已保存: models/faiss/faiss.index
文档ID映射已保存: models/faiss/doc_ids.pkl
元数据已保存: models/faiss/metadata.json
```

### 步骤3: 测试单个查询

```bash
# 中文查询
python retrieval/dense/dense_search.py \
  --index models/faiss \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl data/cleaned/corpus_zh_cleaned.jsonl \
  --query "法语语法学习" \
  --top-k 5
```

**预期输出:**
```
检索结果 (Top-5):
================================================================================

Rank 1: doc_zh_000001 (Score: 0.8542)
  标题: 语法 - Lesson 2
  语言: zh
  内容: 法语语法是规范法语语言使用的一套规则体系。它包括句法、词法和语音等方面。要掌握法语,必须深入理解这些语法规则。...

Rank 2: doc_fr_000003 (Score: 0.8213)
  标题: grammaire - Lesson 4
  语言: fr
  内容: La grammaire française est un ensemble de règles qui régissent la langue française...
```

### 步骤4: 测试跨语言检索

```bash
# 法语查询 -> 查找中文文档
python retrieval/dense/dense_search.py \
  --index models/faiss \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl data/cleaned/corpus_zh_cleaned.jsonl \
  --query "grammaire française" \
  --top-k 10
```

**验证点:**
- ✅ 返回结果中应包含中文文档
- ✅ 相关文档的分数应该较高 (>0.7)
- ✅ 语法相关的文档应排在前面

### 步骤5: 测试批量查询

创建查询文件 `test_queries.txt`:
```
法语语法学习
La prononciation française
French vocabulary
动词变位
```

运行批量检索:
```bash
python retrieval/dense/dense_search.py \
  --index models/faiss \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl data/cleaned/corpus_zh_cleaned.jsonl \
  --queries-file test_queries.txt \
  --top-k 5 \
  --output results/dense_batch_results.jsonl
```

### 步骤6: 交互模式测试

```bash
# 进入交互模式
python retrieval/dense/dense_search.py \
  --index models/faiss \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl data/cleaned/corpus_zh_cleaned.jsonl \
  --top-k 5
```

测试以下查询:
1. `法语学习` (中文)
2. `grammaire` (法语)
3. `pronunciation` (英语)
4. `动词` (中文)

### 步骤7: 性能测试

```bash
# 测试编码速度
python -c "
from retrieval.dense.labse_encoder import LaBSEEncoder
import time

encoder = LaBSEEncoder()
texts = ['test sentence'] * 100

start = time.time()
embeddings = encoder.encode(texts)
elapsed = time.time() - start

print(f'编码100个句子: {elapsed:.2f}秒')
print(f'平均速度: {100/elapsed:.1f} 句/秒')
print(f'向量维度: {embeddings.shape}')
"
```

**预期性能:**
- CPU: 5-15 句/秒
- GPU: 50-200 句/秒

## 🔍 验证检查清单

### 功能验证
- [ ] FAISS索引成功构建
- [ ] 索引文件正确保存 (faiss.index, doc_ids.pkl, metadata.json)
- [ ] 单查询检索返回正确结果
- [ ] 批量查询正常工作
- [ ] 跨语言检索有效 (法语查中文文档)

### 质量验证
- [ ] Top-1结果相关性高
- [ ] 跨语言查询相似度 > 0.7
- [ ] 无关查询分数低 < 0.5
- [ ] 结果排序符合预期

### 性能验证
- [ ] 索引构建时间合理 (100文档 < 2分钟)
- [ ] 单次查询响应快 < 1秒
- [ ] 批量查询高效

## 🐛 常见问题

### 问题1: FAISS索引训练失败
```
Error: IndexIVF needs training
```

**解决**: 确保文档数量足够 (至少 > nlist)
```bash
# 减小nlist或增加文档
python scripts/06_index_dense.py --nlist 10
```

### 问题2: 内存不足
```
MemoryError: Unable to allocate array
```

**解决**: 使用IVFPQ压缩索引
```bash
python scripts/06_index_dense.py --index-type IVFPQ
```

### 问题3: LaBSE下载慢
```
Downloading model...
```

**解决**: 使用镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/06_index_dense.py
```

### 问题4: 查询结果为空
```
检索完成: 返回 0 个结果
```

**解决**: 检查索引是否正确加载
```bash
# 查看索引统计
python -c "
from retrieval.dense.dense_search import DenseSearcher
from pathlib import Path

searcher = DenseSearcher(index_dir=Path('models/faiss'))
print(searcher.get_statistics())
"
```

## 📊 预期结果

### 索引统计
```json
{
  "index_type": "IVF",
  "num_docs": 300,
  "dimension": 768,
  "model": "sentence-transformers/LaBSE",
  "corpus_loaded": 300
}
```

### 检索示例
**查询**: "法语语法学习"

**Top-3结果**:
1. `doc_zh_000001` (0.854) - 法语语法是规范法语语言使用的一套规则体系...
2. `doc_fr_000045` (0.823) - La grammaire française est importante...
3. `doc_zh_000023` (0.798) - 学习法语语法需要掌握动词变位...

## ✅ 完成标准

Dense检索模块通过以下标准即视为完成:

1. ✅ 索引构建成功,无错误
2. ✅ 单查询检索返回合理结果
3. ✅ 跨语言检索相似度 > 0.7
4. ✅ Top-3结果相关性高
5. ✅ 性能满足要求 (查询 < 1秒)

## 🎯 下一步

Dense检索完成后,可以:

1. **继续 Phase 1** - 实现Sparse检索 (BM25)
2. **测试对比** - Dense vs Sparse性能对比
3. **集成UI** - 将Dense检索集成到Streamlit

---

**测试完成后,请在TODO.md中标记:** ✅ Dense检索测试通过
