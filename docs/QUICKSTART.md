# 快速测试指南

本指南帮助你测试当前已实现的功能。

## 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量(可选)

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件
# 至少需要配置 Neo4j (如果要使用图数据库)
```

---

## 功能测试

### ✅ 测试1: 配置系统

```bash
# 测试配置加载
python config.py
```

**预期输出:**
```
=== 配置信息 ===
项目根目录: /path/to/ai-tech-france
数据目录: /path/to/ai-tech-france/data
Neo4j URI: bolt://localhost:7687
LaBSE 模型: sentence-transformers/LaBSE
融合权重: α=0.4, β=0.3, γ=0.3
```

### ✅ 测试2: 日志系统

```bash
# 测试日志
python logger.py
```

**预期输出:** 彩色日志输出

### ✅ 测试3: Mock数据生成

```bash
# 生成3种语言的Mock语料
python scripts/01_clean_corpus.py \
  --create-mock \
  --output data/raw \
  --mock-size 50
```

**预期结果:**
- `data/raw/corpus_fr.jsonl` (法语)
- `data/raw/corpus_zh.jsonl` (中文)
- `data/raw/corpus_en.jsonl` (英语)

**验证:**
```bash
# 查看生成的文件
ls -lh data/raw/
head -n 2 data/raw/corpus_fr.jsonl | python -m json.tool
```

### ✅ 测试4: 语料清洗

```bash
# 清洗法语语料
python scripts/01_clean_corpus.py \
  --input data/raw/corpus_fr.jsonl \
  --output data/cleaned \
  --lang fr
```

**预期结果:**
- `data/cleaned/corpus_fr_cleaned.jsonl`
- 日志显示有效/无效文档数

### ✅ 测试5: LaBSE跨语言编码

```bash
# 测试跨语言相似度
python retrieval/dense/labse_encoder.py
```

**预期输出:**
```
跨语言测试: ['La grammaire française est importante', '法语语法很重要', ...]
[0] x [1]: 0.8542  # 法语与中文相似度高
[0] x [2]: 0.8213  # 法语与英语相似度高
[0] x [3]: 0.2145  # 与无关文本相似度低
```

**自定义测试:**
```bash
# 测试自己的句子
python retrieval/dense/labse_encoder.py \
  --text "Bonjour" "你好" "Hello" "Goodbye" \
  --query "问候语"
```

### ✅ 测试6: 法语NER

```bash
# 测试单个句子
python kg/extraction/ner_fr.py \
  --text "La grammaire française est importante pour apprendre la syntaxe et le vocabulaire."
```

**预期输出:**
```
提取实体: [
  {"entity": "grammaire", "type": "Concept", "score": 0.95, ...},
  {"entity": "syntaxe", "type": "Concept", "score": 0.92, ...}
]
```

**批量测试:**
```bash
# 对清洗后的语料批量NER
python kg/extraction/ner_fr.py \
  --input data/cleaned/corpus_fr_cleaned.jsonl \
  --output data/cleaned/entities_fr.jsonl
```

### ✅ 测试7: 中文NER

```bash
# 测试单个句子
python kg/extraction/ner_zh.py \
  --text "学习法语语法需要掌握动词变位和句法结构。"
```

**预期输出:**
```
提取实体: [
  {"entity": "法语语法", "type": "Concept", "score": 1.0, ...},
  {"entity": "动词变位", "type": "Concept", "score": 1.0, ...}
]
```

### ✅ 测试8: 关系抽取

```bash
# 先生成实体文件(如果还没有)
python kg/extraction/ner_fr.py \
  --input data/cleaned/corpus_fr_cleaned.jsonl \
  --output data/cleaned/entities_fr.jsonl

# 提取关系
python kg/extraction/relation_extract.py \
  --entities data/cleaned/entities_fr.jsonl \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl \
  --output data/cleaned/relations_fr.jsonl
```

**预期输出:**
```
开始提取关系...
已处理 50 篇文档,提取 23 个关系
关系提取完成,共 23 个关系
```

**查看结果:**
```bash
head -n 3 data/cleaned/relations_fr.jsonl | python -m json.tool
```

### ✅ 测试9: MTransE对齐

```bash
# 测试MTransE训练
python kg/alignment/mtranse.py
```

**预期输出:**
```
MTransE初始化: entities=5, relations=2, dim=128
开始训练: epochs=20, batch_size=128
Epoch 10/20, Loss: 0.5234
Epoch 20/20, Loss: 0.3145
训练完成!
对齐预测: {'语法_zh': [('grammaire_fr', 0.245), ...]}
```

### ✅ 测试10: Streamlit UI

```bash
# 启动Web界面
streamlit run app/ui/streamlit_app.py
```

**访问:** http://localhost:8501

**测试功能:**
1. 跨语言检索 - 输入查询,查看结果
2. 知识图谱 - 浏览概念关系
3. 学习路径 - 查看推荐路径
4. 参数调整 - 调整融合权重

---

## 性能基准

### LaBSE编码速度

```bash
# 测试100个句子的编码时间
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
"
```

**预期:** 
- CPU: 10-20 句/秒
- GPU: 100-200 句/秒

### NER处理速度

```bash
# 测试50篇文档的NER时间
time python kg/extraction/ner_fr.py \
  --input data/cleaned/corpus_fr_cleaned.jsonl \
  --output /tmp/test_ner.jsonl
```

---

## 故障排查

### 问题1: 依赖安装失败

```bash
# 如果torch安装失败,先安装torch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 如果transformers版本冲突
pip install transformers==4.36.0 --force-reinstall

# 如果HanLP安装失败
pip install hanlp==2.1.0b54 --no-deps
pip install toposort pynvml alnlp penman isort termcolor dill
```

### 问题2: 模型下载缓慢

```bash
# 设置HuggingFace镜像(中国用户)
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
# 然后修改 config.py 中的模型路径
```

### 问题3: 内存不足

```bash
# 减小batch_size
# 编辑 config.py:
# MTRANSE_BATCH_SIZE = 32  # 改小
# 或在运行时指定 --batch-size 32
```

### 问题4: CUDA不可用

```bash
# 检查CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回False,系统会自动使用CPU
# 性能会慢一些但功能正常
```

---

## 下一步

完成测试后,你可以:

1. **查看 PROGRESS.md** - 了解项目进度
2. **查看 FILE_CHECKLIST.md** - 查看完整文件清单
3. **运行 run_demo.sh** - 运行完整演示(需要先生成剩余文件)

---

## 获取帮助

如果遇到问题:
1. 检查日志输出
2. 查看 logs/ 目录下的日志文件
3. 参考 README.md 中的详细说明

---

**Happy Testing! 🚀**
