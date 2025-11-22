#!/bin/bash
# 快速启动脚本 - 完整Pipeline演示

set -e  # 遇到错误立即退出

echo "=================================="
echo "跨语言知识服务系统 - 快速启动"
echo "=================================="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python版本
echo -e "${YELLOW}检查Python版本...${NC}"
python --version

# 步骤1: 创建Mock数据
echo -e "\n${GREEN}[1/10] 创建Mock数据${NC}"
python scripts/01_clean_corpus.py \
  --create-mock \
  --output data/raw \
  --mock-size 100

# 步骤2: 语料清洗
echo -e "\n${GREEN}[2/10] 清洗语料${NC}"
for lang in fr zh en; do
  echo "  清洗 ${lang} 语料..."
  python scripts/01_clean_corpus.py \
    --input data/raw/corpus_${lang}.jsonl \
    --output data/cleaned \
    --lang ${lang}
done

# 步骤3: 实体提取
echo -e "\n${GREEN}[3/10] 提取实体${NC}"
echo "  提取法语实体..."
python scripts/02_extract_entities.py \
  --lang fr \
  --input data/cleaned/corpus_fr_cleaned.jsonl \
  --output data/cleaned/entities_fr.jsonl

echo "  提取中文实体..."
python scripts/02_extract_entities.py \
  --lang zh \
  --input data/cleaned/corpus_zh_cleaned.jsonl \
  --output data/cleaned/entities_zh.jsonl

# 步骤4: 关系提取
echo -e "\n${GREEN}[4/10] 提取关系${NC}"
python scripts/03_extract_relations.py \
  --entities data/cleaned/entities_fr.jsonl \
  --corpus data/cleaned/corpus_fr_cleaned.jsonl \
  --output data/cleaned/relations.jsonl

# 步骤5: 构建知识图谱
echo -e "\n${GREEN}[5/10] 构建知识图谱 (Neo4j)${NC}"
echo "  请确保Neo4j已启动 (docker-compose up -d)"
read -p "  按Enter继续..."

python scripts/04_build_mkg.py \
  --entities-dir data/cleaned \
  --relations data/cleaned/relations.jsonl

# 步骤6: 跨语言对齐
echo -e "\n${GREEN}[6/10] 训练跨语言对齐${NC}"
python scripts/05_train_alignment.py \
  --seeds data/seeds/seed_align.tsv \
  --epochs 20 \
  --output models/alignment/mtranse.pt

# 步骤7: 构建Dense索引
echo -e "\n${GREEN}[7/10] 构建Dense索引 (FAISS)${NC}"
python scripts/06_index_dense.py \
  --corpus-dir data/cleaned \
  --output models/faiss

# 步骤8: 构建Sparse索引
echo -e "\n${GREEN}[8/10] 构建Sparse索引 (Whoosh)${NC}"
python scripts/07_index_sparse.py \
  --corpus-dir data/cleaned \
  --output models/whoosh

# 步骤9: 测试检索
echo -e "\n${GREEN}[9/10] 测试跨语言检索${NC}"
python scripts/08_run_kg_clir.py \
  --query "法语语法学习" \
  --lang zh \
  --top-k 10

python scripts/08_run_kg_clir.py \
  --query "grammaire française" \
  --lang fr \
  --top-k 10

# 步骤10: 运行评测
echo -e "\n${GREEN}[10/10] 运行评测${NC}"
python scripts/09_eval_clir.py \
  --queries data/eval/clir_queries.jsonl \
  --qrels data/eval/qrels.tsv

echo -e "\n${GREEN}=================================="
echo "✅ Pipeline执行完成!"
echo "==================================${NC}"

echo -e "\n${YELLOW}启动UI服务:${NC}"
echo "  Streamlit: streamlit run app/ui/streamlit_app.py"
echo "  FastAPI:   uvicorn app.api.main_api:app --reload"

echo -e "\n${YELLOW}查看结果:${NC}"
echo "  知识图谱: http://localhost:7474 (Neo4j Browser)"
echo "  Streamlit: http://localhost:8501"
echo "  API文档:   http://localhost:8000/docs"
