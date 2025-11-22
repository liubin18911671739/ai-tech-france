# 完整项目文件清单

本文档列出所有需要生成的文件及其功能描述。

## 已生成文件 ✓

### 根目录
- [x] requirements.txt - 依赖包
- [x] config.py - 全局配置
- [x] logger.py - 日志管理
- [x] README.md - 项目说明
- [x] docker-compose.yml - Neo4j容器
- [x] .env.example - 环境变量模板

### kg/ - 知识图谱
- [x] kg/__init__.py
- [x] kg/ontology/flo_schema.json - FLO本体
- [x] kg/extraction/__init__.py
- [x] kg/extraction/ner_fr.py - 法语NER
- [x] kg/extraction/ner_zh.py - 中文NER
- [x] kg/extraction/relation_extract.py - 关系抽取
- [x] kg/alignment/__init__.py
- [x] kg/alignment/mtranse.py - MTransE对齐

### retrieval/ - 检索
- [x] retrieval/__init__.py
- [x] retrieval/dense/__init__.py
- [x] retrieval/dense/labse_encoder.py - LaBSE编码器

## 待生成文件清单

### kg/ 继续
- [ ] kg/alignment/train_alignment.py - 对齐训练脚本
- [ ] kg/neo4j_import/__init__.py
- [ ] kg/neo4j_import/build_nodes_rels.py - 构建节点关系
- [ ] kg/neo4j_import/import_to_neo4j.py - 导入Neo4j
- [ ] kg/stats/__init__.py
- [ ] kg/stats/graph_stats.py - 图谱统计

### retrieval/ 继续
- [ ] retrieval/dense/build_faiss.py - 构建FAISS索引
- [ ] retrieval/dense/dense_search.py - Dense检索
- [ ] retrieval/sparse/__init__.py
- [ ] retrieval/sparse/build_whoosh.py - 构建Whoosh索引
- [ ] retrieval/sparse/sparse_search.py - BM25检索
- [ ] retrieval/kg_expansion/__init__.py
- [ ] retrieval/kg_expansion/entity_linking.py - 实体链接
- [ ] retrieval/kg_expansion/hop_expand.py - N-hop扩展
- [ ] retrieval/kg_expansion/kg_path_score.py - 路径评分
- [ ] retrieval/rerank/__init__.py
- [ ] retrieval/rerank/fusion_rerank.py - 融合重排
- [ ] retrieval/eval/__init__.py
- [ ] retrieval/eval/metrics.py - 评测指标
- [ ] retrieval/eval/run_eval.py - 运行评测

### adaptive/ - 自适应学习
- [ ] adaptive/__init__.py
- [ ] adaptive/learner_model/__init__.py
- [ ] adaptive/learner_model/mastery.py - 掌握度模型
- [ ] adaptive/learner_model/profile.py - 学习画像
- [ ] adaptive/path_reco/__init__.py
- [ ] adaptive/path_reco/recommend_path.py - 路径推荐
- [ ] adaptive/rag_tutor/__init__.py
- [ ] adaptive/rag_tutor/rag_retrieve.py - RAG检索
- [ ] adaptive/rag_tutor/generate_exercise.py - 生成练习
- [ ] adaptive/ablation/__init__.py
- [ ] adaptive/ablation/run_ablation.py - 消融实验

### app/ - 应用层
- [ ] app/__init__.py
- [ ] app/api/__init__.py
- [ ] app/api/main_api.py - FastAPI服务
- [ ] app/ui/__init__.py
- [ ] app/ui/streamlit_app.py - Streamlit界面

### scripts/ - 执行脚本
- [ ] scripts/__init__.py
- [ ] scripts/01_clean_corpus.py - 语料清洗
- [ ] scripts/02_extract_entities.py - 实体提取
- [ ] scripts/03_extract_relations.py - 关系提取
- [ ] scripts/04_build_mkg.py - 构建知识图谱
- [ ] scripts/05_train_alignment.py - 训练对齐
- [ ] scripts/06_index_dense.py - 构建Dense索引
- [ ] scripts/07_index_sparse.py - 构建Sparse索引
- [ ] scripts/08_run_kg_clir.py - 运行KG-CLIR
- [ ] scripts/09_eval_clir.py - 评测CLIR
- [ ] scripts/10_run_pilot_analysis.py - 试点分析

### data/ - 数据模板
- [ ] data/README.md - 数据说明
- [ ] data/raw/sample_corpus_fr.jsonl - 法语样例
- [ ] data/raw/sample_corpus_zh.jsonl - 中文样例
- [ ] data/seeds/seed_align.tsv - 对齐种子
- [ ] data/eval/clir_queries.jsonl - 评测查询
- [ ] data/eval/qrels.tsv - 相关性标注

## 生成顺序

建议按以下顺序继续生成:

1. **Phase 1**: 完成 kg/ 模块 (对齐训练 + Neo4j导入)
2. **Phase 2**: 完成 retrieval/ 模块 (Dense + Sparse + KG扩展 + 融合)
3. **Phase 3**: 完成 adaptive/ 模块 (学习支持)
4. **Phase 4**: 完成 app/ 模块 (API + UI)
5. **Phase 5**: 完成 scripts/ 模块 (10个执行脚本)
6. **Phase 6**: 完成 data/ 模板文件

## 使用说明

每个文件都包含:
- 详细注释
- 类型标注
- CLI参数支持
- 日志输出
- 错误处理
- 单元测试或示例代码

所有模块可独立运行和测试。
