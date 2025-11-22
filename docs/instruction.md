你是“高校图书馆跨语言知识服务 + 多语种知识图谱 + CLIR + 自适应学习支持”项目的首席工程师与图情学研究者，为我生成一个完整可复现的 Python 项目。
我需要你按文件逐个输出代码，并确保整体能运行、模块之间接口一致、具备清晰 README 与可复现实验脚本。

【Repo 指令】（若我把这段放到 .github/copilot-instructions.md，你在整个仓库里都要遵守）

代码语言：Python 3.10+，遵循 PEP8，类型注解尽量齐全。

每个模块必须：config.py（集中参数）、logger.py（统一日志）、__init__.py。

所有脚本使用 CLI 参数（argparse），并给出默认值与帮助信息。

所有外部依赖写入 requirements.txt，标出推荐版本。

不能依赖私有数据：所有数据都用 data/ 下的占位符样例或 mock 生成逻辑。

重要算法给出清晰注释与数学对应（尤其是 KG 路径重排、融合排序）。

任何需要钥匙/账户的服务都要用环境变量占位（比如 LLM API Key）。

输出必须文件分批生成：先给 repo 总览与步骤，再逐文件输出，禁止把所有文件混在一起。

生成代码后，自带最小可运行 demo（命令行 + Streamlit UI）。

【项目目标（论文对应）】

实现一个面向大学图书馆跨语言知识服务的系统，支持：
A. 多语种法语学习知识图谱（mKG）构建：

语料清洗（zh/fr/en）

法语侧 CamemBERT NER，中文侧 RoBERTa/HanLP NER

规则+LLM 辅助关系抽取

FLO 本体约束入库 Neo4j
B. 跨语言图谱对齐：

读取 data/seeds/seed_align.tsv 作为对齐种子

MTransE 训练（可用简化实现）→ 输出对齐表
C. 图谱增强跨语言检索 KG-CLIR：

Dense CLIR：LaBSE 统一向量空间 + FAISS 召回

Sparse：Whoosh BM25 召回

Query-to-KG linking（用 label/alias + LaBSE 相似度）

n-hop 邻域扩展（Neo4j 查询）

融合排序：
Score = α·dense + β·bm25 + γ·kg_path

返回证据路径解释
D. 学习分析与自适应支持：

学习日志 mock + 画像

概念掌握度估计（简单 mastery 模型即可）

学习路径推荐（基于 prerequisite 关系拓扑排序/最短链）

RAG 练习生成（默认本地 LLM stub，可替换）
E. Demo：

CLI：支持批量构图、对齐、索引、检索评测

Streamlit：提供跨语种检索 + 图谱路径可视化 + 推荐学习路径

【仓库结构（必须严格按这个来生成）】
clir-french-mkg-lib/
├─ data/
│  ├─ raw/              # 原始语料占位符
│  ├─ cleaned/          # 清洗后语料
│  ├─ parallel/         # OPUS 平行语料占位符
│  ├─ seeds/            # seed_align.tsv
│  └─ eval/             # clir_queries.jsonl, qrels.tsv
├─ kg/
│  ├─ ontology/
│  │   └─ flo_schema.json
│  ├─ extraction/
│  │   ├─ ner_fr.py
│  │   ├─ ner_zh.py
│  │   └─ relation_extract.py
│  ├─ alignment/
│  │   ├─ mtranse.py
│  │   └─ train_alignment.py
│  ├─ neo4j_import/
│  │   ├─ build_nodes_rels.py
│  │   └─ import_to_neo4j.py
│  └─ stats/
│      └─ graph_stats.py
├─ retrieval/
│  ├─ dense/
│  │   ├─ labse_encoder.py
│  │   ├─ build_faiss.py
│  │   └─ dense_search.py
│  ├─ sparse/
│  │   ├─ build_whoosh.py
│  │   └─ sparse_search.py
│  ├─ kg_expansion/
│  │   ├─ entity_linking.py
│  │   ├─ hop_expand.py
│  │   └─ kg_path_score.py
│  ├─ rerank/
│  │   └─ fusion_rerank.py
│  └─ eval/
│      ├─ metrics.py
│      └─ run_eval.py
├─ adaptive/
│  ├─ learner_model/
│  │   ├─ mastery.py
│  │   └─ profile.py
│  ├─ path_reco/
│  │   └─ recommend_path.py
│  ├─ rag_tutor/
│  │   ├─ rag_retrieve.py
│  │   └─ generate_exercise.py
│  └─ ablation/
│      └─ run_ablation.py
├─ app/
│  ├─ api/
│  │   └─ main_api.py   # FastAPI
│  └─ ui/
│      └─ streamlit_app.py
├─ scripts/
│  ├─ 01_clean_corpus.py
│  ├─ 02_extract_entities.py
│  ├─ 03_extract_relations.py
│  ├─ 04_build_mkg.py
│  ├─ 05_train_alignment.py
│  ├─ 06_index_dense.py
│  ├─ 07_index_sparse.py
│  ├─ 08_run_kg_clir.py
│  ├─ 09_eval_clir.py
│  └─ 10_run_pilot_analysis.py
├─ config.py
├─ logger.py
├─ requirements.txt
└─ README.md

【详细实现要求】

每个模块给出完整实现，能在 mock 数据上跑通。

Neo4j：

连接参数从 .env 或环境变量读（NEO4J_URI/USER/PASS）。

给出最小可运行的 docker-compose.yml（可选，但加分）。

FAISS & Whoosh：

Dense/Sparse 索引都能在 data/cleaned/ mock 文档上构建并查询。

CLIR Eval：

data/eval/clir_queries.jsonl：每行 {qid, lang, query, gold_concepts[]}

qrels.tsv：qid doc_id relevance(0/1/2)

输出 nDCG@10, MRR, Recall@50

自适应：

adaptive/ 里提供 mock 学习日志生成器。

路径推荐必须返回“概念链 + 推荐资源 doc_id 列表”。

RAG 练习生成：

先检索证据，再 prompt LLM（提供默认本地 stub，不依赖外部 API）。

Streamlit UI：

主页输入查询（可选语种）→ 返回 Top-k 资源 + KG 路径解释 + 学习路径推荐。

【输出方式（你必须按这个顺序回答我）】

Step 0：给出整体实现说明 + 运行步骤总览。
Step 1：输出 requirements.txt, config.py, logger.py, README.md
Step 2：按目录顺序逐文件输出代码（每次输出一个文件）：

先 kg/ 全部

再 retrieval/ 全部

再 adaptive/ 全部

再 app/

最后 scripts/
每个文件输出格式：

# filename: path/to/file.py
<full code here>


Step 3：提供一组可跑通的示例命令：

构图、对齐、索引、检索、评测、启动 UI。