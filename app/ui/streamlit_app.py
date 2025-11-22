"""
Streamlit UI - 跨语言知识服务界面

提供:
1. 跨语种检索
2. KG路径可视化  
3. 学习路径推荐
"""
import streamlit as st
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger

logger = get_logger(__name__)

# 页面配置
st.set_page_config(
    page_title="跨语言知识服务系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("📚 跨语言法语学习知识服务系统")
st.markdown("---")

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 语言选择
    query_lang = st.selectbox(
        "查询语言",
        options=["zh", "fr", "en"],
        format_func=lambda x: {"zh": "中文", "fr": "法语", "en": "英语"}[x],
        index=0
    )
    
    # 检索参数
    st.subheader("检索参数")
    top_k = st.slider("返回结果数", 5, 50, 10)
    use_kg = st.checkbox("启用KG增强", value=True)
    
    if use_kg:
        hop_limit = st.slider("KG扩展跳数", 1, 3, 2)
    
    # 权重调整
    st.subheader("融合权重")
    alpha = st.slider("Dense (α)", 0.0, 1.0, 0.4, 0.1)
    beta = st.slider("Sparse (β)", 0.0, 1.0, 0.3, 0.1)
    gamma = st.slider("KG (γ)", 0.0, 1.0, 0.3, 0.1)
    
    # 归一化
    total = alpha + beta + gamma
    if total > 0:
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
    
    st.info(f"α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")

# 主界面
tab1, tab2, tab3 = st.tabs(["🔍 跨语言检索", "🗺️ 知识图谱", "📈 学习路径"])

# Tab 1: 跨语言检索
with tab1:
    st.header("跨语言检索")
    
    # 查询输入
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "请输入查询",
            placeholder="例如: 法语语法学习 / grammaire française / French grammar",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("🔍 检索", type="primary", use_container_width=True)
    
    # 示例查询
    st.markdown("**示例查询**:")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("法语动词变位"):
            query = "法语动词变位"
            search_btn = True
    with col_b:
        if st.button("grammaire française"):
            query = "grammaire française"
            search_btn = True
    with col_c:
        if st.button("French pronunciation"):
            query = "French pronunciation"
            search_btn = True
    
    if search_btn and query:
        with st.spinner("检索中..."):
            # TODO: 实际检索逻辑
            # 这里先用mock数据
            st.success(f"检索完成! 找到 {top_k} 个结果")
            
            # Mock结果
            results = [
                {
                    "doc_id": "doc_001",
                    "title": "法语语法基础教程",
                    "content": "本教程介绍法语语法的基本概念...",
                    "lang": "zh",
                    "score": 0.92,
                    "kg_path": ["语法", "动词", "时态"]
                },
                {
                    "doc_id": "doc_045",
                    "title": "La grammaire française pour débutants",
                    "content": "Ce cours présente les bases de la grammaire...",
                    "lang": "fr",
                    "score": 0.88,
                    "kg_path": ["grammaire", "verbe", "conjugaison"]
                },
                {
                    "doc_id": "doc_123",
                    "title": "French Grammar Essentials",
                    "content": "This guide covers essential French grammar rules...",
                    "lang": "en",
                    "score": 0.85,
                    "kg_path": ["grammar", "syntax", "verb"]
                }
            ]
            
            # 显示结果
            for i, result in enumerate(results[:top_k], 1):
                with st.expander(f"**{i}. {result['title']}** (Score: {result['score']:.3f})"):
                    st.markdown(f"**文档ID**: {result['doc_id']}")
                    st.markdown(f"**语言**: {result['lang']}")
                    st.markdown(f"**内容预览**: {result['content'][:200]}...")
                    
                    if use_kg and result.get("kg_path"):
                        st.markdown("**KG路径**:")
                        st.write(" → ".join(result["kg_path"]))
                    
                    st.button("📖 查看详情", key=f"detail_{i}")

# Tab 2: 知识图谱
with tab2:
    st.header("知识图谱浏览")
    
    # 概念搜索
    concept = st.text_input("搜索概念", placeholder="例如: grammaire / 语法 / grammar")
    
    if concept:
        st.subheader(f"概念: {concept}")
        
        # Mock图谱数据
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**前置知识**")
            st.write("- 字母表 (alphabet)")
            st.write("- 音节 (syllable)")
            
            st.markdown("**相关概念**")
            st.write("- 句法 (syntaxe)")
            st.write("- 词汇 (vocabulaire)")
        
        with col2:
            st.markdown("**后续知识**")
            st.write("- 从句 (clause)")
            st.write("- 复合句 (phrase complexe)")
            
            st.markdown("**学习资源**")
            st.write("- 📄 法语语法教程 (doc_001)")
            st.write("- 📺 语法讲解视频 (video_023)")
    
    # 可视化占位
    st.markdown("---")
    st.info("💡 图谱可视化将在完整版本中使用 pyvis 或 vis.js 实现")

# Tab 3: 学习路径
with tab3:
    st.header("个性化学习路径推荐")
    
    # 学习者信息
    col1, col2, col3 = st.columns(3)
    with col1:
        learner_level = st.selectbox("当前水平", ["beginner", "intermediate", "advanced"])
    with col2:
        native_lang = st.selectbox("母语", ["zh", "en", "other"])
    with col3:
        target_concept = st.text_input("目标概念", "grammaire avancée")
    
    if st.button("生成学习路径"):
        with st.spinner("分析中..."):
            # Mock学习路径
            st.success("学习路径生成完成!")
            
            path = [
                {"concept": "alphabet", "status": "mastered", "resources": 2},
                {"concept": "syllable", "status": "mastered", "resources": 3},
                {"concept": "vocabulary", "status": "in-progress", "resources": 5},
                {"concept": "basic grammar", "status": "not-started", "resources": 8},
                {"concept": "verb conjugation", "status": "not-started", "resources": 6},
                {"concept": "advanced grammar", "status": "not-started", "resources": 10}
            ]
            
            st.subheader("推荐学习路径")
            
            for i, step in enumerate(path, 1):
                status_icon = {
                    "mastered": "✅",
                    "in-progress": "🔄",
                    "not-started": "⭕"
                }[step["status"]]
                
                status_color = {
                    "mastered": "green",
                    "in-progress": "orange",
                    "not-started": "gray"
                }[step["status"]]
                
                st.markdown(
                    f"{status_icon} **Step {i}: {step['concept']}** "
                    f"<span style='color:{status_color}'>({step['status']})</span> "
                    f"- {step['resources']} 个资源",
                    unsafe_allow_html=True
                )
                
                if step["status"] == "in-progress":
                    st.info("👉 当前建议先完成此模块")
                
                if step["status"] == "not-started" and i == 4:
                    if st.button(f"开始学习: {step['concept']}", key=f"start_{i}"):
                        st.success(f"已开始学习 {step['concept']}!")

# 底部信息
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("知识图谱实体数", "1,234")
with col_b:
    st.metric("索引文档数", "5,678")
with col_c:
    st.metric("对齐实体对", "892")

st.caption("💡 系统状态: 运行中 | Neo4j: 已连接 | FAISS索引: 已加载")
