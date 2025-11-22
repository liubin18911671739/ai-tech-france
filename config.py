"""
集中配置文件 - 所有模块的参数管理
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """全局配置类"""
    
    # ==================== 项目路径 ====================
    PROJECT_ROOT: Path = Path(__file__).parent.absolute()
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    CLEANED_DIR: Path = DATA_DIR / "cleaned"
    PARALLEL_DIR: Path = DATA_DIR / "parallel"
    SEEDS_DIR: Path = DATA_DIR / "seeds"
    EVAL_DIR: Path = DATA_DIR / "eval"
    KG_DATA_DIR: Path = DATA_DIR / "kg"
    
    KG_DIR: Path = PROJECT_ROOT / "kg"
    RETRIEVAL_DIR: Path = PROJECT_ROOT / "retrieval"
    
    # 模型缓存
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    FAISS_INDEX_DIR: Path = MODELS_DIR / "faiss"
    WHOOSH_INDEX_DIR: Path = MODELS_DIR / "whoosh"
    ALIGNMENT_DIR: Path = MODELS_DIR / "alignment"
    
    # ==================== Neo4j ====================
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # ==================== NER 模型 ====================
    # 法语 NER
    FR_NER_MODEL: str = "Jean-Baptiste/camembert-ner"
    # 中文 NER
    ZH_NER_MODEL: str = "hanlp"  # 使用 HanLP
    # 英文 NER
    EN_NER_MODEL: str = "dslim/bert-base-NER"
    
    # ==================== 跨语言向量模型 ====================
    LABSE_MODEL: str = "sentence-transformers/LaBSE"
    EMBEDDING_DIM: int = 768
    
    # ==================== 检索参数 ====================
    # Dense
    FAISS_TOP_K: int = 100
    FAISS_NPROBE: int = 10
    
    # Sparse
    WHOOSH_TOP_K: int = 100
    
    # KG 扩展
    KG_HOP_LIMIT: int = 2
    KG_MAX_NEIGHBORS: int = 20
    
    # 融合排序权重
    ALPHA_DENSE: float = 0.4
    BETA_SPARSE: float = 0.3
    GAMMA_KG: float = 0.3
    
    # ==================== 对齐参数 ====================
    MTRANSE_DIM: int = 128
    MTRANSE_MARGIN: float = 1.0
    MTRANSE_EPOCHS: int = 50
    MTRANSE_BATCH_SIZE: int = 128
    MTRANSE_LR: float = 0.001
    
    # ==================== 自适应学习 ====================
    MASTERY_THRESHOLD: float = 0.7
    PATH_MAX_LENGTH: int = 10
    
    # LLM API (可选)
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", None)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:8000")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # ==================== 评测 ====================
    EVAL_METRICS: list = ["ndcg@10", "mrr", "recall@50"]
    
    # ==================== 其他 ====================
    RANDOM_SEED: int = 42
    LOG_LEVEL: str = "INFO"
    MAX_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        self._create_dirs()
    
    def _create_dirs(self):
        """创建必要的目录"""
        dirs = [
            self.DATA_DIR, self.RAW_DIR, self.CLEANED_DIR,
            self.PARALLEL_DIR, self.SEEDS_DIR, self.EVAL_DIR,
            self.KG_DATA_DIR,
            self.MODELS_DIR, self.FAISS_INDEX_DIR, 
            self.WHOOSH_INDEX_DIR, self.ALIGNMENT_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = Config()


if __name__ == "__main__":
    print("=== 配置信息 ===")
    print(f"项目根目录: {config.PROJECT_ROOT}")
    print(f"数据目录: {config.DATA_DIR}")
    print(f"Neo4j URI: {config.NEO4J_URI}")
    print(f"LaBSE 模型: {config.LABSE_MODEL}")
    print(f"融合权重: α={config.ALPHA_DENSE}, β={config.BETA_SPARSE}, γ={config.GAMMA_KG}")
