"""
LaBSE 编码器 - 跨语言向量表示

使用 LaBSE (Language-agnostic BERT Sentence Embedding)
实现跨语言的统一向量空间
"""
import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer
from logger import get_logger
from config import config

logger = get_logger(__name__)


class LaBSEEncoder:
    """LaBSE跨语言编码器"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化LaBSE编码器
        
        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu)
        """
        self.model_name = model_name or config.LABSE_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"加载LaBSE模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        self.embedding_dim = config.EMBEDDING_DIM
        logger.info(f"LaBSE加载完成, 维度: {self.embedding_dim}, 设备: {self.device}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批大小
            normalize: 是否L2归一化
            
        Returns:
            向量数组 [num_texts, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # 批量编码
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        return embeddings
    
    def encode_corpus(
        self,
        corpus: List[dict],
        text_field: str = "content",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        编码语料库
        
        Args:
            corpus: 文档列表 [{"doc_id": ..., "content": ...}, ...]
            text_field: 文本字段名
            batch_size: 批大小
            
        Returns:
            向量矩阵 [num_docs, embedding_dim]
        """
        logger.info(f"开始编码语料库: {len(corpus)} 篇文档")
        
        texts = [doc.get(text_field, "") for doc in corpus]
        
        embeddings = self.encode(texts, batch_size=batch_size)
        
        logger.info(f"编码完成: shape={embeddings.shape}")
        
        return embeddings
    
    def similarity(
        self,
        query_emb: np.ndarray,
        doc_embs: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        计算相似度
        
        Args:
            query_emb: 查询向量 [embedding_dim] 或 [num_queries, embedding_dim]
            doc_embs: 文档向量矩阵 [num_docs, embedding_dim]
            metric: 相似度度量 (cosine/dot)
            
        Returns:
            相似度分数 [num_docs] 或 [num_queries, num_docs]
        """
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        
        if metric == "cosine":
            # 余弦相似度 (假设已归一化)
            scores = np.dot(query_emb, doc_embs.T)
        elif metric == "dot":
            scores = np.dot(query_emb, doc_embs.T)
        else:
            raise ValueError(f"不支持的度量: {metric}")
        
        return scores.squeeze()


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LaBSE编码器测试")
    parser.add_argument("--text", type=str, nargs="+", help="测试文本")
    parser.add_argument("--query", type=str, help="查询文本")
    
    args = parser.parse_args()
    
    # 初始化编码器
    encoder = LaBSEEncoder()
    
    if args.text:
        # 编码测试
        logger.info(f"编码文本: {args.text}")
        embeddings = encoder.encode(args.text)
        logger.info(f"向量shape: {embeddings.shape}")
        logger.info(f"前5维: {embeddings[0][:5]}")
        
        if args.query:
            # 相似度测试
            query_emb = encoder.encode(args.query)
            scores = encoder.similarity(query_emb, embeddings)
            logger.info(f"相似度: {scores}")
    else:
        # 跨语言测试
        texts = [
            "La grammaire française est importante",
            "法语语法很重要",
            "French grammar is important",
            "天气很好"
        ]
        logger.info(f"跨语言测试: {texts}")
        
        embeddings = encoder.encode(texts)
        
        # 计算相似度矩阵
        for i, text1 in enumerate(texts):
            for j, text2 in enumerate(texts):
                if i < j:
                    sim = np.dot(embeddings[i], embeddings[j])
                    logger.info(f"[{i}] x [{j}]: {sim:.4f}")


if __name__ == "__main__":
    main()
