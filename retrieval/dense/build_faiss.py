"""
构建FAISS索引 - Dense检索核心

使用LaBSE编码后的向量构建FAISS索引
支持IVF索引(大规模数据)和Flat索引(小规模数据)
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import faiss
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config
from retrieval.dense.labse_encoder import LaBSEEncoder

logger = get_logger(__name__)


class FAISSIndexBuilder:
    """FAISS索引构建器"""
    
    def __init__(
        self,
        encoder: LaBSEEncoder = None,
        index_type: str = "IVF",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        初始化FAISS索引构建器
        
        Args:
            encoder: LaBSE编码器
            index_type: 索引类型 (Flat/IVF/IVFPQ)
            nlist: IVF聚类中心数
            nprobe: 搜索时探测的聚类数
        """
        self.encoder = encoder or LaBSEEncoder()
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        self.index = None
        self.doc_ids = []
        self.metadata = {}
        
        logger.info(f"FAISS索引构建器初始化: type={index_type}, nlist={nlist}")
    
    def build_index(
        self,
        corpus: List[Dict],
        embedding_field: str = None,
        recompute: bool = False
    ) -> faiss.Index:
        """
        构建FAISS索引
        
        Args:
            corpus: 文档列表 [{"doc_id": ..., "content": ..., "embeddings": ...}, ...]
            embedding_field: 如果文档已有嵌入,指定字段名
            recompute: 是否重新计算嵌入
            
        Returns:
            FAISS索引对象
        """
        logger.info(f"开始构建FAISS索引: {len(corpus)} 篇文档")
        
        # 获取或计算嵌入
        if embedding_field and not recompute:
            logger.info(f"使用预计算的嵌入: {embedding_field}")
            embeddings = np.array([doc[embedding_field] for doc in corpus])
        else:
            logger.info("使用LaBSE编码文档...")
            embeddings = self.encoder.encode_corpus(corpus)
        
        # 记录文档ID
        self.doc_ids = [doc.get("doc_id", f"doc_{i}") for i in range(len(corpus))]
        
        # 构建索引
        dimension = embeddings.shape[1]
        num_docs = embeddings.shape[0]
        
        logger.info(f"向量维度: {dimension}, 文档数: {num_docs}")
        
        if self.index_type == "Flat":
            # 暴力搜索,精确但慢
            self.index = faiss.IndexFlatIP(dimension)  # 内积(余弦相似度)
            logger.info("使用Flat索引(精确搜索)")
        
        elif self.index_type == "IVF":
            # IVF索引,快速近似搜索
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, self.nlist)
            logger.info(f"使用IVF索引: nlist={self.nlist}")
            
            # 训练
            logger.info("训练IVF索引...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        elif self.index_type == "IVFPQ":
            # IVF+PQ,压缩向量,节省内存
            quantizer = faiss.IndexFlatIP(dimension)
            m = 8  # PQ子向量数
            bits = 8  # 每个子向量的比特数
            self.index = faiss.IndexIVFPQ(quantizer, dimension, self.nlist, m, bits)
            logger.info(f"使用IVFPQ索引: nlist={self.nlist}, m={m}")
            
            # 训练
            logger.info("训练IVFPQ索引...")
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe
        
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        # 添加向量
        logger.info("添加向量到索引...")
        self.index.add(embeddings)
        
        # 保存元数据
        self.metadata = {
            "index_type": self.index_type,
            "dimension": dimension,
            "num_docs": num_docs,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "model": self.encoder.model_name
        }
        
        logger.info(f"索引构建完成! 总文档数: {self.index.ntotal}")
        
        return self.index
    
    def build_from_files(
        self,
        corpus_files: List[Path],
        batch_size: int = 1000
    ):
        """
        从多个语料文件构建索引
        
        Args:
            corpus_files: 语料文件列表
            batch_size: 批处理大小
        """
        logger.info(f"从{len(corpus_files)}个文件构建索引")
        
        all_corpus = []
        for corpus_file in corpus_files:
            logger.info(f"加载: {corpus_file}")
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line.strip())
                    all_corpus.append(doc)
        
        logger.info(f"总文档数: {len(all_corpus)}")
        
        return self.build_index(all_corpus)
    
    def save(self, save_dir: Path):
        """
        保存索引
        
        Args:
            save_dir: 保存目录
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        index_path = save_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"FAISS索引已保存: {index_path}")
        
        # 保存文档ID映射
        docid_path = save_dir / "doc_ids.pkl"
        with open(docid_path, 'wb') as f:
            pickle.dump(self.doc_ids, f)
        logger.info(f"文档ID映射已保存: {docid_path}")
        
        # 保存元数据
        meta_path = save_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"元数据已保存: {meta_path}")
    
    @classmethod
    def load(cls, load_dir: Path) -> 'FAISSIndexBuilder':
        """
        加载索引
        
        Args:
            load_dir: 索引目录
            
        Returns:
            FAISSIndexBuilder实例
        """
        load_dir = Path(load_dir)
        logger.info(f"加载FAISS索引: {load_dir}")
        
        # 加载元数据
        meta_path = load_dir / "metadata.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # 创建实例
        builder = cls(
            index_type=metadata["index_type"],
            nlist=metadata.get("nlist", 100),
            nprobe=metadata.get("nprobe", 10)
        )
        
        # 加载FAISS索引
        index_path = load_dir / "faiss.index"
        builder.index = faiss.read_index(str(index_path))
        logger.info(f"FAISS索引已加载: {builder.index.ntotal} 个向量")
        
        # 加载文档ID
        docid_path = load_dir / "doc_ids.pkl"
        with open(docid_path, 'rb') as f:
            builder.doc_ids = pickle.load(f)
        logger.info(f"文档ID已加载: {len(builder.doc_ids)} 个")
        
        builder.metadata = metadata
        
        return builder


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建FAISS索引")
    parser.add_argument("--corpus", type=str, nargs="+", required=True,
                       help="语料文件路径(可多个)")
    parser.add_argument("--output", type=str, required=True,
                       help="索引输出目录")
    parser.add_argument("--index-type", type=str, default="IVF",
                       choices=["Flat", "IVF", "IVFPQ"],
                       help="索引类型")
    parser.add_argument("--nlist", type=int, default=100,
                       help="IVF聚类数")
    parser.add_argument("--nprobe", type=int, default=10,
                       help="搜索探测数")
    
    args = parser.parse_args()
    
    # 构建索引
    builder = FAISSIndexBuilder(
        index_type=args.index_type,
        nlist=args.nlist,
        nprobe=args.nprobe
    )
    
    corpus_files = [Path(f) for f in args.corpus]
    builder.build_from_files(corpus_files)
    
    # 保存
    output_dir = Path(args.output)
    builder.save(output_dir)
    
    logger.info("索引构建完成!")
    logger.info(f"索引类型: {builder.index_type}")
    logger.info(f"文档总数: {builder.index.ntotal}")
    logger.info(f"保存路径: {output_dir}")


if __name__ == "__main__":
    main()
