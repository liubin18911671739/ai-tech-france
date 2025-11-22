"""
融合排序 - 多路检索结果融合

融合Dense、Sparse、KG三路检索结果
支持多种融合策略
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config

logger = get_logger(__name__)


class FusionReranker:
    """融合重排序器"""
    
    def __init__(
        self,
        alpha: float = None,
        beta: float = None,
        gamma: float = None,
        normalize: bool = True
    ):
        """
        初始化融合器
        
        Args:
            alpha: Dense检索权重
            beta: Sparse检索权重
            gamma: KG增强权重
            normalize: 是否归一化得分
        """
        # 从配置读取默认权重
        self.alpha = alpha if alpha is not None else config.ALPHA_DENSE
        self.beta = beta if beta is not None else config.BETA_SPARSE
        self.gamma = gamma if gamma is not None else config.GAMMA_KG
        self.normalize = normalize
        
        # 验证权重和为1
        total = self.alpha + self.beta + self.gamma
        if abs(total - 1.0) > 0.01:
            logger.warning(f"权重和不为1 ({total:.3f}), 将自动归一化")
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
        
        logger.info(f"融合器初始化: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")
    
    def fuse_scores(
        self,
        dense_results: List[Dict] = None,
        sparse_results: List[Dict] = None,
        kg_scores: Dict[str, float] = None,
        method: str = "weighted_sum"
    ) -> List[Dict]:
        """
        融合三路检索得分
        
        Args:
            dense_results: Dense检索结果 [{"doc_id": ..., "score": ...}, ...]
            sparse_results: Sparse检索结果 [{"doc_id": ..., "score": ...}, ...]
            kg_scores: KG节点得分 {doc_id: score}
            method: 融合方法 ("weighted_sum", "rrf", "max")
            
        Returns:
            融合后的结果列表(按得分降序)
        """
        # 转换为字典格式
        dense_dict = self._to_dict(dense_results) if dense_results else {}
        sparse_dict = self._to_dict(sparse_results) if sparse_results else {}
        kg_dict = kg_scores or {}
        
        # 归一化得分
        if self.normalize:
            dense_dict = self._normalize_scores(dense_dict)
            sparse_dict = self._normalize_scores(sparse_dict)
            kg_dict = self._normalize_scores(kg_dict)
        
        # 收集所有文档ID
        all_doc_ids = set()
        all_doc_ids.update(dense_dict.keys())
        all_doc_ids.update(sparse_dict.keys())
        all_doc_ids.update(kg_dict.keys())
        
        logger.info(f"融合文档: Dense={len(dense_dict)}, Sparse={len(sparse_dict)}, KG={len(kg_dict)}, Total={len(all_doc_ids)}")
        
        # 根据方法融合
        if method == "weighted_sum":
            fused = self._weighted_sum(all_doc_ids, dense_dict, sparse_dict, kg_dict)
        elif method == "rrf":
            fused = self._reciprocal_rank_fusion(dense_results, sparse_results, kg_scores)
        elif method == "max":
            fused = self._max_fusion(all_doc_ids, dense_dict, sparse_dict, kg_dict)
        else:
            logger.warning(f"未知融合方法: {method}, 使用weighted_sum")
            fused = self._weighted_sum(all_doc_ids, dense_dict, sparse_dict, kg_dict)
        
        # 按得分降序排序
        fused.sort(key=lambda x: x["fused_score"], reverse=True)
        
        logger.info(f"融合完成: {len(fused)} 个文档")
        return fused
    
    def _to_dict(self, results: List[Dict]) -> Dict[str, float]:
        """
        将结果列表转为字典
        
        Args:
            results: [{"doc_id": ..., "score": ...}, ...]
            
        Returns:
            {doc_id: score}
        """
        return {item["doc_id"]: item.get("score", 0.0) for item in results}
    
    def _normalize_scores(self, score_dict: Dict[str, float]) -> Dict[str, float]:
        """
        归一化得分到[0, 1]
        
        Args:
            score_dict: {doc_id: score}
            
        Returns:
            归一化后的得分字典
        """
        if not score_dict:
            return {}
        
        scores = list(score_dict.values())
        min_score = min(scores)
        max_score = max(scores)
        
        # 避免除零
        if max_score - min_score < 1e-6:
            return {doc_id: 1.0 for doc_id in score_dict}
        
        # Min-Max归一化
        normalized = {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in score_dict.items()
        }
        
        return normalized
    
    def _weighted_sum(
        self,
        doc_ids: set,
        dense_dict: Dict[str, float],
        sparse_dict: Dict[str, float],
        kg_dict: Dict[str, float]
    ) -> List[Dict]:
        """
        加权求和融合
        
        Score = α·dense + β·sparse + γ·kg
        """
        results = []
        
        for doc_id in doc_ids:
            dense_score = dense_dict.get(doc_id, 0.0)
            sparse_score = sparse_dict.get(doc_id, 0.0)
            kg_score = kg_dict.get(doc_id, 0.0)
            
            # 加权求和
            fused_score = (
                self.alpha * dense_score +
                self.beta * sparse_score +
                self.gamma * kg_score
            )
            
            results.append({
                "doc_id": doc_id,
                "fused_score": fused_score,
                "dense_score": dense_score,
                "sparse_score": sparse_score,
                "kg_score": kg_score,
                "method": "weighted_sum"
            })
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        kg_scores: Dict[str, float],
        k: int = 60
    ) -> List[Dict]:
        """
        倒数排名融合(Reciprocal Rank Fusion, RRF)
        
        RRF(d) = Σ 1/(k + rank_i(d))
        
        Args:
            dense_results: Dense检索结果(有序)
            sparse_results: Sparse检索结果(有序)
            kg_scores: KG得分
            k: RRF常数(默认60)
            
        Returns:
            融合结果
        """
        rrf_scores = defaultdict(float)
        
        # Dense排名
        if dense_results:
            for rank, item in enumerate(dense_results, 1):
                doc_id = item["doc_id"]
                rrf_scores[doc_id] += self.alpha / (k + rank)
        
        # Sparse排名
        if sparse_results:
            for rank, item in enumerate(sparse_results, 1):
                doc_id = item["doc_id"]
                rrf_scores[doc_id] += self.beta / (k + rank)
        
        # KG得分转排名(按得分降序)
        if kg_scores:
            sorted_kg = sorted(kg_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_kg, 1):
                rrf_scores[doc_id] += self.gamma / (k + rank)
        
        # 构建结果
        results = []
        for doc_id, fused_score in rrf_scores.items():
            results.append({
                "doc_id": doc_id,
                "fused_score": fused_score,
                "method": "rrf"
            })
        
        return results
    
    def _max_fusion(
        self,
        doc_ids: set,
        dense_dict: Dict[str, float],
        sparse_dict: Dict[str, float],
        kg_dict: Dict[str, float]
    ) -> List[Dict]:
        """
        最大值融合
        
        Score = max(α·dense, β·sparse, γ·kg)
        """
        results = []
        
        for doc_id in doc_ids:
            dense_score = self.alpha * dense_dict.get(doc_id, 0.0)
            sparse_score = self.beta * sparse_dict.get(doc_id, 0.0)
            kg_score = self.gamma * kg_dict.get(doc_id, 0.0)
            
            # 取最大值
            fused_score = max(dense_score, sparse_score, kg_score)
            
            results.append({
                "doc_id": doc_id,
                "fused_score": fused_score,
                "dense_score": dense_dict.get(doc_id, 0.0),
                "sparse_score": sparse_dict.get(doc_id, 0.0),
                "kg_score": kg_dict.get(doc_id, 0.0),
                "method": "max"
            })
        
        return results
    
    def rerank_with_details(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        kg_scores: Dict[str, float],
        corpus: Dict[str, Dict] = None,
        top_k: int = None
    ) -> List[Dict]:
        """
        融合重排并添加文档详情
        
        Args:
            dense_results: Dense检索结果
            sparse_results: Sparse检索结果
            kg_scores: KG得分
            corpus: 文档语料 {doc_id: {"title": ..., "content": ...}}
            top_k: 返回Top-K结果
            
        Returns:
            详细结果列表
        """
        # 融合得分
        fused = self.fuse_scores(dense_results, sparse_results, kg_scores)
        
        # 限制Top-K
        if top_k:
            fused = fused[:top_k]
        
        # 添加文档详情
        if corpus:
            for item in fused:
                doc_id = item["doc_id"]
                if doc_id in corpus:
                    doc = corpus[doc_id]
                    item["title"] = doc.get("title", "")
                    item["content"] = doc.get("content", "")
                    item["lang"] = doc.get("lang", "")

        return fused

    def rerank(
        self,
        dense_results: List[Dict] = None,
        sparse_results: List[Dict] = None,
        kg_results: Union[Dict[str, float], List[Dict]] = None,
        top_k: int = None,
        method: str = "weighted_sum",
        corpus: Dict[str, Dict] = None
    ) -> List[Dict]:
        """统一的rerank入口,兼容list/dict格式的KG结果"""
        if isinstance(kg_results, list):
            kg_scores = {item["doc_id"]: item.get("score", 0.0) for item in kg_results}
        else:
            kg_scores = kg_results or {}

        fused = self.rerank_with_details(
            dense_results or [],
            sparse_results or [],
            kg_scores,
            corpus=corpus,
            top_k=top_k
        )

        # 保留得分贡献,便于Explain
        for item in fused:
            item["score_contributions"] = {
                "dense": item.get("dense_score", 0.0),
                "sparse": item.get("sparse_score", 0.0),
                "kg": item.get("kg_score", 0.0)
            }

        return fused
    
    def explain_fusion(self, doc_id: str, fused_result: Dict) -> Dict:
        """
        解释融合得分
        
        Args:
            doc_id: 文档ID
            fused_result: 融合结果项
            
        Returns:
            解释信息
        """
        explanation = {
            "doc_id": doc_id,
            "fused_score": fused_result.get("fused_score", 0.0),
            "components": {
                "dense": {
                    "score": fused_result.get("dense_score", 0.0),
                    "weight": self.alpha,
                    "contribution": self.alpha * fused_result.get("dense_score", 0.0)
                },
                "sparse": {
                    "score": fused_result.get("sparse_score", 0.0),
                    "weight": self.beta,
                    "contribution": self.beta * fused_result.get("sparse_score", 0.0)
                },
                "kg": {
                    "score": fused_result.get("kg_score", 0.0),
                    "weight": self.gamma,
                    "contribution": self.gamma * fused_result.get("kg_score", 0.0)
                }
            },
            "method": fused_result.get("method", "weighted_sum")
        }
        
        return explanation
    
    def batch_fusion(
        self,
        queries_results: Dict[str, Dict],
        method: str = "weighted_sum"
    ) -> Dict[str, List[Dict]]:
        """
        批量融合
        
        Args:
            queries_results: {
                "q1": {
                    "dense": [...],
                    "sparse": [...],
                    "kg": {...}
                },
                ...
            }
            method: 融合方法
            
        Returns:
            {qid: [fused_results]}
        """
        logger.info(f"批量融合: {len(queries_results)} 个查询")
        
        batch_results = {}
        
        for qid, results in queries_results.items():
            dense_results = results.get("dense", [])
            sparse_results = results.get("sparse", [])
            kg_scores = results.get("kg", {})
            
            fused = self.fuse_scores(
                dense_results,
                sparse_results,
                kg_scores,
                method=method
            )
            
            batch_results[qid] = fused
        
        logger.info(f"批量融合完成")
        return batch_results
    
    def export_results(
        self,
        results: List[Dict],
        output_path: Path,
        format: str = "jsonl"
    ):
        """
        导出融合结果
        
        Args:
            results: 融合结果
            output_path: 输出路径
            format: 输出格式 ("jsonl" 或 "trec")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            logger.info(f"结果已导出(JSONL): {output_path}")
        
        elif format == "trec":
            with open(output_path, 'w', encoding='utf-8') as f:
                for rank, item in enumerate(results, 1):
                    # TREC格式: qid Q0 doc_id rank score run_name
                    qid = item.get("qid", "Q0")
                    doc_id = item["doc_id"]
                    score = item["fused_score"]
                    f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} FUSION\n")
            
            logger.info(f"结果已导出(TREC): {output_path}")
        
        else:
            raise ValueError(f"不支持的格式: {format}")


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="融合排序测试")
    parser.add_argument("--alpha", type=float, default=0.4,
                       help="Dense权重")
    parser.add_argument("--beta", type=float, default=0.3,
                       help="Sparse权重")
    parser.add_argument("--gamma", type=float, default=0.3,
                       help="KG权重")
    parser.add_argument("--method", type=str, default="weighted_sum",
                       choices=["weighted_sum", "rrf", "max"],
                       help="融合方法")
    parser.add_argument("--demo", action="store_true",
                       help="运行演示")
    
    args = parser.parse_args()
    
    # 创建融合器
    reranker = FusionReranker(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    if args.demo:
        logger.info("\n=== 融合排序演示 ===\n")
        
        # 模拟三路检索结果
        dense_results = [
            {"doc_id": "doc1", "score": 0.95},
            {"doc_id": "doc2", "score": 0.85},
            {"doc_id": "doc3", "score": 0.75}
        ]
        
        sparse_results = [
            {"doc_id": "doc2", "score": 0.90},
            {"doc_id": "doc4", "score": 0.80},
            {"doc_id": "doc1", "score": 0.70}
        ]
        
        kg_scores = {
            "doc1": 0.60,
            "doc3": 0.85,
            "doc5": 0.75
        }
        
        # 融合
        fused = reranker.fuse_scores(
            dense_results,
            sparse_results,
            kg_scores,
            method=args.method
        )
        
        # 输出结果
        logger.info(f"\n融合方法: {args.method}")
        logger.info(f"权重: α={args.alpha}, β={args.beta}, γ={args.gamma}\n")
        
        logger.info("融合结果(Top-5):")
        for i, item in enumerate(fused[:5], 1):
            logger.info(f"{i}. {item['doc_id']}")
            logger.info(f"   总分: {item['fused_score']:.4f}")
            logger.info(f"   Dense: {item.get('dense_score', 0):.4f} (×{args.alpha})")
            logger.info(f"   Sparse: {item.get('sparse_score', 0):.4f} (×{args.beta})")
            logger.info(f"   KG: {item.get('kg_score', 0):.4f} (×{args.gamma})")
            
            # 详细解释
            explanation = reranker.explain_fusion(item['doc_id'], item)
            logger.info(f"   贡献度:")
            for comp, info in explanation['components'].items():
                logger.info(f"     {comp}: {info['contribution']:.4f}")
            logger.info("")
        
        # 对比不同融合方法
        logger.info("\n=== 对比不同融合方法 ===\n")
        
        methods = ["weighted_sum", "rrf", "max"]
        for method in methods:
            fused = reranker.fuse_scores(
                dense_results,
                sparse_results,
                kg_scores,
                method=method
            )
            logger.info(f"{method}: Top-3 = {[item['doc_id'] for item in fused[:3]]}")
    
    else:
        logger.info("使用 --demo 参数运行演示")


if __name__ == "__main__":
    main()
