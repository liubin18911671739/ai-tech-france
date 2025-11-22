"""
评测指标实现

实现标准IR评测指标: nDCG, MRR, Recall
"""
import math
from typing import List, Dict, Set
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger

logger = get_logger(__name__)


def calculate_ndcg(
    results: List[str],
    relevance: Dict[str, int],
    k: int = 10
) -> float:
    """
    计算nDCG@k (Normalized Discounted Cumulative Gain)
    
    Args:
        results: 检索结果文档ID列表(有序)
        relevance: 相关性标注 {doc_id: relevance_score}
        k: 截断位置
        
    Returns:
        nDCG@k值 [0, 1]
    """
    # 截断到k
    results = results[:k]
    
    # 计算DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(results, 1):
        rel = relevance.get(doc_id, 0)
        # DCG公式: Σ (2^rel - 1) / log2(i + 1)
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    
    # 计算IDCG (Ideal DCG)
    # 按相关性降序排列
    ideal_rels = sorted(relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, 1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)
    
    # 避免除零
    if idcg == 0:
        return 0.0
    
    # nDCG = DCG / IDCG
    ndcg = dcg / idcg
    return ndcg


def calculate_mrr(
    results: List[str],
    relevant_docs: Set[str]
) -> float:
    """
    计算MRR (Mean Reciprocal Rank)
    
    Args:
        results: 检索结果文档ID列表(有序)
        relevant_docs: 相关文档ID集合
        
    Returns:
        RR值 (对于单个查询就是RR,批量平均为MRR)
    """
    # 找到第一个相关文档的位置
    for i, doc_id in enumerate(results, 1):
        if doc_id in relevant_docs:
            return 1.0 / i
    
    # 没有找到相关文档
    return 0.0


def calculate_recall(
    results: List[str],
    relevant_docs: Set[str],
    k: int = 50
) -> float:
    """
    计算Recall@k
    
    Args:
        results: 检索结果文档ID列表(有序)
        relevant_docs: 相关文档ID集合
        k: 截断位置
        
    Returns:
        Recall@k值 [0, 1]
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    # 截断到k
    results = results[:k]
    
    # 计算召回率
    retrieved_relevant = set(results) & relevant_docs
    recall = len(retrieved_relevant) / len(relevant_docs)
    
    return recall


def calculate_precision(
    results: List[str],
    relevant_docs: Set[str],
    k: int = 10
) -> float:
    """
    计算Precision@k
    
    Args:
        results: 检索结果文档ID列表(有序)
        relevant_docs: 相关文档ID集合
        k: 截断位置
        
    Returns:
        Precision@k值 [0, 1]
    """
    if k == 0:
        return 0.0
    
    # 截断到k
    results = results[:k]
    
    # 计算准确率
    retrieved_relevant = set(results) & relevant_docs
    precision = len(retrieved_relevant) / k
    
    return precision


def calculate_map(
    results: List[str],
    relevant_docs: Set[str]
) -> float:
    """
    计算MAP (Mean Average Precision)
    
    Args:
        results: 检索结果文档ID列表(有序)
        relevant_docs: 相关文档ID集合
        
    Returns:
        AP值 (对于单个查询就是AP,批量平均为MAP)
    """
    if len(relevant_docs) == 0:
        return 0.0
    
    # 计算每个相关文档位置的Precision
    precisions = []
    relevant_count = 0
    
    for i, doc_id in enumerate(results, 1):
        if doc_id in relevant_docs:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precisions.append(precision_at_i)
    
    # AP = 平均Precision
    if len(precisions) == 0:
        return 0.0
    
    ap = sum(precisions) / len(relevant_docs)
    return ap


def evaluate_results(
    results_dict: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str] = None,
    k_values: Dict[str, int] = None
) -> Dict[str, float]:
    """
    批量评测
    
    Args:
        results_dict: 检索结果 {qid: [doc_ids]}
        qrels: 相关性标注 {qid: {doc_id: relevance}}
        metrics: 评测指标列表 ["ndcg", "mrr", "recall", "precision", "map"]
        k_values: 截断值 {"ndcg": 10, "recall": 50, "precision": 10}
        
    Returns:
        评测结果 {metric_name: score}
    """
    if metrics is None:
        metrics = ["ndcg", "mrr", "recall"]
    
    if k_values is None:
        k_values = {"ndcg": 10, "recall": 50, "precision": 10}
    
    # 初始化累加器
    metric_scores = {metric: [] for metric in metrics}
    
    # 逐个查询评测
    for qid, results in results_dict.items():
        if qid not in qrels:
            logger.warning(f"查询 {qid} 没有相关性标注,跳过")
            continue
        
        relevance = qrels[qid]
        relevant_docs = {doc_id for doc_id, rel in relevance.items() if rel > 0}
        
        # 计算各指标
        if "ndcg" in metrics:
            k = k_values.get("ndcg", 10)
            ndcg = calculate_ndcg(results, relevance, k=k)
            metric_scores["ndcg"].append(ndcg)
        
        if "mrr" in metrics:
            mrr = calculate_mrr(results, relevant_docs)
            metric_scores["mrr"].append(mrr)
        
        if "recall" in metrics:
            k = k_values.get("recall", 50)
            recall = calculate_recall(results, relevant_docs, k=k)
            metric_scores["recall"].append(recall)
        
        if "precision" in metrics:
            k = k_values.get("precision", 10)
            precision = calculate_precision(results, relevant_docs, k=k)
            metric_scores["precision"].append(precision)
        
        if "map" in metrics:
            map_score = calculate_map(results, relevant_docs)
            metric_scores["map"].append(map_score)
    
    # 计算平均值
    avg_metrics = {}
    for metric, scores in metric_scores.items():
        if len(scores) > 0:
            avg_metrics[metric] = sum(scores) / len(scores)
        else:
            avg_metrics[metric] = 0.0
    
    # 添加k值到指标名
    formatted_metrics = {}
    for metric, score in avg_metrics.items():
        if metric in k_values:
            k = k_values[metric]
            formatted_metrics[f"{metric}@{k}"] = score
        else:
            formatted_metrics[metric] = score
    
    return formatted_metrics


def main():
    """测试函数"""
    # 模拟数据
    results = ["doc1", "doc3", "doc5", "doc2", "doc4"]
    
    relevance = {
        "doc1": 3,  # 高度相关
        "doc2": 2,  # 相关
        "doc3": 1,  # 部分相关
        "doc4": 0   # 不相关
    }
    
    relevant_docs = {"doc1", "doc2", "doc3"}
    
    # 计算指标
    ndcg_10 = calculate_ndcg(results, relevance, k=10)
    mrr = calculate_mrr(results, relevant_docs)
    recall_50 = calculate_recall(results, relevant_docs, k=50)
    precision_10 = calculate_precision(results, relevant_docs, k=10)
    map_score = calculate_map(results, relevant_docs)
    
    logger.info("=== 评测指标测试 ===")
    logger.info(f"结果列表: {results}")
    logger.info(f"相关文档: {relevant_docs}")
    logger.info(f"\nnDCG@10: {ndcg_10:.4f}")
    logger.info(f"MRR: {mrr:.4f}")
    logger.info(f"Recall@50: {recall_50:.4f}")
    logger.info(f"Precision@10: {precision_10:.4f}")
    logger.info(f"MAP: {map_score:.4f}")
    
    # 批量评测
    logger.info("\n=== 批量评测测试 ===")
    results_dict = {
        "q1": results,
        "q2": ["doc2", "doc4", "doc1", "doc5"]
    }
    qrels = {
        "q1": relevance,
        "q2": {"doc1": 2, "doc2": 3, "doc5": 1}
    }
    
    metrics = evaluate_results(results_dict, qrels, metrics=["ndcg", "mrr", "recall"])
    for metric, score in metrics.items():
        logger.info(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()
