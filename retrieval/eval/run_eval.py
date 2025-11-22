"""
评测执行器

运行完整的检索评测流程
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.eval.metrics import evaluate_results
from logger import get_logger

logger = get_logger(__name__)


class Evaluator:
    """评测器"""
    
    def __init__(
        self,
        qrels_file: Path,
        metrics: List[str] = None,
        k_values: Dict[str, int] = None
    ):
        """
        初始化评测器
        
        Args:
            qrels_file: 相关性标注文件(TSV或JSONL)
            metrics: 评测指标
            k_values: 截断值
        """
        self.qrels_file = Path(qrels_file)
        self.metrics = metrics or ["ndcg", "mrr", "recall"]
        self.k_values = k_values or {"ndcg": 10, "recall": 50, "precision": 10}
        
        # 加载qrels
        self.qrels = self._load_qrels()
        
        logger.info(f"评测器初始化: {len(self.qrels)} 个查询")
    
    def _load_qrels(self) -> Dict[str, Dict[str, int]]:
        """
        加载相关性标注
        
        Returns:
            {qid: {doc_id: relevance}}
        """
        qrels = {}
        
        if not self.qrels_file.exists():
            logger.warning(f"Qrels文件不存在: {self.qrels_file}")
            return qrels
        
        # 判断文件格式
        if self.qrels_file.suffix == ".tsv":
            qrels = self._load_qrels_tsv()
        elif self.qrels_file.suffix == ".jsonl":
            qrels = self._load_qrels_jsonl()
        else:
            logger.error(f"不支持的qrels格式: {self.qrels_file.suffix}")
        
        logger.info(f"加载qrels: {len(qrels)} 个查询")
        return qrels
    
    def _load_qrels_tsv(self) -> Dict[str, Dict[str, int]]:
        """
        加载TSV格式qrels
        
        格式: qid\tdoc_id\trelevance
        """
        qrels = {}
        
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                
                qid = parts[0]
                doc_id = parts[1]
                relevance = int(parts[2])
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = relevance
        
        return qrels
    
    def _load_qrels_jsonl(self) -> Dict[str, Dict[str, int]]:
        """
        加载JSONL格式qrels
        
        格式: {"qid": "q1", "doc_id": "doc1", "relevance": 2}
        """
        qrels = {}
        
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    qid = obj["qid"]
                    doc_id = obj["doc_id"]
                    relevance = obj["relevance"]
                    
                    if qid not in qrels:
                        qrels[qid] = {}
                    qrels[qid][doc_id] = relevance
                
                except Exception as e:
                    logger.error(f"解析qrels失败: {e}")
                    continue
        
        return qrels
    
    def evaluate(
        self,
        results_file: Path,
        run_name: str = "run"
    ) -> Dict[str, float]:
        """
        评测检索结果
        
        Args:
            results_file: 检索结果文件(JSONL或TREC)
            run_name: 运行名称
            
        Returns:
            评测结果
        """
        logger.info(f"开始评测: {run_name}")
        logger.info(f"结果文件: {results_file}")
        
        # 加载结果
        results_dict = self._load_results(results_file)
        
        if not results_dict:
            logger.error("结果为空,无法评测")
            return {}
        
        # 评测
        metrics = evaluate_results(
            results_dict,
            self.qrels,
            metrics=self.metrics,
            k_values=self.k_values
        )
        
        logger.info(f"\n评测完成: {run_name}")
        for metric, score in metrics.items():
            logger.info(f"  {metric}: {score:.4f}")
        
        return metrics
    
    def _load_results(self, results_file: Path) -> Dict[str, List[str]]:
        """
        加载检索结果
        
        Returns:
            {qid: [doc_ids]}
        """
        results_file = Path(results_file)
        
        if not results_file.exists():
            logger.error(f"结果文件不存在: {results_file}")
            return {}
        
        # 判断格式
        if results_file.suffix == ".jsonl":
            return self._load_results_jsonl(results_file)
        else:
            # 默认TREC格式
            return self._load_results_trec(results_file)
    
    def _load_results_jsonl(self, results_file: Path) -> Dict[str, List[str]]:
        """
        加载JSONL格式结果
        
        格式: {"qid": "q1", "doc_id": "doc1", "rank": 1, "score": 0.95}
        """
        results_dict = {}
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    qid = obj.get("qid", "Q0")
                    doc_id = obj["doc_id"]
                    rank = obj.get("rank", 0)
                    
                    if qid not in results_dict:
                        results_dict[qid] = []
                    
                    results_dict[qid].append((rank, doc_id))
                
                except Exception as e:
                    logger.error(f"解析结果失败: {e}")
                    continue
        
        # 按rank排序
        for qid in results_dict:
            results_dict[qid].sort(key=lambda x: x[0])
            results_dict[qid] = [doc_id for _, doc_id in results_dict[qid]]
        
        return results_dict
    
    def _load_results_trec(self, results_file: Path) -> Dict[str, List[str]]:
        """
        加载TREC格式结果
        
        格式: qid Q0 doc_id rank score run_name
        """
        results_dict = {}
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 6:
                    continue
                
                qid = parts[0]
                doc_id = parts[2]
                rank = int(parts[3])
                
                if qid not in results_dict:
                    results_dict[qid] = []
                
                results_dict[qid].append((rank, doc_id))
        
        # 按rank排序
        for qid in results_dict:
            results_dict[qid].sort(key=lambda x: x[0])
            results_dict[qid] = [doc_id for _, doc_id in results_dict[qid]]
        
        return results_dict
    
    def compare_runs(
        self,
        runs: Dict[str, Path]
    ) -> Dict[str, Dict[str, float]]:
        """
        对比多个运行结果
        
        Args:
            runs: {run_name: results_file}
            
        Returns:
            {run_name: {metric: score}}
        """
        logger.info(f"\n对比 {len(runs)} 个运行:")
        
        all_metrics = {}
        
        for run_name, results_file in runs.items():
            metrics = self.evaluate(results_file, run_name=run_name)
            all_metrics[run_name] = metrics
        
        # 输出对比表
        logger.info("\n=== 对比结果 ===")
        
        # 获取所有指标名
        metric_names = set()
        for metrics in all_metrics.values():
            metric_names.update(metrics.keys())
        metric_names = sorted(metric_names)
        
        # 表头
        header = "Run".ljust(20)
        for metric in metric_names:
            header += f"{metric}".rjust(12)
        logger.info(header)
        logger.info("-" * len(header))
        
        # 各运行结果
        for run_name, metrics in all_metrics.items():
            row = run_name.ljust(20)
            for metric in metric_names:
                score = metrics.get(metric, 0.0)
                row += f"{score:.4f}".rjust(12)
            logger.info(row)
        
        return all_metrics
    
    def export_metrics(
        self,
        metrics: Dict[str, float],
        output_file: Path,
        run_name: str = "run"
    ):
        """
        导出评测结果
        
        Args:
            metrics: 评测结果
            output_file: 输出文件
            run_name: 运行名称
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            "run_name": run_name,
            "metrics": metrics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评测结果已导出: {output_file}")


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="评测检索结果")
    parser.add_argument("--results", type=str, required=True,
                       help="检索结果文件")
    parser.add_argument("--qrels", type=str, required=True,
                       help="相关性标注文件")
    parser.add_argument("--metrics", type=str, nargs="+",
                       default=["ndcg", "mrr", "recall"],
                       help="评测指标")
    parser.add_argument("--output", type=str,
                       help="输出文件")
    parser.add_argument("--run-name", type=str, default="run",
                       help="运行名称")
    
    args = parser.parse_args()
    
    # 创建评测器
    evaluator = Evaluator(
        qrels_file=Path(args.qrels),
        metrics=args.metrics
    )
    
    # 评测
    metrics = evaluator.evaluate(
        results_file=Path(args.results),
        run_name=args.run_name
    )
    
    # 导出
    if args.output:
        evaluator.export_metrics(metrics, Path(args.output), args.run_name)


if __name__ == "__main__":
    main()
