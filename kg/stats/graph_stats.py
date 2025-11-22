"""
图谱统计分析

分析知识图谱的统计特征
"""
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger

logger = get_logger(__name__)


class GraphStatistics:
    """图谱统计分析器"""
    
    def __init__(self):
        """初始化统计器"""
        self.nodes = []
        self.relations = []
        
        self.node_by_lang = defaultdict(list)
        self.node_by_type = defaultdict(list)
        self.rel_by_type = defaultdict(list)
        
        self.in_degree = defaultdict(int)
        self.out_degree = defaultdict(int)
        
        logger.info("图谱统计器初始化")
    
    def load_graph(
        self,
        nodes_file: Path,
        relations_file: Path
    ):
        """
        加载图谱
        
        Args:
            nodes_file: 节点文件
            relations_file: 关系文件
        """
        logger.info("加载图谱数据")
        
        # 加载节点
        with open(nodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                self.nodes.append(obj)
                
                # 分类
                self.node_by_lang[obj["lang"]].append(obj)
                self.node_by_type[obj["type"]].append(obj)
        
        logger.info(f"加载 {len(self.nodes)} 个节点")
        
        # 加载关系
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                self.relations.append(obj)
                
                # 分类
                self.rel_by_type[obj["type"]].append(obj)
                
                # 度数统计
                self.out_degree[obj["source"]] += 1
                self.in_degree[obj["target"]] += 1
        
        logger.info(f"加载 {len(self.relations)} 条关系")
    
    def compute_basic_stats(self) -> Dict:
        """
        计算基本统计
        
        Returns:
            统计信息
        """
        stats = {
            "total_nodes": len(self.nodes),
            "total_relations": len(self.relations),
            "nodes_by_lang": {
                lang: len(nodes)
                for lang, nodes in self.node_by_lang.items()
            },
            "nodes_by_type": {
                node_type: len(nodes)
                for node_type, nodes in self.node_by_type.items()
            },
            "relations_by_type": {
                rel_type: len(rels)
                for rel_type, rels in self.rel_by_type.items()
            }
        }
        
        return stats
    
    def compute_degree_stats(self) -> Dict:
        """
        计算度数统计
        
        Returns:
            度数统计
        """
        in_degrees = list(self.in_degree.values())
        out_degrees = list(self.out_degree.values())
        
        stats = {
            "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "isolated_nodes": len([n for n in self.nodes if self.out_degree.get(n["id"], 0) == 0 and self.in_degree.get(n["id"], 0) == 0])
        }
        
        return stats
    
    def get_top_nodes(self, top_k: int = 10) -> Dict:
        """
        获取Top节点
        
        Args:
            top_k: Top数量
            
        Returns:
            Top节点
        """
        # 按总度数排序
        total_degree = {}
        for node_id in set(list(self.in_degree.keys()) + list(self.out_degree.keys())):
            total_degree[node_id] = self.in_degree.get(node_id, 0) + self.out_degree.get(node_id, 0)
        
        sorted_nodes = sorted(total_degree.items(), key=lambda x: x[1], reverse=True)
        
        top_nodes = []
        for node_id, degree in sorted_nodes[:top_k]:
            # 查找节点信息
            node_info = next((n for n in self.nodes if n["id"] == node_id), None)
            if node_info:
                top_nodes.append({
                    "id": node_id,
                    "name": node_info["name"],
                    "lang": node_info["lang"],
                    "in_degree": self.in_degree.get(node_id, 0),
                    "out_degree": self.out_degree.get(node_id, 0),
                    "total_degree": degree
                })
        
        return {"top_nodes": top_nodes}
    
    def compute_connectivity_stats(self) -> Dict:
        """
        计算连通性统计
        
        Returns:
            连通性统计
        """
        # 构建邻接表
        adj_list = defaultdict(list)
        for rel in self.relations:
            adj_list[rel["source"]].append(rel["target"])
        
        # BFS计算连通分量
        visited = set()
        components = []
        
        for node in self.nodes:
            node_id = node["id"]
            if node_id in visited:
                continue
            
            # BFS
            component = set()
            queue = [node_id]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                component.add(current)
                
                # 添加邻居
                for neighbor in adj_list.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        # 统计
        stats = {
            "num_components": len(components),
            "largest_component_size": max(len(c) for c in components) if components else 0,
            "smallest_component_size": min(len(c) for c in components) if components else 0,
            "avg_component_size": sum(len(c) for c in components) / len(components) if components else 0
        }
        
        return stats
    
    def compute_alignment_stats(self) -> Dict:
        """
        计算对齐统计
        
        Returns:
            对齐统计
        """
        # 统计对齐关系
        alignment_rels = self.rel_by_type.get("ALIGNED_WITH", [])
        
        # 按语言对统计
        lang_pairs = defaultdict(int)
        
        for rel in alignment_rels:
            source_id = rel["source"]
            target_id = rel["target"]
            
            # 查找语言
            source_node = next((n for n in self.nodes if n["id"] == source_id), None)
            target_node = next((n for n in self.nodes if n["id"] == target_id), None)
            
            if source_node and target_node:
                lang1 = source_node["lang"]
                lang2 = target_node["lang"]
                pair = tuple(sorted([lang1, lang2]))
                lang_pairs[pair] += 1
        
        # 统计对齐覆盖率
        aligned_nodes = set()
        for rel in alignment_rels:
            aligned_nodes.add(rel["source"])
            aligned_nodes.add(rel["target"])
        
        coverage_by_lang = {}
        for lang, nodes in self.node_by_lang.items():
            lang_aligned = sum(1 for n in nodes if n["id"] in aligned_nodes)
            coverage_by_lang[lang] = lang_aligned / len(nodes) if nodes else 0
        
        stats = {
            "total_alignments": len(alignment_rels),
            "aligned_nodes": len(aligned_nodes),
            "alignment_coverage": len(aligned_nodes) / len(self.nodes) if self.nodes else 0,
            "alignments_by_lang_pair": {
                f"{p[0]}-{p[1]}": count
                for p, count in lang_pairs.items()
            },
            "coverage_by_lang": coverage_by_lang
        }
        
        return stats
    
    def generate_report(self) -> str:
        """
        生成统计报告
        
        Returns:
            Markdown格式报告
        """
        logger.info("生成统计报告")
        
        # 计算各项统计
        basic_stats = self.compute_basic_stats()
        degree_stats = self.compute_degree_stats()
        top_nodes = self.get_top_nodes(top_k=10)
        connectivity_stats = self.compute_connectivity_stats()
        alignment_stats = self.compute_alignment_stats()
        
        # 生成Markdown报告
        report = f"""# 知识图谱统计报告

## 1. 基本统计

- **总节点数**: {basic_stats['total_nodes']:,}
- **总关系数**: {basic_stats['total_relations']:,}

### 节点按语言分布

| 语言 | 节点数 |
|------|--------|
"""
        
        for lang, count in basic_stats['nodes_by_lang'].items():
            report += f"| {lang} | {count:,} |\n"
        
        report += f"""
### 节点按类型分布

| 类型 | 节点数 |
|------|--------|
"""
        
        for node_type, count in basic_stats['nodes_by_type'].items():
            report += f"| {node_type} | {count:,} |\n"
        
        report += f"""
### 关系按类型分布

| 关系类型 | 关系数 |
|----------|--------|
"""
        
        for rel_type, count in basic_stats['relations_by_type'].items():
            report += f"| {rel_type} | {count:,} |\n"
        
        report += f"""
## 2. 度数统计

- **平均入度**: {degree_stats['avg_in_degree']:.2f}
- **最大入度**: {degree_stats['max_in_degree']}
- **平均出度**: {degree_stats['avg_out_degree']:.2f}
- **最大出度**: {degree_stats['max_out_degree']}
- **孤立节点数**: {degree_stats['isolated_nodes']}

## 3. Top-10 节点

| 排名 | 节点名称 | 语言 | 入度 | 出度 | 总度数 |
|------|----------|------|------|------|--------|
"""
        
        for i, node in enumerate(top_nodes['top_nodes'], 1):
            report += f"| {i} | {node['name']} | {node['lang']} | {node['in_degree']} | {node['out_degree']} | {node['total_degree']} |\n"
        
        report += f"""
## 4. 连通性统计

- **连通分量数**: {connectivity_stats['num_components']}
- **最大分量大小**: {connectivity_stats['largest_component_size']}
- **最小分量大小**: {connectivity_stats['smallest_component_size']}
- **平均分量大小**: {connectivity_stats['avg_component_size']:.2f}

## 5. 对齐统计

- **总对齐关系数**: {alignment_stats['total_alignments']:,}
- **对齐节点数**: {alignment_stats['aligned_nodes']:,}
- **对齐覆盖率**: {alignment_stats['alignment_coverage']:.2%}

### 按语言对分布

| 语言对 | 对齐数 |
|--------|--------|
"""
        
        for pair, count in alignment_stats['alignments_by_lang_pair'].items():
            report += f"| {pair} | {count:,} |\n"
        
        report += f"""
### 各语言对齐覆盖率

| 语言 | 覆盖率 |
|------|--------|
"""
        
        for lang, coverage in alignment_stats['coverage_by_lang'].items():
            report += f"| {lang} | {coverage:.2%} |\n"
        
        report += "\n---\n生成时间: " + str(Path(__file__).stat().st_mtime)
        
        return report
    
    def export_report(self, output_file: Path):
        """
        导出报告
        
        Args:
            output_file: 输出文件
        """
        report = self.generate_report()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已导出: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="图谱统计分析")
    parser.add_argument("--nodes", type=str,
                       default="data/kg/nodes.jsonl",
                       help="节点文件")
    parser.add_argument("--relations", type=str,
                       default="data/kg/relations.jsonl",
                       help="关系文件")
    parser.add_argument("--output", type=str,
                       default="artifacts/graph_stats.md",
                       help="输出报告文件")
    
    args = parser.parse_args()
    
    # 创建统计器
    stats = GraphStatistics()
    
    # 加载图谱
    stats.load_graph(
        nodes_file=Path(args.nodes),
        relations_file=Path(args.relations)
    )
    
    # 生成报告
    stats.export_report(Path(args.output))
    
    # 控制台输出
    print(stats.generate_report())


if __name__ == "__main__":
    main()
