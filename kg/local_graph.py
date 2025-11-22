"""
本地知识图谱索引

用于在 Neo4j 不可用的情况下,直接从导出的 nodes/relations JSONL 文件中
加载并查询概念与关系,满足轻量级 MVP 场景。
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from logger import get_logger

logger = get_logger(__name__)


class LocalGraphStore:
    """简单的本地知识图谱查询接口"""

    def __init__(self, kg_dir: Path):
        self.kg_dir = Path(kg_dir)
        self.nodes: Dict[str, Dict] = {}
        self.nodes_by_lang: Dict[str, List[Dict]] = {}
        self.edges: Dict[str, List[Dict]] = {}

        self._load()

    def _load(self) -> None:
        nodes_file = self.kg_dir / "nodes.jsonl"
        relations_file = self.kg_dir / "relations.jsonl"

        if not nodes_file.exists() or not relations_file.exists():
            raise FileNotFoundError(
                f"缺少本地知识图谱文件: {nodes_file} or {relations_file}"
            )

        logger.info(f"加载本地知识图谱: {self.kg_dir}")

        with open(nodes_file, "r", encoding="utf-8") as f:
            for line in f:
                node = json.loads(line.strip())
                node_id = node.get("id")
                if not node_id:
                    continue
                self.nodes[node_id] = node
                lang = node.get("lang", "unknown")
                self.nodes_by_lang.setdefault(lang, []).append(node)

        with open(relations_file, "r", encoding="utf-8") as f:
            for line in f:
                rel = json.loads(line.strip())
                source = rel.get("source")
                target = rel.get("target")
                if not source or not target:
                    continue
                rel_entry = {
                    "target_id": target,
                    "relation": rel.get("type", "RELATED_TO"),
                    "weight": rel.get("properties", {}).get("weight", 1.0)
                }
                self.edges.setdefault(source, []).append(rel_entry)

        logger.info(
            "本地知识图谱加载完成: %d 个节点, %d 个有出边的节点",
            len(self.nodes),
            len(self.edges)
        )

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def search(self, name: str, lang: Optional[str] = None, entity_type: Optional[str] = None,
               limit: int = 5, threshold: float = 0.4) -> List[Dict]:
        """根据名称进行模糊搜索"""
        candidates: List[Dict] = []
        pool = self.nodes.values() if lang is None else self.nodes_by_lang.get(lang, [])

        name_lower = name.lower()

        for node in pool:
            if entity_type and node.get("type") != entity_type:
                continue

            candidate_name = node.get("name", "").lower()
            similarity = self._string_similarity(name_lower, candidate_name)
            if similarity >= threshold:
                candidates.append({
                    "id": node["id"],
                    "name": node.get("name", ""),
                    "type": node.get("type", "CONCEPT"),
                    "lang": node.get("lang", ""),
                    "confidence": similarity
                })

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates[:limit]

    def neighbors(self, node_id: str, limit: int = 20,
                  relation_types: Optional[List[str]] = None) -> List[Dict]:
        """返回节点的邻居"""
        neighbors = []
        for edge in self.edges.get(node_id, [])[:limit]:
            if relation_types and edge["relation"] not in relation_types:
                continue
            neighbor_node = self.nodes.get(edge["target_id"])
            if not neighbor_node:
                continue
            neighbors.append({
                "source": node_id,
                "target_id": edge["target_id"],
                "relation": edge["relation"],
                "weight": edge["weight"],
                "target_lang": neighbor_node.get("lang", ""),
                "target_name": neighbor_node.get("name", "")
            })

        return neighbors

    def get_node(self, node_id: str) -> Optional[Dict]:
        return self.nodes.get(node_id)

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0

        set_a = set(a)
        set_b = set(b)
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return len(set_a & set_b) / union

