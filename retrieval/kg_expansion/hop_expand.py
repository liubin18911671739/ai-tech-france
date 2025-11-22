"""
N-hop图谱扩展

从链接的实体出发进行N-hop邻居扩展
支持路径过滤和权重衰减
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from logger import get_logger
from config import config
from kg.local_graph import LocalGraphStore

logger = get_logger(__name__)


class HopExpander:
    """N-hop扩展器"""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        max_hops: int = None,
        max_neighbors: int = None
    ):
        """
        初始化扩展器
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名
            max_hops: 最大跳数
            max_neighbors: 每个节点最大邻居数
        """
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.database = database or config.NEO4J_DATABASE
        self.max_hops = max_hops or config.KG_HOP_LIMIT
        self.max_neighbors = max_neighbors or config.KG_MAX_NEIGHBORS
        
        self.driver = None
        self.local_store: Optional[LocalGraphStore] = None
        self.kg_data_dir = config.KG_DATA_DIR
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.info(f"N-hop扩展器连接Neo4j成功")
        except ServiceUnavailable:
            logger.warning("Neo4j服务不可用,使用本地知识图谱扩展")
            self.driver = None
            self._load_local_store()
        except Exception as e:
            logger.warning(f"Neo4j连接失败: {e}, 使用本地知识图谱扩展")
            self.driver = None
            self._load_local_store()

    def _load_local_store(self):
        if self.local_store:
            return
        try:
            self.local_store = LocalGraphStore(self.kg_data_dir)
            logger.info("N-hop扩展器已切换至本地知识图谱模式")
        except FileNotFoundError as exc:
            logger.warning(f"未找到本地知识图谱: {exc}")
            self.local_store = None
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("N-hop扩展器连接已关闭")
    
    def expand_from_nodes(
        self,
        node_ids: List[str],
        hops: int = None,
        relation_types: List[str] = None
    ) -> Dict:
        """
        从给定节点进行N-hop扩展
        
        Args:
            node_ids: 起始节点ID列表
            hops: 扩展跳数(默认使用配置值)
            relation_types: 关系类型过滤(可选)
            
        Returns:
            扩展结果 {"nodes": [...], "edges": [...], "paths": [...]}
        """
        if not node_ids:
            return {"nodes": [], "edges": [], "paths": []}

        if not self.driver and not self.local_store:
            logger.warning("无Neo4j或本地KG,返回空扩展结果")
            return {"nodes": [], "edges": [], "paths": []}

        hops = hops or self.max_hops

        logger.info(f"开始{hops}-hop扩展: {len(node_ids)} 个起始节点")

        if self.driver:
            return self._expand_via_neo4j(node_ids, hops, relation_types)

        return self._expand_locally(node_ids, hops, relation_types)

    def _expand_via_neo4j(self, node_ids: List[str], hops: int, relation_types: Optional[List[str]]):
        # 使用BFS进行逐层扩展
        all_nodes = {}  # {node_id: node_data}
        all_edges = []
        all_paths = []

        visited = set(node_ids)
        current_layer = node_ids
        
        with self.driver.session(database=self.database) as session:
            # 获取起始节点信息
            for node_id in node_ids:
                node_data = self._get_node(session, node_id)
                if node_data:
                    all_nodes[node_id] = node_data
            
            # 逐层扩展
            for hop in range(1, hops + 1):
                logger.info(f"扩展第 {hop} 层, 当前节点数: {len(current_layer)}")
                
                next_layer = set()
                
                for node_id in current_layer:
                    # 获取邻居
                    neighbors = self._get_neighbors(
                        session,
                        node_id,
                        relation_types=relation_types,
                        limit=self.max_neighbors
                    )
                    
                    for neighbor in neighbors:
                        target_id = neighbor["target_id"]
                        
                        # 添加边
                        edge = {
                            "source": node_id,
                            "target": target_id,
                            "relation": neighbor["relation"],
                            "weight": neighbor["weight"],
                            "hop": hop
                        }
                        all_edges.append(edge)
                        
                        # 添加路径
                        path = {
                            "start": node_ids[0] if node_ids else node_id,  # 起始节点
                            "end": target_id,
                            "length": hop,
                            "nodes": [node_id, target_id],
                            "relations": [neighbor["relation"]]
                        }
                        all_paths.append(path)
                        
                        # 如果是新节点,加入下一层
                        if target_id not in visited:
                            visited.add(target_id)
                            next_layer.add(target_id)
                            
                            # 获取节点信息
                            node_data = self._get_node(session, target_id)
                            if node_data:
                                all_nodes[target_id] = node_data
                
                current_layer = list(next_layer)
                
                # 如果没有更多节点,提前结束
                if not current_layer:
                    logger.info(f"第 {hop} 层无新节点,扩展终止")
                    break
        
        result = {
            "nodes": list(all_nodes.values()),
            "edges": all_edges,
            "paths": all_paths
        }

        logger.info(f"扩展完成: {len(result['nodes'])} 个节点, {len(result['edges'])} 条边, {len(result['paths'])} 条路径")
        return result

    def _expand_locally(self, node_ids: List[str], hops: int, relation_types: Optional[List[str]]):
        if not self.local_store:
            return {"nodes": [], "edges": [], "paths": []}

        all_nodes = {}
        all_edges = []
        all_paths = []

        visited = set(node_ids)
        current_layer = list(node_ids)

        for node_id in node_ids:
            node = self.local_store.get_node(node_id)
            if node:
                all_nodes[node_id] = node

        for hop in range(1, hops + 1):
            logger.info(f"[Local] 扩展第 {hop} 层, 当前节点数: {len(current_layer)}")
            next_layer = set()

            for node_id in current_layer:
                neighbors = self.local_store.neighbors(
                    node_id,
                    limit=self.max_neighbors,
                    relation_types=relation_types
                )

                for neighbor in neighbors:
                    target_id = neighbor["target_id"]
                    edge = {
                        "source": node_id,
                        "target": target_id,
                        "relation": neighbor["relation"],
                        "weight": neighbor.get("weight", 1.0),
                        "hop": hop
                    }
                    all_edges.append(edge)

                    path = {
                        "start": node_ids[0],
                        "end": target_id,
                        "length": hop,
                        "relations": [neighbor["relation"]],
                        "weights": [neighbor.get("weight", 1.0)],
                        "node_ids": [node_id, target_id]
                    }
                    all_paths.append(path)

                    if target_id not in visited:
                        visited.add(target_id)
                        next_layer.add(target_id)
                        node_data = self.local_store.get_node(target_id)
                        if node_data:
                            all_nodes[target_id] = node_data

            current_layer = list(next_layer)
            if not current_layer:
                break

        return {
            "nodes": list(all_nodes.values()),
            "edges": all_edges,
            "paths": all_paths
        }

    def expand_multi_entities(
        self,
        entity_ids: List[str],
        max_hops: Optional[int] = None,
        relation_types: Optional[List[str]] = None
    ) -> Dict:
        """对多个实体同时进行扩展并合并结果"""
        if not entity_ids:
            return {"nodes": [], "edges": [], "paths": []}

        combined = {"nodes": [], "edges": [], "paths": []}
        seen_nodes = {}

        for entity_id in entity_ids:
            expanded = self.expand_from_nodes(
                node_ids=[entity_id],
                hops=max_hops,
                relation_types=relation_types
            )

            for node in expanded.get("nodes", []):
                seen_nodes[node["id"]] = node

            combined["edges"].extend(expanded.get("edges", []))
            combined["paths"].extend(expanded.get("paths", []))

        combined["nodes"] = list(seen_nodes.values())
        return combined
    
    def _get_node(self, session, node_id: str) -> Optional[Dict]:
        """获取节点信息"""
        query = """
        MATCH (c:Concept {id: $id})
        RETURN c.id AS id, c.name AS name, c.type AS type, c.lang AS lang
        """
        
        try:
            result = session.run(query, id=node_id)
            record = result.single()
            
            if record:
                return {
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "lang": record.get("lang", "")
                }
            return None
        
        except Exception as e:
            logger.error(f"获取节点失败 ({node_id}): {e}")
            return None
    
    def _get_neighbors(
        self,
        session,
        node_id: str,
        relation_types: List[str] = None,
        limit: int = None
    ) -> List[Dict]:
        """
        获取邻居节点
        
        Args:
            session: Neo4j会话
            node_id: 节点ID
            relation_types: 关系类型过滤
            limit: 邻居数限制
            
        Returns:
            邻居列表
        """
        # 构建查询
        if relation_types:
            rel_filter = "|".join(relation_types)
            query = f"""
            MATCH (source:Concept {{id: $id}})-[r:{rel_filter}]->(target:Concept)
            RETURN target.id AS target_id, type(r) AS relation, r.weight AS weight
            ORDER BY r.weight DESC
            """
        else:
            query = """
            MATCH (source:Concept {id: $id})-[r]->(target:Concept)
            RETURN target.id AS target_id, type(r) AS relation, r.weight AS weight
            ORDER BY r.weight DESC
            """
        
        # 添加限制
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            result = session.run(query, id=node_id)
            neighbors = []
            
            for record in result:
                neighbors.append({
                    "target_id": record["target_id"],
                    "relation": record["relation"],
                    "weight": record.get("weight", 1.0)
                })
            
            return neighbors
        
        except Exception as e:
            logger.error(f"获取邻居失败 ({node_id}): {e}")
            return []
    
    def expand_with_constraints(
        self,
        node_ids: List[str],
        hops: int = None,
        relation_types: List[str] = None,
        target_types: List[str] = None,
        min_weight: float = 0.0
    ) -> Dict:
        """
        带约束的扩展
        
        Args:
            node_ids: 起始节点ID列表
            hops: 扩展跳数
            relation_types: 关系类型过滤
            target_types: 目标节点类型过滤
            min_weight: 最小边权重
            
        Returns:
            扩展结果
        """
        result = self.expand_from_nodes(node_ids, hops, relation_types)
        
        # 过滤节点类型
        if target_types:
            result["nodes"] = [
                node for node in result["nodes"]
                if node["type"] in target_types
            ]
            valid_node_ids = {node["id"] for node in result["nodes"]}
            
            # 过滤边
            result["edges"] = [
                edge for edge in result["edges"]
                if edge["target"] in valid_node_ids
            ]
            
            # 过滤路径
            result["paths"] = [
                path for path in result["paths"]
                if path["end"] in valid_node_ids
            ]
        
        # 过滤边权重
        if min_weight > 0:
            result["edges"] = [
                edge for edge in result["edges"]
                if edge["weight"] >= min_weight
            ]
            
            # 同步过滤路径
            valid_edges = {(e["source"], e["target"]) for e in result["edges"]}
            result["paths"] = [
                path for path in result["paths"]
                if (path["nodes"][-2], path["nodes"][-1]) in valid_edges
            ]
        
        logger.info(f"约束过滤后: {len(result['nodes'])} 个节点, {len(result['edges'])} 条边")
        return result
    
    def get_shortest_paths(
        self,
        start_ids: List[str],
        end_ids: List[str],
        max_length: int = None
    ) -> List[Dict]:
        """
        获取最短路径
        
        Args:
            start_ids: 起始节点ID列表
            end_ids: 终止节点ID列表
            max_length: 最大路径长度
            
        Returns:
            路径列表
        """
        if not self.driver:
            return []
        
        max_length = max_length or self.max_hops
        
        logger.info(f"查找最短路径: {len(start_ids)} -> {len(end_ids)}")
        
        paths = []
        
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH path = shortestPath((start:Concept)-[*..{max_length}]->(end:Concept))
            WHERE start.id IN $start_ids AND end.id IN $end_ids
            RETURN 
                [node IN nodes(path) | node.id] AS node_ids,
                [node IN nodes(path) | node.name] AS node_names,
                [rel IN relationships(path) | type(rel)] AS relations,
                length(path) AS length
            """
            
            try:
                result = session.run(query, start_ids=start_ids, end_ids=end_ids)
                
                for record in result:
                    paths.append({
                        "node_ids": record["node_ids"],
                        "node_names": record["node_names"],
                        "relations": record["relations"],
                        "length": record["length"]
                    })
            
            except Exception as e:
                logger.error(f"最短路径查询失败: {e}")
        
        logger.info(f"找到 {len(paths)} 条路径")
        return paths
    
    def __enter__(self):
        """上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.close()


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="N-hop扩展测试")
    parser.add_argument("--nodes", type=str, nargs="+", required=True,
                       help="起始节点ID")
    parser.add_argument("--hops", type=int, default=2,
                       help="扩展跳数")
    parser.add_argument("--relations", type=str, nargs="+",
                       help="关系类型过滤(可选)")
    parser.add_argument("--output", type=str,
                       help="输出文件路径(JSON)")
    
    args = parser.parse_args()
    
    # 创建扩展器
    expander = HopExpander()
    
    # 扩展
    result = expander.expand_from_nodes(
        node_ids=args.nodes,
        hops=args.hops,
        relation_types=args.relations
    )
    
    # 输出统计
    logger.info(f"\n扩展结果:")
    logger.info(f"  节点数: {len(result['nodes'])}")
    logger.info(f"  边数: {len(result['edges'])}")
    logger.info(f"  路径数: {len(result['paths'])}")
    
    # 输出到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"结果已保存: {args.output}")
    
    expander.close()


if __name__ == "__main__":
    main()
