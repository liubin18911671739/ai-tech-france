"""
实体链接模块

将查询文本中的实体链接到知识图谱
支持多语言实体识别和模糊匹配
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from logger import get_logger
from config import config

logger = get_logger(__name__)


class EntityLinker:
    """实体链接器"""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
        similarity_threshold: float = 0.8
    ):
        """
        初始化实体链接器
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名
            similarity_threshold: 相似度阈值
        """
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.database = database or config.NEO4J_DATABASE
        self.similarity_threshold = similarity_threshold
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.info(f"实体链接器连接Neo4j成功")
        except ServiceUnavailable:
            logger.error("Neo4j服务不可用,实体链接功能将不可用")
            self.driver = None
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            self.driver = None
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("实体链接器连接已关闭")
    
    def link_entities(
        self,
        entities: List[Dict],
        lang: Optional[str] = None
    ) -> List[Dict]:
        """
        链接实体到知识图谱
        
        Args:
            entities: 识别的实体列表 [{"entity": "...", "type": "...", "score": ...}, ...]
            lang: 语言(可选,用于过滤)
            
        Returns:
            链接结果 [{"entity": ..., "kg_id": ..., "kg_name": ..., "confidence": ...}, ...]
        """
        if not self.driver:
            logger.warning("Neo4j未连接,返回空链接结果")
            return []
        
        if not entities:
            return []
        
        logger.info(f"开始链接实体: {len(entities)} 个")
        
        linked = []
        
        with self.driver.session(database=self.database) as session:
            for entity in entities:
                entity_name = entity.get("entity", "")
                entity_type = entity.get("type", "")
                
                if not entity_name:
                    continue
                
                # 尝试精确匹配
                kg_nodes = self._exact_match(session, entity_name, entity_type, lang)
                
                # 如果精确匹配失败,尝试模糊匹配
                if not kg_nodes:
                    kg_nodes = self._fuzzy_match(session, entity_name, entity_type, lang)
                
                # 选择最佳匹配
                if kg_nodes:
                    best_match = kg_nodes[0]  # 已按相似度排序
                    
                    linked.append({
                        "entity": entity_name,
                        "entity_type": entity_type,
                        "kg_id": best_match["id"],
                        "kg_name": best_match["name"],
                        "kg_type": best_match["type"],
                        "confidence": best_match["confidence"]
                    })
                else:
                    logger.debug(f"实体未链接: {entity_name} ({entity_type})")
        
        logger.info(f"实体链接完成: {len(linked)}/{len(entities)} 个成功")
        return linked
    
    def _exact_match(
        self,
        session,
        entity_name: str,
        entity_type: str = None,
        lang: str = None
    ) -> List[Dict]:
        """
        精确匹配
        
        Args:
            session: Neo4j会话
            entity_name: 实体名称
            entity_type: 实体类型(可选)
            lang: 语言(可选)
            
        Returns:
            匹配的节点列表
        """
        # 构建查询
        query = "MATCH (c:Concept {name: $name})"
        params = {"name": entity_name}
        
        # 添加类型过滤
        if entity_type:
            query += " WHERE c.type = $type"
            params["type"] = entity_type
        
        # 添加语言过滤
        if lang:
            if entity_type:
                query += " AND c.lang = $lang"
            else:
                query += " WHERE c.lang = $lang"
            params["lang"] = lang
        
        query += " RETURN c.id AS id, c.name AS name, c.type AS type, c.lang AS lang"
        
        try:
            result = session.run(query, **params)
            nodes = []
            for record in result:
                nodes.append({
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "lang": record.get("lang", ""),
                    "confidence": 1.0  # 精确匹配置信度1.0
                })
            return nodes
        except Exception as e:
            logger.error(f"精确匹配失败: {e}")
            return []
    
    def _fuzzy_match(
        self,
        session,
        entity_name: str,
        entity_type: str = None,
        lang: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        模糊匹配(基于字符串相似度)
        
        Args:
            session: Neo4j会话
            entity_name: 实体名称
            entity_type: 实体类型(可选)
            lang: 语言(可选)
            limit: 返回结果数
            
        Returns:
            匹配的节点列表(按相似度降序)
        """
        # 使用CONTAINS进行模糊匹配
        query = "MATCH (c:Concept) WHERE c.name CONTAINS $name"
        params = {"name": entity_name, "limit": limit}
        
        # 添加类型过滤
        if entity_type:
            query += " AND c.type = $type"
            params["type"] = entity_type
        
        # 添加语言过滤
        if lang:
            query += " AND c.lang = $lang"
            params["lang"] = lang
        
        query += " RETURN c.id AS id, c.name AS name, c.type AS type, c.lang AS lang LIMIT $limit"
        
        try:
            result = session.run(query, **params)
            nodes = []
            for record in result:
                # 计算简单的字符串相似度
                similarity = self._string_similarity(entity_name, record["name"])
                
                if similarity >= self.similarity_threshold:
                    nodes.append({
                        "id": record["id"],
                        "name": record["name"],
                        "type": record["type"],
                        "lang": record.get("lang", ""),
                        "confidence": similarity
                    })
            
            # 按相似度降序排序
            nodes.sort(key=lambda x: x["confidence"], reverse=True)
            return nodes
        
        except Exception as e:
            logger.error(f"模糊匹配失败: {e}")
            return []
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        计算字符串相似度(简单Jaccard相似度)
        
        Args:
            s1: 字符串1
            s2: 字符串2
            
        Returns:
            相似度 [0, 1]
        """
        # 转小写
        s1 = s1.lower()
        s2 = s2.lower()
        
        # 字符集合
        set1 = set(s1)
        set2 = set(s2)
        
        # Jaccard相似度
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def link_query(
        self,
        query: str,
        ner_model = None,
        lang: str = None
    ) -> List[Dict]:
        """
        对查询文本进行实体链接
        
        Args:
            query: 查询文本
            ner_model: NER模型(可选,如果提供则先识别实体)
            lang: 语言
            
        Returns:
            链接结果
        """
        # 如果提供NER模型,先识别实体
        if ner_model:
            entities = ner_model.extract_entities(query)
        else:
            # 否则将整个查询作为实体
            entities = [{"entity": query, "type": "CONCEPT", "score": 1.0}]
        
        # 链接实体
        linked = self.link_entities(entities, lang=lang)
        
        return linked
    
    def batch_link_queries(
        self,
        queries: List[Dict],
        ner_model = None
    ) -> Dict[str, List[Dict]]:
        """
        批量链接查询
        
        Args:
            queries: 查询列表 [{"qid": ..., "query": ..., "lang": ...}, ...]
            ner_model: NER模型
            
        Returns:
            链接结果 {qid: [linked_entities], ...}
        """
        logger.info(f"批量实体链接: {len(queries)} 个查询")
        
        results = {}
        
        for q in queries:
            qid = q.get("qid", "")
            query = q.get("query", "")
            lang = q.get("lang", None)
            
            linked = self.link_query(query, ner_model=ner_model, lang=lang)
            results[qid] = linked
        
        logger.info(f"批量链接完成")
        return results
    
    def __enter__(self):
        """上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.close()


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="实体链接测试")
    parser.add_argument("--query", type=str,
                       help="查询文本")
    parser.add_argument("--lang", type=str,
                       help="语言(fr/zh/en)")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="相似度阈值")
    
    args = parser.parse_args()
    
    # 创建链接器
    linker = EntityLinker(similarity_threshold=args.threshold)
    
    if args.query:
        logger.info(f"\n查询: {args.query}")
        
        # 简单分词(实际应用应使用NER)
        entities = [{"entity": word, "type": "CONCEPT"} for word in args.query.split()]
        
        # 链接
        linked = linker.link_entities(entities, lang=args.lang)
        
        logger.info(f"\n链接结果: {len(linked)} 个")
        for item in linked:
            logger.info(f"  {item['entity']} -> {item['kg_name']} ({item['kg_type']}, confidence={item['confidence']:.2f})")
    
    else:
        logger.info("请使用 --query 参数提供查询文本")
    
    linker.close()


if __name__ == "__main__":
    main()
