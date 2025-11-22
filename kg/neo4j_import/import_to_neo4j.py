"""
Neo4j导入工具

将实体和关系导入到Neo4j知识图谱
支持批量导入和增量更新
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from logger import get_logger
from config import config

logger = get_logger(__name__)


class Neo4jImporter:
    """Neo4j导入器"""
    
    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None
    ):
        """
        初始化Neo4j连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名
        """
        self.uri = uri or config.NEO4J_URI
        self.user = user or config.NEO4J_USER
        self.password = password or config.NEO4J_PASSWORD
        self.database = database or config.NEO4J_DATABASE
        
        self.driver = None
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # 测试连接
            self.driver.verify_connectivity()
            logger.info(f"Neo4j连接成功: {self.uri}")
        
        except AuthError:
            logger.error("Neo4j认证失败,请检查用户名密码")
            raise
        except ServiceUnavailable:
            logger.error("Neo4j服务不可用,请检查URI或启动Neo4j")
            raise
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            raise
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    
    def clear_database(self):
        """清空数据库(谨慎使用)"""
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("数据库已清空")
    
    def create_constraints(self):
        """创建约束和索引"""
        constraints = [
            # 概念唯一性约束
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            # 实体唯一性约束
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            # 名称索引(加速搜索)
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            # 语言索引
            "CREATE INDEX concept_lang IF NOT EXISTS FOR (c:Concept) ON (c.lang)",
            "CREATE INDEX entity_lang IF NOT EXISTS FOR (e:Entity) ON (e.lang)"
        ]
        
        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"约束/索引创建成功: {constraint[:50]}...")
                except Exception as e:
                    logger.warning(f"约束/索引已存在或创建失败: {e}")
    
    def import_concepts(
        self,
        concepts: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        导入概念节点
        
        Args:
            concepts: 概念列表 [{"id": ..., "name": ..., "type": ..., "lang": ...}, ...]
            batch_size: 批处理大小
            
        Returns:
            导入的概念数
        """
        logger.info(f"开始导入概念: {len(concepts)} 个")
        
        query = """
        UNWIND $concepts AS concept
        MERGE (c:Concept {id: concept.id})
        SET c.name = concept.name,
            c.type = concept.type,
            c.lang = concept.lang,
            c.description = concept.description,
            c.updated_at = timestamp()
        """
        
        count = 0
        with self.driver.session(database=self.database) as session:
            for i in range(0, len(concepts), batch_size):
                batch = concepts[i:i+batch_size]
                try:
                    session.run(query, concepts=batch)
                    count += len(batch)
                    if (i + batch_size) % 1000 == 0:
                        logger.info(f"已导入 {count}/{len(concepts)} 个概念")
                except Exception as e:
                    logger.error(f"批次导入失败 ({i}-{i+batch_size}): {e}")
        
        logger.info(f"概念导入完成: {count} 个")
        return count
    
    def import_relations(
        self,
        relations: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        导入关系
        
        Args:
            relations: 关系列表 [{"source": ..., "target": ..., "type": ..., "weight": ...}, ...]
            batch_size: 批处理大小
            
        Returns:
            导入的关系数
        """
        logger.info(f"开始导入关系: {len(relations)} 条")
        
        # 按关系类型分组导入
        relation_types = {}
        for rel in relations:
            rel_type = rel.get("type", "RELATED_TO")
            if rel_type not in relation_types:
                relation_types[rel_type] = []
            relation_types[rel_type].append(rel)
        
        count = 0
        with self.driver.session(database=self.database) as session:
            for rel_type, rels in relation_types.items():
                logger.info(f"导入关系类型: {rel_type} ({len(rels)} 条)")
                
                # 动态构建Cypher查询
                query = f"""
                UNWIND $relations AS rel
                MATCH (source:Concept {{id: rel.source}})
                MATCH (target:Concept {{id: rel.target}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r.weight = rel.weight,
                    r.confidence = rel.confidence,
                    r.updated_at = timestamp()
                """
                
                for i in range(0, len(rels), batch_size):
                    batch = rels[i:i+batch_size]
                    try:
                        session.run(query, relations=batch)
                        count += len(batch)
                    except Exception as e:
                        logger.error(f"关系导入失败 ({rel_type}, {i}-{i+batch_size}): {e}")
        
        logger.info(f"关系导入完成: {count} 条")
        return count
    
    def import_from_files(
        self,
        concepts_file: Path,
        relations_file: Path,
        clear_first: bool = False
    ):
        """
        从文件导入
        
        Args:
            concepts_file: 概念文件(JSONL)
            relations_file: 关系文件(JSONL)
            clear_first: 是否先清空数据库
        """
        logger.info("=" * 60)
        logger.info("开始从文件导入Neo4j")
        logger.info("=" * 60)
        
        # 清空数据库
        if clear_first:
            logger.warning("清空数据库...")
            self.clear_database()
        
        # 创建约束
        logger.info("创建约束和索引...")
        self.create_constraints()
        
        # 加载概念
        concepts = []
        logger.info(f"加载概念文件: {concepts_file}")
        with open(concepts_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    concept = json.loads(line.strip())
                    concepts.append(concept)
                except Exception as e:
                    logger.error(f"解析概念失败: {e}")
        
        logger.info(f"加载概念数: {len(concepts)}")
        
        # 加载关系
        relations = []
        logger.info(f"加载关系文件: {relations_file}")
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    relation = json.loads(line.strip())
                    relations.append(relation)
                except Exception as e:
                    logger.error(f"解析关系失败: {e}")
        
        logger.info(f"加载关系数: {len(relations)}")
        
        # 导入概念
        concept_count = self.import_concepts(concepts)
        
        # 导入关系
        relation_count = self.import_relations(relations)
        
        # 统计
        stats = self.get_statistics()
        
        logger.info("\n" + "=" * 60)
        logger.info("导入完成")
        logger.info("=" * 60)
        logger.info(f"概念节点: {stats['concepts']}")
        logger.info(f"关系数: {stats['relations']}")
        logger.info(f"关系类型: {stats['relation_types']}")
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        with self.driver.session(database=self.database) as session:
            # 概念数
            result = session.run("MATCH (c:Concept) RETURN count(c) AS count")
            concept_count = result.single()["count"]
            
            # 关系数
            result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            relation_count = result.single()["count"]
            
            # 关系类型
            result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS count"
            )
            relation_types = {record["type"]: record["count"] for record in result}
            
            # 语言分布
            result = session.run(
                "MATCH (c:Concept) RETURN c.lang AS lang, count(c) AS count"
            )
            lang_dist = {record["lang"]: record["count"] for record in result}
        
        return {
            "concepts": concept_count,
            "relations": relation_count,
            "relation_types": relation_types,
            "language_distribution": lang_dist
        }
    
    def test_query(self, concept_name: str, limit: int = 5):
        """
        测试查询
        
        Args:
            concept_name: 概念名称
            limit: 返回结果数
        """
        query = """
        MATCH (c:Concept {name: $name})-[r]->(target:Concept)
        RETURN c.name AS source, type(r) AS relation, target.name AS target, r.weight AS weight
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, name=concept_name, limit=limit)
            
            logger.info(f"\n测试查询: {concept_name}")
            for record in result:
                logger.info(f"  {record['source']} --[{record['relation']}]-> {record['target']} (weight: {record['weight']})")
    
    def __enter__(self):
        """上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.close()


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="导入知识图谱到Neo4j")
    parser.add_argument("--concepts", type=str, required=True,
                       help="概念文件路径(JSONL)")
    parser.add_argument("--relations", type=str, required=True,
                       help="关系文件路径(JSONL)")
    parser.add_argument("--clear", action="store_true",
                       help="导入前清空数据库")
    parser.add_argument("--test", type=str,
                       help="测试查询(概念名)")
    
    args = parser.parse_args()
    
    # 导入
    with Neo4jImporter() as importer:
        importer.import_from_files(
            concepts_file=Path(args.concepts),
            relations_file=Path(args.relations),
            clear_first=args.clear
        )
        
        # 统计
        stats = importer.get_statistics()
        logger.info("\n知识图谱统计:")
        logger.info(f"  概念数: {stats['concepts']}")
        logger.info(f"  关系数: {stats['relations']}")
        logger.info(f"  关系类型: {stats['relation_types']}")
        logger.info(f"  语言分布: {stats['language_distribution']}")
        
        # 测试查询
        if args.test:
            importer.test_query(args.test)


if __name__ == "__main__":
    main()
