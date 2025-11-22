"""
构建图谱节点和关系

从NER和关系抽取结果构建知识图谱的节点和边
"""
import json
from pathlib import Path
from typing import List, Dict, Set, Tuple
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """图谱构建器"""
    
    def __init__(self):
        """初始化构建器"""
        self.concepts = {}  # {concept_id: concept_info}
        self.relations = []  # [relation_info]
        self.concept_counter = defaultdict(int)
        
        logger.info("图谱构建器初始化")
    
    def add_concept(
        self,
        name: str,
        lang: str,
        concept_type: str = "CONCEPT",
        properties: Dict = None
    ) -> str:
        """
        添加概念节点
        
        Args:
            name: 概念名称
            lang: 语言
            concept_type: 概念类型
            properties: 其他属性
            
        Returns:
            concept_id
        """
        # 生成概念ID (去重)
        key = f"{name}_{lang}".lower()
        
        if key in self.concepts:
            return self.concepts[key]["id"]
        
        # 生成新ID
        concept_id = f"{lang}_{concept_type}_{self.concept_counter[concept_type]:06d}"
        self.concept_counter[concept_type] += 1
        
        # 创建概念
        concept = {
            "id": concept_id,
            "name": name,
            "lang": lang,
            "type": concept_type,
            "properties": properties or {}
        }
        
        self.concepts[key] = concept
        return concept_id
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Dict = None
    ):
        """
        添加关系
        
        Args:
            source_id: 源概念ID
            target_id: 目标概念ID
            relation_type: 关系类型
            properties: 其他属性
        """
        relation = {
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "properties": properties or {}
        }
        
        self.relations.append(relation)
    
    def build_from_entities(
        self,
        entities_file: Path,
        lang: str
    ):
        """
        从实体文件构建节点
        
        Args:
            entities_file: 实体文件(JSONL)
            lang: 语言
        """
        logger.info(f"从实体文件构建节点: {entities_file}")
        
        if not entities_file.exists():
            logger.warning(f"实体文件不存在: {entities_file}")
            return
        
        count = 0
        with open(entities_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    
                    # 获取实体信息
                    doc_id = obj.get("doc_id", "")
                    entities = obj.get("entities", [])
                    
                    for entity in entities:
                        name = entity.get("text", "")
                        entity_type = entity.get("type", "CONCEPT")
                        
                        if not name:
                            continue
                        
                        # 添加概念
                        properties = {
                            "source_doc": doc_id,
                            "frequency": 1
                        }
                        
                        concept_id = self.add_concept(
                            name=name,
                            lang=lang,
                            concept_type=entity_type,
                            properties=properties
                        )
                        
                        count += 1
                
                except Exception as e:
                    logger.error(f"处理实体失败: {e}")
                    continue
        
        logger.info(f"从实体文件添加 {count} 个节点")
    
    def build_from_relations(
        self,
        relations_file: Path
    ):
        """
        从关系文件构建边
        
        Args:
            relations_file: 关系文件(JSONL)
        """
        logger.info(f"从关系文件构建边: {relations_file}")
        
        if not relations_file.exists():
            logger.warning(f"关系文件不存在: {relations_file}")
            return
        
        count = 0
        with open(relations_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    
                    # 获取关系信息
                    relations = obj.get("relations", [])
                    
                    for relation in relations:
                        head = relation.get("head", "")
                        tail = relation.get("tail", "")
                        rel_type = relation.get("type", "RELATED_TO")
                        
                        if not head or not tail:
                            continue
                        
                        # 查找概念ID
                        head_id = self._find_concept_id(head)
                        tail_id = self._find_concept_id(tail)
                        
                        if not head_id or not tail_id:
                            continue
                        
                        # 添加关系
                        self.add_relation(
                            source_id=head_id,
                            target_id=tail_id,
                            relation_type=rel_type
                        )
                        
                        count += 1
                
                except Exception as e:
                    logger.error(f"处理关系失败: {e}")
                    continue
        
        logger.info(f"从关系文件添加 {count} 条边")
    
    def _find_concept_id(self, name: str) -> str:
        """
        根据名称查找概念ID
        
        Args:
            name: 概念名称
            
        Returns:
            concept_id 或 None
        """
        name_lower = name.lower()
        
        # 尝试匹配所有语言
        for key, concept in self.concepts.items():
            if concept["name"].lower() == name_lower:
                return concept["id"]
        
        return None
    
    def add_alignment_relations(
        self,
        alignment_file: Path
    ):
        """
        添加对齐关系
        
        Args:
            alignment_file: 对齐文件(TSV)
        """
        logger.info(f"添加对齐关系: {alignment_file}")
        
        if not alignment_file.exists():
            logger.warning(f"对齐文件不存在: {alignment_file}")
            return
        
        count = 0
        with open(alignment_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                concept1 = parts[0]
                concept2 = parts[1]
                
                # 查找概念ID
                id1 = self._find_concept_id(concept1)
                id2 = self._find_concept_id(concept2)
                
                if not id1 or not id2:
                    continue
                
                # 添加双向对齐关系
                self.add_relation(
                    source_id=id1,
                    target_id=id2,
                    relation_type="ALIGNED_WITH",
                    properties={"bidirectional": True}
                )
                
                count += 1
        
        logger.info(f"添加 {count} 条对齐关系")
    
    def export_nodes(self, output_file: Path):
        """
        导出节点
        
        Args:
            output_file: 输出文件(JSONL)
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for concept in self.concepts.values():
                f.write(json.dumps(concept, ensure_ascii=False) + '\n')
        
        logger.info(f"导出 {len(self.concepts)} 个节点: {output_file}")
    
    def export_relations(self, output_file: Path):
        """
        导出关系
        
        Args:
            output_file: 输出文件(JSONL)
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for relation in self.relations:
                f.write(json.dumps(relation, ensure_ascii=False) + '\n')
        
        logger.info(f"导出 {len(self.relations)} 条关系: {output_file}")
    
    def get_statistics(self) -> Dict:
        """
        获取图谱统计
        
        Returns:
            统计信息
        """
        # 按语言统计
        lang_stats = defaultdict(int)
        type_stats = defaultdict(int)
        
        for concept in self.concepts.values():
            lang_stats[concept["lang"]] += 1
            type_stats[concept["type"]] += 1
        
        # 按关系类型统计
        rel_stats = defaultdict(int)
        for relation in self.relations:
            rel_stats[relation["type"]] += 1
        
        stats = {
            "total_concepts": len(self.concepts),
            "total_relations": len(self.relations),
            "concepts_by_lang": dict(lang_stats),
            "concepts_by_type": dict(type_stats),
            "relations_by_type": dict(rel_stats)
        }
        
        return stats
    
    def build_mock_graph(self):
        """构建Mock图谱(用于测试)"""
        logger.info("构建Mock图谱")
        
        # 法语概念
        fr_ml = self.add_concept("apprentissage automatique", "fr", "CONCEPT")
        fr_dl = self.add_concept("apprentissage profond", "fr", "CONCEPT")
        fr_nn = self.add_concept("réseaux de neurones", "fr", "CONCEPT")
        fr_cnn = self.add_concept("réseaux convolutifs", "fr", "CONCEPT")
        
        # 中文概念
        zh_ml = self.add_concept("机器学习", "zh", "CONCEPT")
        zh_dl = self.add_concept("深度学习", "zh", "CONCEPT")
        zh_nn = self.add_concept("神经网络", "zh", "CONCEPT")
        zh_cnn = self.add_concept("卷积神经网络", "zh", "CONCEPT")
        
        # 英文概念
        en_ml = self.add_concept("machine learning", "en", "CONCEPT")
        en_dl = self.add_concept("deep learning", "en", "CONCEPT")
        en_nn = self.add_concept("neural networks", "en", "CONCEPT")
        en_cnn = self.add_concept("convolutional neural networks", "en", "CONCEPT")
        
        # 层级关系
        self.add_relation(fr_dl, fr_ml, "IS_A")
        self.add_relation(fr_cnn, fr_dl, "IS_A")
        self.add_relation(fr_nn, fr_ml, "RELATED_TO")
        
        self.add_relation(zh_dl, zh_ml, "IS_A")
        self.add_relation(zh_cnn, zh_dl, "IS_A")
        self.add_relation(zh_nn, zh_ml, "RELATED_TO")
        
        self.add_relation(en_dl, en_ml, "IS_A")
        self.add_relation(en_cnn, en_dl, "IS_A")
        self.add_relation(en_nn, en_ml, "RELATED_TO")
        
        # 对齐关系
        self.add_relation(fr_ml, zh_ml, "ALIGNED_WITH")
        self.add_relation(zh_ml, en_ml, "ALIGNED_WITH")
        self.add_relation(fr_ml, en_ml, "ALIGNED_WITH")
        
        self.add_relation(fr_dl, zh_dl, "ALIGNED_WITH")
        self.add_relation(zh_dl, en_dl, "ALIGNED_WITH")
        self.add_relation(fr_dl, en_dl, "ALIGNED_WITH")
        
        self.add_relation(fr_nn, zh_nn, "ALIGNED_WITH")
        self.add_relation(zh_nn, en_nn, "ALIGNED_WITH")
        
        self.add_relation(fr_cnn, zh_cnn, "ALIGNED_WITH")
        self.add_relation(zh_cnn, en_cnn, "ALIGNED_WITH")
        
        logger.info(f"Mock图谱构建完成: {len(self.concepts)} 个节点, {len(self.relations)} 条边")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="构建知识图谱节点和关系")
    parser.add_argument("--entities-fr", type=str,
                       help="法语实体文件")
    parser.add_argument("--entities-zh", type=str,
                       help="中文实体文件")
    parser.add_argument("--entities-en", type=str,
                       help="英文实体文件")
    parser.add_argument("--relations-fr", type=str,
                       help="法语关系文件")
    parser.add_argument("--relations-zh", type=str,
                       help="中文关系文件")
    parser.add_argument("--relations-en", type=str,
                       help="英文关系文件")
    parser.add_argument("--alignment", type=str,
                       help="对齐文件")
    parser.add_argument("--output-nodes", type=str,
                       default="data/kg/nodes.jsonl",
                       help="输出节点文件")
    parser.add_argument("--output-rels", type=str,
                       default="data/kg/relations.jsonl",
                       help="输出关系文件")
    parser.add_argument("--mock", action="store_true",
                       help="构建Mock图谱")
    
    args = parser.parse_args()
    
    # 创建构建器
    builder = GraphBuilder()
    
    if args.mock:
        # Mock模式
        builder.build_mock_graph()
    else:
        # 从文件构建
        if args.entities_fr:
            builder.build_from_entities(Path(args.entities_fr), "fr")
        if args.entities_zh:
            builder.build_from_entities(Path(args.entities_zh), "zh")
        if args.entities_en:
            builder.build_from_entities(Path(args.entities_en), "en")
        
        if args.relations_fr:
            builder.build_from_relations(Path(args.relations_fr))
        if args.relations_zh:
            builder.build_from_relations(Path(args.relations_zh))
        if args.relations_en:
            builder.build_from_relations(Path(args.relations_en))
        
        if args.alignment:
            builder.add_alignment_relations(Path(args.alignment))
    
    # 导出
    builder.export_nodes(Path(args.output_nodes))
    builder.export_relations(Path(args.output_rels))
    
    # 统计
    stats = builder.get_statistics()
    logger.info("\n=== 图谱统计 ===")
    logger.info(f"总节点数: {stats['total_concepts']}")
    logger.info(f"总边数: {stats['total_relations']}")
    logger.info(f"按语言: {stats['concepts_by_lang']}")
    logger.info(f"按类型: {stats['concepts_by_type']}")
    logger.info(f"关系类型: {stats['relations_by_type']}")


if __name__ == "__main__":
    main()
