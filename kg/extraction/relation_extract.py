"""
关系抽取 - 规则模板 + LLM辅助

提取概念间的关系:
- prerequisite (前置知识)
- related_to (相关概念)
- has_resource (资源关联)
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from logger import get_logger
from config import config

logger = get_logger(__name__)


class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self):
        """初始化关系抽取规则"""
        self.patterns = self._load_patterns()
        logger.info("关系抽取器初始化完成")
    
    def _load_patterns(self) -> Dict[str, List[re.Pattern]]:
        """
        加载关系模式
        
        Returns:
            关系类型 -> 正则模式列表
        """
        patterns = {
            "prerequisite": [
                # 法语模式
                re.compile(r"(\w+)\s+(?:avant|précède|nécessite)\s+(\w+)", re.IGNORECASE),
                re.compile(r"pour\s+(?:apprendre|comprendre)\s+(\w+),\s+(?:il faut|on doit)\s+(?:connaître|maîtriser)\s+(\w+)", re.IGNORECASE),
                
                # 中文模式
                re.compile(r"(\S+)(?:是|为)(\S+)的(?:基础|前提|先决条件)"),
                re.compile(r"学习(\S+)(?:之前|前)需要(?:掌握|了解)(\S+)"),
                re.compile(r"(\S+)→(\S+)"),  # 箭头表示
                
                # 英语模式
                re.compile(r"(\w+)\s+(?:before|precedes|requires)\s+(\w+)", re.IGNORECASE),
                re.compile(r"(\w+)\s+is\s+(?:a\s+)?prerequisite\s+(?:for|of)\s+(\w+)", re.IGNORECASE),
            ],
            
            "related_to": [
                # 法语
                re.compile(r"(\w+)\s+(?:et|ou)\s+(\w+)\s+sont\s+(?:liés|similaires)", re.IGNORECASE),
                re.compile(r"(\w+)\s+(?:concerne|se rapporte à)\s+(\w+)", re.IGNORECASE),
                
                # 中文
                re.compile(r"(\S+)(?:与|和)(\S+)(?:相关|有关|类似)"),
                re.compile(r"(\S+)(?:涉及|包含)(\S+)"),
                
                # 英语
                re.compile(r"(\w+)\s+(?:and|or)\s+(\w+)\s+are\s+related", re.IGNORECASE),
                re.compile(r"(\w+)\s+(?:relates to|concerns)\s+(\w+)", re.IGNORECASE),
            ],
            
            "has_resource": [
                # 通用模式
                re.compile(r"(\S+):\s*(.+?\.(?:pdf|doc|html|mp4))", re.IGNORECASE),
                re.compile(r"资源.*?(\S+).*?(doc_\d+)", re.IGNORECASE),
            ]
        }
        return patterns
    
    def extract_relations(
        self,
        text: str,
        entities: List[Dict],
        doc_id: str = None
    ) -> List[Dict]:
        """
        从文本中提取关系
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表
            doc_id: 文档ID
            
        Returns:
            关系列表 [{"relation": "prerequisite", "source": "alphabet", "target": "syllable", "confidence": 0.9}]
        """
        relations = []
        
        # 规则匹配
        for rel_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    if len(match.groups()) >= 2:
                        source, target = match.group(1), match.group(2)
                        
                        # 验证source和target是否在实体列表中
                        if self._is_valid_entity(source, entities) or self._is_valid_entity(target, entities):
                            relations.append({
                                "relation": rel_type,
                                "source": source.strip(),
                                "target": target.strip(),
                                "confidence": 0.8,
                                "method": "rule",
                                "doc_id": doc_id
                            })
        
        # 去重
        relations = self._deduplicate_relations(relations)
        
        return relations
    
    def _is_valid_entity(self, text: str, entities: List[Dict]) -> bool:
        """检查文本是否在实体列表中"""
        text_lower = text.lower()
        for entity in entities:
            if text_lower in entity.get("entity", "").lower():
                return True
        return True  # 放宽条件,允许不在NER中的概念
    
    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """关系去重"""
        seen = set()
        dedup = []
        
        for rel in relations:
            key = (rel["relation"], rel["source"], rel["target"])
            if key not in seen:
                seen.add(key)
                dedup.append(rel)
        
        return dedup
    
    def extract_from_corpus(
        self,
        entities_path: Path,
        corpus_path: Path,
        output_path: Path = None
    ) -> List[Dict]:
        """
        从语料库批量提取关系
        
        Args:
            entities_path: 实体文件路径
            corpus_path: 原始语料路径
            output_path: 输出路径
            
        Returns:
            关系列表
        """
        logger.info(f"开始提取关系: entities={entities_path}, corpus={corpus_path}")
        
        # 加载实体
        entities_map = {}
        with open(entities_path, 'r', encoding='utf-8') as f:
            for line in f:
                entity_doc = json.loads(line.strip())
                entities_map[entity_doc["doc_id"]] = entity_doc["entities"]
        
        # 处理语料
        all_relations = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    doc_id = doc.get("doc_id", f"doc_{line_num}")
                    text = doc.get("content", "")
                    
                    entities = entities_map.get(doc_id, [])
                    
                    # 提取关系
                    relations = self.extract_relations(text, entities, doc_id)
                    all_relations.extend(relations)
                    
                    if line_num % 100 == 0:
                        logger.info(f"已处理 {line_num} 篇文档,提取 {len(all_relations)} 个关系")
                
                except Exception as e:
                    logger.error(f"处理第{line_num}行失败: {e}")
                    continue
        
        logger.info(f"关系提取完成,共 {len(all_relations)} 个关系")
        
        # 保存结果
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for rel in all_relations:
                    f.write(json.dumps(rel, ensure_ascii=False) + '\n')
            logger.info(f"结果已保存至: {output_path}")
        
        return all_relations


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="关系抽取")
    parser.add_argument("--entities", type=str, required=True, help="实体文件路径")
    parser.add_argument("--corpus", type=str, required=True, help="语料文件路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--text", type=str, help="测试单个文本")
    
    args = parser.parse_args()
    
    extractor = RelationExtractor()
    
    # 单文本测试
    if args.text:
        logger.info(f"测试文本: {args.text}")
        relations = extractor.extract_relations(args.text, [])
        logger.info(f"提取关系: {relations}")
        return
    
    # 批量处理
    entities_path = Path(args.entities)
    corpus_path = Path(args.corpus)
    output_path = Path(args.output) if args.output else corpus_path.parent / "relations.jsonl"
    
    extractor.extract_from_corpus(entities_path, corpus_path, output_path)


if __name__ == "__main__":
    main()
