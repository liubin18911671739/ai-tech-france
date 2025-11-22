#!/usr/bin/env python3
"""
批量关系提取

从清洗后的语料中提取实体关系
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from kg.extraction.rel_extract import RelationExtractor

logger = get_logger(__name__)


class BatchRelationExtractor:
    """批量关系提取器"""
    
    def __init__(self):
        """初始化提取器"""
        self.extractor = RelationExtractor()
        logger.info("关系提取器初始化")
    
    def extract_from_file(
        self,
        input_file: Path,
        output_file: Path,
        lang: str,
        batch_size: int = 32
    ):
        """
        从文件提取关系
        
        Args:
            input_file: 输入语料文件(JSONL)
            output_file: 输出关系文件(JSONL)
            lang: 语言
            batch_size: 批大小
        """
        logger.info(f"从 {input_file} 提取关系")
        
        if not input_file.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取文档
        documents = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    documents.append(obj)
                except Exception as e:
                    logger.error(f"解析文档失败: {e}")
                    continue
        
        logger.info(f"加载 {len(documents)} 个文档")
        
        # 批量提取
        total_relations = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                for doc in batch:
                    doc_id = doc.get("doc_id", "")
                    text = doc.get("content") or doc.get("text", "")
                    
                    if not text:
                        continue
                    
                    # 提取关系
                    relations = self.extractor.extract_relations(text, lang=lang)
                    
                    # 写入结果
                    result = {
                        "doc_id": doc_id,
                        "lang": lang,
                        "relations": relations
                    }
                    
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    total_relations += len(relations)
                
                if (i + batch_size) % 100 == 0:
                    logger.info(f"已处理 {i + batch_size} / {len(documents)} 个文档")
        
        logger.info(f"提取完成: {total_relations} 条关系 -> {output_file}")
    
    def extract_from_entities(
        self,
        entities_file: Path,
        corpus_file: Path,
        output_file: Path,
        lang: str
    ):
        """
        从实体文件提取关系
        
        Args:
            entities_file: 实体文件
            corpus_file: 语料文件
            output_file: 输出关系文件
            lang: 语言
        """
        logger.info(f"从实体文件提取关系: {entities_file}")
        
        # 加载语料
        corpus = {}
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        doc_id = obj.get("doc_id", "")
                        corpus[doc_id] = obj.get("content") or obj.get("text", "")
                    except:
                        continue
        
        logger.info(f"加载 {len(corpus)} 个文档")
        
        # 确保输出目录存在
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 读取实体
        total_relations = 0
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            with open(entities_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    try:
                        obj = json.loads(line.strip())
                        doc_id = obj.get("doc_id", "")
                        entities = obj.get("entities", [])
                        
                        if not entities:
                            continue
                        
                        # 获取文档文本
                        text = corpus.get(doc_id, "")
                        if not text:
                            continue
                        
                        # 提取关系
                        relations = self.extractor.extract_relations_from_entities(
                            text=text,
                            entities=entities,
                            lang=lang
                        )
                        
                        # 写入结果
                        result = {
                            "doc_id": doc_id,
                            "lang": lang,
                            "relations": relations
                        }
                        
                        out_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        
                        total_relations += len(relations)
                    
                    except Exception as e:
                        logger.error(f"处理失败: {e}")
                        continue
        
        logger.info(f"提取完成: {total_relations} 条关系 -> {output_file}")
    
    def extract_mock_relations(
        self,
        output_file: Path,
        lang: str,
        num_docs: int = 50
    ):
        """
        生成Mock关系数据
        
        Args:
            output_file: 输出文件
            lang: 语言
            num_docs: 文档数
        """
        logger.info(f"生成Mock关系数据: {lang}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Mock关系模板
        templates = {
            "fr": [
                ("apprentissage profond", "apprentissage automatique", "IS_A"),
                ("réseaux de neurones", "apprentissage automatique", "RELATED_TO"),
                ("vision par ordinateur", "apprentissage profond", "USES"),
                ("traitement du langage", "réseaux de neurones", "USES"),
                ("classification", "apprentissage automatique", "IS_A"),
                ("régression linéaire", "apprentissage automatique", "IS_A"),
                ("TensorFlow", "apprentissage profond", "TOOL_FOR"),
                ("Python", "intelligence artificielle", "USED_IN"),
            ],
            "zh": [
                ("深度学习", "机器学习", "IS_A"),
                ("神经网络", "机器学习", "RELATED_TO"),
                ("计算机视觉", "深度学习", "USES"),
                ("自然语言处理", "神经网络", "USES"),
                ("分类", "机器学习", "IS_A"),
                ("线性回归", "机器学习", "IS_A"),
                ("TensorFlow", "深度学习", "TOOL_FOR"),
                ("Python", "人工智能", "USED_IN"),
            ],
            "en": [
                ("deep learning", "machine learning", "IS_A"),
                ("neural networks", "machine learning", "RELATED_TO"),
                ("computer vision", "deep learning", "USES"),
                ("natural language processing", "neural networks", "USES"),
                ("classification", "machine learning", "IS_A"),
                ("linear regression", "machine learning", "IS_A"),
                ("TensorFlow", "deep learning", "TOOL_FOR"),
                ("Python", "artificial intelligence", "USED_IN"),
            ]
        }
        
        relations_template = templates.get(lang, templates["en"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(num_docs):
                # 随机选择2-4条关系
                import random
                selected = random.sample(relations_template, k=random.randint(2, 4))
                
                relations = [
                    {
                        "head": head,
                        "tail": tail,
                        "type": rel_type
                    }
                    for head, tail, rel_type in selected
                ]
                
                result = {
                    "doc_id": f"doc_{lang}_{i:03d}",
                    "lang": lang,
                    "relations": relations
                }
                
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"Mock关系数据已生成: {output_file} ({num_docs} docs)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量关系提取")
    
    # 输入输出
    parser.add_argument("--input", type=str,
                       help="输入语料文件")
    parser.add_argument("--entities", type=str,
                       help="实体文件(可选)")
    parser.add_argument("--output", type=str,
                       help="输出关系文件")
    parser.add_argument("--lang", type=str, required=True,
                       choices=["fr", "zh", "en"],
                       help="语言")
    
    # 参数
    parser.add_argument("--batch-size", type=int, default=32,
                       help="批大小")
    
    # Mock模式
    parser.add_argument("--mock", action="store_true",
                       help="生成Mock数据")
    parser.add_argument("--num-docs", type=int, default=50,
                       help="Mock文档数")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("批量关系提取")
    logger.info("=" * 60)
    
    # 创建提取器
    extractor = BatchRelationExtractor()
    
    # 确定输出路径
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"data/relations/relations_{args.lang}.jsonl")
    
    if args.mock:
        # Mock模式
        extractor.extract_mock_relations(
            output_file=output_file,
            lang=args.lang,
            num_docs=args.num_docs
        )
    else:
        # 真实提取
        if args.entities:
            # 从实体文件提取
            if not args.input:
                logger.error("需要指定 --input (语料文件) 参数")
                return
            
            extractor.extract_from_entities(
                entities_file=Path(args.entities),
                corpus_file=Path(args.input),
                output_file=output_file,
                lang=args.lang
            )
        else:
            # 从语料直接提取
            if not args.input:
                logger.error("需要指定 --input 参数")
                return
            
            extractor.extract_from_file(
                input_file=Path(args.input),
                output_file=output_file,
                lang=args.lang,
                batch_size=args.batch_size
            )
    
    logger.info(f"\n✅ 关系提取完成: {output_file}")


if __name__ == "__main__":
    main()
