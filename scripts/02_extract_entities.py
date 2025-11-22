#!/usr/bin/env python3
"""
批量实体提取

从清洗后的语料中提取实体
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from kg.extraction.fr_ner import FrenchNER
from kg.extraction.zh_ner import ChineseNER

logger = get_logger(__name__)


class EntityExtractor:
    """实体提取器"""
    
    def __init__(self):
        """初始化提取器"""
        self.extractors = {}
        logger.info("实体提取器初始化")
    
    def _get_extractor(self, lang: str):
        """
        获取语言对应的NER模型
        
        Args:
            lang: 语言代码
            
        Returns:
            NER提取器
        """
        if lang in self.extractors:
            return self.extractors[lang]
        
        # 初始化提取器
        if lang == "fr":
            extractor = FrenchNER()
        elif lang == "zh":
            extractor = ChineseNER()
        else:
            logger.warning(f"不支持的语言: {lang}, 使用通用提取器")
            return None
        
        self.extractors[lang] = extractor
        logger.info(f"加载 {lang} NER模型")
        
        return extractor
    
    def extract_from_file(
        self,
        input_file: Path,
        output_file: Path,
        lang: str,
        batch_size: int = 32
    ):
        """
        从文件提取实体
        
        Args:
            input_file: 输入语料文件(JSONL)
            output_file: 输出实体文件(JSONL)
            lang: 语言
            batch_size: 批大小
        """
        logger.info(f"从 {input_file} 提取实体")
        
        if not input_file.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return
        
        # 获取提取器
        extractor = self._get_extractor(lang)
        if not extractor:
            logger.error(f"无法为语言 {lang} 创建提取器")
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
        total_entities = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                for doc in batch:
                    doc_id = doc.get("doc_id", "")
                    text = doc.get("text", "")
                    
                    if not text:
                        continue
                    
                    # 提取实体
                    entities = extractor.extract_entities(text)
                    
                    # 写入结果
                    result = {
                        "doc_id": doc_id,
                        "lang": lang,
                        "entities": entities
                    }
                    
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
                    total_entities += len(entities)
                
                if (i + batch_size) % 100 == 0:
                    logger.info(f"已处理 {i + batch_size} / {len(documents)} 个文档")
        
        logger.info(f"提取完成: {total_entities} 个实体 -> {output_file}")
    
    def extract_mock_entities(
        self,
        output_file: Path,
        lang: str,
        num_docs: int = 50
    ):
        """
        生成Mock实体数据
        
        Args:
            output_file: 输出文件
            lang: 语言
            num_docs: 文档数
        """
        logger.info(f"生成Mock实体数据: {lang}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Mock实体模板
        templates = {
            "fr": [
                ("apprentissage automatique", "CONCEPT"),
                ("apprentissage profond", "CONCEPT"),
                ("réseaux de neurones", "CONCEPT"),
                ("intelligence artificielle", "CONCEPT"),
                ("traitement du langage", "CONCEPT"),
                ("vision par ordinateur", "CONCEPT"),
                ("régression linéaire", "ALGORITHM"),
                ("classification", "TASK"),
                ("Python", "TECHNOLOGY"),
                ("TensorFlow", "TECHNOLOGY"),
            ],
            "zh": [
                ("机器学习", "CONCEPT"),
                ("深度学习", "CONCEPT"),
                ("神经网络", "CONCEPT"),
                ("人工智能", "CONCEPT"),
                ("自然语言处理", "CONCEPT"),
                ("计算机视觉", "CONCEPT"),
                ("线性回归", "ALGORITHM"),
                ("分类", "TASK"),
                ("Python", "TECHNOLOGY"),
                ("TensorFlow", "TECHNOLOGY"),
            ],
            "en": [
                ("machine learning", "CONCEPT"),
                ("deep learning", "CONCEPT"),
                ("neural networks", "CONCEPT"),
                ("artificial intelligence", "CONCEPT"),
                ("natural language processing", "CONCEPT"),
                ("computer vision", "CONCEPT"),
                ("linear regression", "ALGORITHM"),
                ("classification", "TASK"),
                ("Python", "TECHNOLOGY"),
                ("TensorFlow", "TECHNOLOGY"),
            ]
        }
        
        entities_template = templates.get(lang, templates["en"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(num_docs):
                # 随机选择3-5个实体
                import random
                selected = random.sample(entities_template, k=random.randint(3, 5))
                
                entities = [
                    {
                        "text": text,
                        "type": entity_type,
                        "start": 0,
                        "end": len(text)
                    }
                    for text, entity_type in selected
                ]
                
                result = {
                    "doc_id": f"doc_{lang}_{i:03d}",
                    "lang": lang,
                    "entities": entities
                }
                
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"Mock实体数据已生成: {output_file} ({num_docs} docs)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量实体提取")
    
    # 输入输出
    parser.add_argument("--input", type=str,
                       help="输入语料文件")
    parser.add_argument("--output", type=str,
                       help="输出实体文件")
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
    logger.info("批量实体提取")
    logger.info("=" * 60)
    
    # 创建提取器
    extractor = EntityExtractor()
    
    # 确定输出路径
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = Path(f"data/entities/entities_{args.lang}.jsonl")
    
    if args.mock:
        # Mock模式
        extractor.extract_mock_entities(
            output_file=output_file,
            lang=args.lang,
            num_docs=args.num_docs
        )
    else:
        # 真实提取
        if not args.input:
            logger.error("需要指定 --input 参数")
            return
        
        input_file = Path(args.input)
        
        extractor.extract_from_file(
            input_file=input_file,
            output_file=output_file,
            lang=args.lang,
            batch_size=args.batch_size
        )
    
    logger.info(f"\n✅ 实体提取完成: {output_file}")


if __name__ == "__main__":
    main()
