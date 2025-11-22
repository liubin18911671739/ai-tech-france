"""
法语 NER - 使用 CamemBERT

基于 Jean-Baptiste/camembert-ner 模型进行法语命名实体识别
识别学习概念相关实体
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from logger import get_logger
from config import config

logger = get_logger(__name__)


class FrenchNER:
    """法语命名实体识别"""
    
    def __init__(self, model_name: str = None):
        """
        初始化法语NER模型
        
        Args:
            model_name: 模型名称,默认使用config中的配置
        """
        self.model_name = model_name or config.FR_NER_MODEL
        self.device = 0 if torch.cuda.is_available() else -1
        
        logger.info(f"加载法语NER模型: {self.model_name}")
        self.ner_pipeline = pipeline(
            "ner",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",
            device=self.device
        )
        logger.info("法语NER模型加载完成")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表 [{"entity": "grammaire", "type": "MISC", "score": 0.95, "start": 10, "end": 19}]
        """
        if not text or not text.strip():
            return []
        
        try:
            results = self.ner_pipeline(text)
            
            entities = []
            for entity in results:
                entities.append({
                    "entity": entity["word"],
                    "type": self._map_entity_type(entity["entity_group"]),
                    "score": float(entity["score"]),
                    "start": int(entity["start"]),
                    "end": int(entity["end"])
                })
            
            return entities
        
        except Exception as e:
            logger.error(f"NER处理失败: {e}")
            return []
    
    def _map_entity_type(self, ner_type: str) -> str:
        """
        映射NER类型到本体类型
        
        标准NER类型: PER(人名), LOC(地点), ORG(组织), MISC(其他)
        映射到学习概念类型
        """
        mapping = {
            "PER": "Person",
            "LOC": "Location",
            "ORG": "Organization",
            "MISC": "Concept"  # 将MISC映射为学习概念
        }
        return mapping.get(ner_type, "Concept")
    
    def extract_from_corpus(
        self,
        corpus_path: Path,
        output_path: Path = None
    ) -> List[Dict]:
        """
        从语料库文件批量提取实体
        
        Args:
            corpus_path: 语料文件路径 (JSONL格式)
            output_path: 输出路径
            
        Returns:
            提取结果列表
        """
        logger.info(f"开始处理语料: {corpus_path}")
        
        results = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line.strip())
                    doc_id = doc.get("doc_id", f"doc_{line_num}")
                    text = doc.get("content", "")
                    
                    if not text:
                        continue
                    
                    # 提取实体
                    entities = self.extract_entities(text)
                    
                    results.append({
                        "doc_id": doc_id,
                        "lang": "fr",
                        "entities": entities
                    })
                    
                    if line_num % 100 == 0:
                        logger.info(f"已处理 {line_num} 篇文档")
                
                except Exception as e:
                    logger.error(f"处理第{line_num}行失败: {e}")
                    continue
        
        logger.info(f"处理完成,共提取 {len(results)} 篇文档的实体")
        
        # 保存结果
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            logger.info(f"结果已保存至: {output_path}")
        
        return results


def main():
    """测试函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="法语NER提取")
    parser.add_argument("--input", type=str, required=True, help="输入语料路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--text", type=str, help="测试单个文本")
    
    args = parser.parse_args()
    
    # 初始化NER
    ner = FrenchNER()
    
    # 单文本测试
    if args.text:
        logger.info(f"测试文本: {args.text}")
        entities = ner.extract_entities(args.text)
        logger.info(f"提取实体: {entities}")
        return
    
    # 批量处理
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "entities_fr.jsonl"
    
    ner.extract_from_corpus(input_path, output_path)


if __name__ == "__main__":
    main()
