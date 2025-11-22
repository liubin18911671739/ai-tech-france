"""
中文 NER - 使用 HanLP

使用 HanLP 进行中文命名实体识别
识别学习概念相关实体
"""
import json
from pathlib import Path
from typing import List, Dict
import hanlp
from logger import get_logger
from config import config

logger = get_logger(__name__)


class ChineseNER:
    """中文命名实体识别"""
    
    def __init__(self):
        """初始化HanLP NER"""
        logger.info("加载HanLP中文NER模型")
        try:
            # 使用HanLP的多任务模型
            self.hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
            logger.info("HanLP模型加载完成")
        except Exception as e:
            logger.warning(f"HanLP加载失败,使用备用NER: {e}")
            # 备用: 只加载NER
            self.hanlp = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表 [{"entity": "语法", "type": "Concept", "score": 1.0, "start": 5, "end": 7}]
        """
        if not text or not text.strip():
            return []
        
        try:
            # HanLP处理
            result = self.hanlp(text, tasks='ner')
            
            # 提取NER结果
            if isinstance(result, dict) and 'ner' in result:
                ner_results = result['ner']
            elif isinstance(result, list):
                # 备用NER直接返回列表
                ner_results = result
            else:
                return []
            
            entities = []
            for item in ner_results:
                if isinstance(item, tuple) and len(item) >= 2:
                    entity_text, entity_type = item[0], item[1]
                    
                    # 查找实体在原文中的位置
                    start = text.find(entity_text)
                    end = start + len(entity_text) if start != -1 else -1
                    
                    entities.append({
                        "entity": entity_text,
                        "type": self._map_entity_type(entity_type),
                        "score": 1.0,  # HanLP不返回置信度
                        "start": start,
                        "end": end
                    })
            
            return entities
        
        except Exception as e:
            logger.error(f"中文NER处理失败: {e}")
            return []
    
    def _map_entity_type(self, ner_type: str) -> str:
        """
        映射NER类型到本体类型
        
        HanLP类型: PERSON, LOCATION, ORGANIZATION 等
        """
        mapping = {
            "PERSON": "Person",
            "LOCATION": "Location",
            "ORGANIZATION": "Organization",
            "GPE": "Location",  # Geo-Political Entity
            "TIME": "Time",
            "DATE": "Time"
        }
        return mapping.get(ner_type.upper(), "Concept")
    
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
        logger.info(f"开始处理中文语料: {corpus_path}")
        
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
                        "lang": "zh",
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
    
    parser = argparse.ArgumentParser(description="中文NER提取")
    parser.add_argument("--input", type=str, required=True, help="输入语料路径")
    parser.add_argument("--output", type=str, help="输出路径")
    parser.add_argument("--text", type=str, help="测试单个文本")
    
    args = parser.parse_args()
    
    # 初始化NER
    ner = ChineseNER()
    
    # 单文本测试
    if args.text:
        logger.info(f"测试文本: {args.text}")
        entities = ner.extract_entities(args.text)
        logger.info(f"提取实体: {entities}")
        return
    
    # 批量处理
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "entities_zh.jsonl"
    
    ner.extract_from_corpus(input_path, output_path)


if __name__ == "__main__":
    main()
