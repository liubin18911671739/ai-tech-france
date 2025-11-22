"""
脚本01: 语料清洗

功能:
- 读取原始语料
- 文本清洗与标准化
- 生成JSONL格式的清洗后语料
- 支持多语言(fr/zh/en)
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger import get_logger
from config import config

logger = get_logger(__name__)


class CorpusCleaner:
    """语料清洗器"""
    
    def __init__(self):
        """初始化清洗器"""
        self.min_length = 50  # 最小文本长度
        self.max_length = 10000  # 最大文本长度
    
    def clean_text(self, text: str, lang: str = "fr") -> str:
        """
        清洗单个文本
        
        Args:
            text: 原始文本
            lang: 语言代码
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符(保留标点)
        if lang == "zh":
            # 中文: 保留中文字符、标点
            text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s,.!?;:()""''《》【】]', '', text)
        else:
            # 其他语言: 保留字母、数字、基本标点
            text = re.sub(r'[^a-zA-ZÀ-ÿ0-9\s,.!?;:()\'"«»-]', '', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def is_valid_document(self, doc: Dict) -> bool:
        """
        验证文档是否有效
        
        Args:
            doc: 文档字典
            
        Returns:
            是否有效
        """
        # 检查必需字段
        if "content" not in doc:
            return False
        
        content = doc["content"]
        
        # 检查长度
        if len(content) < self.min_length or len(content) > self.max_length:
            return False
        
        return True
    
    def clean_corpus(
        self,
        input_path: Path,
        output_path: Path,
        lang: str = "fr"
    ):
        """
        清洗整个语料库
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            lang: 语言代码
        """
        logger.info(f"开始清洗语料: {input_path} -> {output_path}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        valid_count = 0
        invalid_count = 0
        
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                try:
                    # 解析JSON
                    doc = json.loads(line.strip())
                    
                    # 清洗内容
                    if "content" in doc:
                        doc["content"] = self.clean_text(doc["content"], lang)
                    
                    if "title" in doc:
                        doc["title"] = self.clean_text(doc["title"], lang)
                    
                    # 添加语言标签
                    doc["lang"] = lang
                    
                    # 添加doc_id(如果没有)
                    if "doc_id" not in doc:
                        doc["doc_id"] = f"doc_{lang}_{line_num:06d}"
                    
                    # 验证
                    if self.is_valid_document(doc):
                        fout.write(json.dumps(doc, ensure_ascii=False) + '\n')
                        valid_count += 1
                    else:
                        invalid_count += 1
                    
                    if line_num % 1000 == 0:
                        logger.info(f"已处理 {line_num} 行: 有效={valid_count}, 无效={invalid_count}")
                
                except Exception as e:
                    logger.error(f"处理第{line_num}行失败: {e}")
                    invalid_count += 1
                    continue
        
        logger.info(f"清洗完成! 有效文档: {valid_count}, 无效文档: {invalid_count}")


def create_mock_corpus(output_dir: Path, lang: str, num_docs: int = 100):
    """
    创建Mock语料(用于测试)
    
    Args:
        output_dir: 输出目录
        lang: 语言
        num_docs: 文档数量
    """
    logger.info(f"创建Mock语料: lang={lang}, num_docs={num_docs}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"corpus_{lang}.jsonl"
    
    # Mock内容模板
    templates = {
        "fr": [
            "La grammaire française est un ensemble de règles qui régissent la langue française. "
            "Elle comprend la syntaxe, la morphologie, et la phonétique. "
            "Pour maîtriser le français, il est essentiel de bien comprendre ces règles.",
            
            "Le verbe est un élément central de la phrase française. "
            "La conjugaison des verbes varie selon le temps, le mode, et la personne. "
            "Les verbes du premier groupe se terminent en -er.",
            
            "La prononciation française nécessite une attention particulière aux sons nasaux. "
            "Les voyelles françaises sont plus nombreuses que dans d'autres langues. "
            "L'accent tonique joue un rôle important."
        ],
        "zh": [
            "法语语法是规范法语语言使用的一套规则体系。"
            "它包括句法、词法和语音等方面。"
            "要掌握法语,必须深入理解这些语法规则。",
            
            "动词是法语句子的核心成分。"
            "法语动词的变位根据时态、语气和人称而变化。"
            "第一组动词以-er结尾。",
            
            "法语发音需要特别注意鼻音。"
            "法语元音比其他语言要多。"
            "重音在法语中扮演重要角色。"
        ],
        "en": [
            "French grammar is a set of rules governing the French language. "
            "It includes syntax, morphology, and phonetics. "
            "To master French, it is essential to understand these rules well.",
            
            "The verb is a central element of the French sentence. "
            "Verb conjugation varies according to tense, mood, and person. "
            "First group verbs end in -er.",
            
            "French pronunciation requires particular attention to nasal sounds. "
            "French vowels are more numerous than in other languages. "
            "Stress plays an important role."
        ]
    }
    
    topics = {
        "fr": ["grammaire", "verbe", "prononciation", "vocabulaire", "syntaxe"],
        "zh": ["语法", "动词", "发音", "词汇", "句法"],
        "en": ["grammar", "verb", "pronunciation", "vocabulary", "syntax"]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_docs):
            doc = {
                "doc_id": f"doc_{lang}_{i:06d}",
                "title": f"{topics[lang][i % len(topics[lang])]} - Lesson {i+1}",
                "content": templates[lang][i % len(templates[lang])],
                "lang": lang,
                "concepts": [topics[lang][i % len(topics[lang])]],
                "difficulty": ["beginner", "intermediate", "advanced"][i % 3]
            }
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    logger.info(f"Mock语料已创建: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="语料清洗脚本")
    parser.add_argument("--input", type=str, help="输入文件/目录路径")
    parser.add_argument("--output", type=str, help="输出目录路径")
    parser.add_argument("--lang", type=str, default="fr", 
                       choices=["fr", "zh", "en"], help="语言代码")
    parser.add_argument("--create-mock", action="store_true", 
                       help="创建Mock数据")
    parser.add_argument("--mock-size", type=int, default=100,
                       help="Mock数据大小")
    
    args = parser.parse_args()
    
    # 创建Mock数据
    if args.create_mock:
        output_dir = Path(args.output) if args.output else config.RAW_DIR
        create_mock_corpus(output_dir, args.lang, args.mock_size)
        
        # 创建所有语言的Mock数据
        for lang in ["fr", "zh", "en"]:
            create_mock_corpus(output_dir, lang, args.mock_size)
        
        logger.info("所有Mock数据创建完成!")
        return
    
    # 清洗语料
    if not args.input:
        logger.error("请指定 --input 参数或使用 --create-mock")
        return
    
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else config.CLEANED_DIR
    output_path = output_dir / f"corpus_{args.lang}_cleaned.jsonl"
    
    cleaner = CorpusCleaner()
    cleaner.clean_corpus(input_path, output_path, args.lang)


if __name__ == "__main__":
    main()
