#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
关键词匹配模块
负责从段落中提取关键词并在OCR结果中进行模糊匹配
"""

from typing import List, Dict
import re


class KeywordMatcher:
    """关键词匹配器"""
    
    def __init__(self, config=None):
        """
        初始化关键词匹配器
        
        Args:
            config: 配置对象
        """
        self.config = config
        # 常见的关键词列表（可根据需要扩展）
        self.keywords = [
            "答案", "参考答案", "标准答案", "解答", "解析",
            "Key", "Answer", "Solution", "Explanation"
        ]
    
    def extract_keyword_following_texts(self, paragraphs: List[str]) -> List[str]:
        """
        从文本段落中提取关键词后面的1-5个中文字符
        
        Args:
            paragraphs: DOCX提取的段落文本列表
        
        Returns:
            关键词后面的文本列表
        """
        keyword_texts = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 检查段落中是否包含关键词
            for keyword in self.keywords:
                if keyword in para:
                    # 找到关键词后的位置
                    keyword_pos = para.find(keyword)
                    if keyword_pos != -1:
                        # 提取关键词后面的文本
                        following_text = para[keyword_pos + len(keyword):]
                        
                        # 提取1-5个中文字符
                        # 匹配中文字符（包括标点符号）
                        chinese_pattern = r'[\u4e00-\u9fff，。、；：！？]{1,5}'
                        match = re.search(chinese_pattern, following_text)
                        if match:
                            keyword_texts.append(match.group())
                        break  # 找到一个关键词后就不再查找其他关键词
        
        return keyword_texts
    
    def fuzzy_match_keywords_in_ocr(self, keyword_texts: List[str], ocr_result: List) -> List[Dict]:
        """
        在OCR结果中模糊匹配关键词文本
        
        Args:
            keyword_texts: 关键词文本列表
            ocr_result: OCR识别结果
        
        Returns:
            匹配到的OCR结果索引和位置信息
        """
        matches = []
        
        if not keyword_texts or not ocr_result:
            return matches
        
        for idx, line_data in enumerate(ocr_result):
            if not line_data or len(line_data) < 2:
                continue
            
            line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
            
            # 检查是否包含关键词文本
            for keyword_text in keyword_texts:
                if keyword_text in line_text:
                    # 提取坐标信息
                    line_box = line_data[0]
                    x_coords = [point[0] for point in line_box]
                    y_coords = [point[1] for point in line_box]
                    
                    matches.append({
                        'index': idx,
                        'text': line_text,
                        'x_min': min(x_coords),
                        'x_max': max(x_coords),
                        'y_min': min(y_coords),
                        'y_max': max(y_coords),
                        'keyword': keyword_text
                    })
                    break  # 找到一个匹配后就不再查找其他关键词
        
        return matches
    
    def is_keyword_text(self, text: str) -> bool:
        """
        检查文本是否是关键词文本
        
        Args:
            text: 待检查文本
        
        Returns:
            是否是关键词文本
        """
        text_stripped = text.strip()
        
        for keyword in self.keywords:
            if keyword in text_stripped:
                return True
        
        return False

