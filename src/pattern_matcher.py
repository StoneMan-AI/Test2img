#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模式匹配模块
负责题目、答案等文本模式的识别和匹配
"""

import re
from typing import Dict, Any, Optional


class PatternMatcher:
    """模式匹配器"""
    
    def __init__(self, config=None):
        """
        初始化模式匹配器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, Any]:
        """编译正则表达式模式"""
        if self.config:
            return self.config.compile_patterns()
        else:
            return {
                'question': [],
                'answer': [],
                'chinese_number': None,
                'sub_question': [],
                'nested_sub_question': []
            }
    
    def matches_pattern(self, text: str, pattern_type: str) -> bool:
        """
        检查文本是否匹配指定模式
        
        Args:
            text: 待检查文本
            pattern_type: 模式类型（'question' 或 'answer'）
            
        Returns:
            是否匹配
        """
        if not self.config or pattern_type not in self.patterns:
            return False
        
        patterns = self.patterns[pattern_type]
        text_stripped = text.strip()
        
        for pattern in patterns:
            if pattern.match(text_stripped):
                return True
        
        return False
    
    def is_sub_question(self, text: str) -> bool:
        """
        检查文本是否是阿拉伯数字小题
        
        Args:
            text: 待检查文本
            
        Returns:
            是否是小题
        """
        if not self.config or 'sub_question' not in self.patterns:
            return False
        
        patterns = self.patterns['sub_question']
        text_stripped = text.strip()
        
        for pattern in patterns:
            if pattern.match(text_stripped):
                return True
        
        return False
    
    def is_nested_sub_question(self, text: str) -> bool:
        """
        检查文本是否是嵌套小题（(1)、（2）、[3]等）
        
        Args:
            text: 待检查文本
            
        Returns:
            是否是嵌套小题
        """
        if not self.config or 'nested_sub_question' not in self.patterns:
            return False
        
        patterns = self.patterns['nested_sub_question']
        text_stripped = text.strip()
        
        for pattern in patterns:
            if pattern.match(text_stripped):
                return True
        
        return False
    
    def extract_nested_sub_number(self, text: str) -> Optional[int]:
        """
        从嵌套子题文本中提取数字编号
        
        Args:
            text: 嵌套子题文本，如 "(1)", "（2）", "[3]"
            
        Returns:
            数字编号，如果无法提取则返回None
        """
        if not text:
            return None
        
        # 匹配括号或方括号中的数字
        match = re.search(r'[（(）)\[\]]([0-9]+)[）)\]）]', text.strip())
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return None
        return None
    
    def is_chinese_number_question(self, text: str) -> bool:
        """
        检查文本是否是中文数字大题
        
        Args:
            text: 待检查文本
            
        Returns:
            是否是中文数字大题
        """
        if not self.config or 'chinese_number' not in self.patterns:
            return False
        
        pattern = self.patterns['chinese_number']
        if pattern:
            return pattern.match(text.strip()) is not None
        return False

