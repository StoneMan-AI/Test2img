#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
区域提取模块
负责从OCR结果中提取题目和答案区域，合并相邻区域
"""

from typing import List, Dict, Tuple
from src.pattern_matcher import PatternMatcher


class ZoneExtractor:
    """区域提取器"""
    
    def __init__(self, config=None, pattern_matcher: PatternMatcher = None):
        """
        初始化区域提取器
        
        Args:
            config: 配置对象
            pattern_matcher: 模式匹配器（如果提供，将使用它；否则创建新的）
        """
        self.config = config
        self.pattern_matcher = pattern_matcher or PatternMatcher(config)
    
    def extract_zones(self, ocr_result: List, page_size: Tuple[int, int], page_num: int) -> List[Dict]:
        """
        从OCR结果中提取题目和答案区域
        
        Args:
            ocr_result: OCR识别结果
            page_size: 页面尺寸 (width, height)
            page_num: 页码
            
        Returns:
            区域列表
        """
        page_width, page_height = page_size
        zones = []
        
        # 打印所有识别到的文本（调试用）
        debug_mode = self.config and self.config.mode == 'debug'
        if debug_mode:
            print(f"\n  调试信息 - 第{page_num}页识别到的所有文本:")
            for idx, line_data in enumerate(ocr_result):
                if line_data and len(line_data) > 1:
                    text = line_data[1][0]
                    print(f"    行{idx}: {text[:60]}")
            print()
        
        # 状态跟踪
        current_question = None
        current_answer = None
        is_chinese_question = False  # 标记是否在中文大题下
        
        for line_data in ocr_result:
            if not line_data:
                continue
            
            # 解析OCR结果
            line_box = line_data[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
            confidence = line_data[1][1] if len(line_data[1]) > 1 else 0
            
            # 计算行的边界框（取四个点的最小/最大值）
            x_coords = [point[0] for point in line_box]
            y_coords = [point[1] for point in line_box]
            
            line_x_min = min(x_coords)
            line_x_max = max(x_coords)
            line_y_min = min(y_coords)
            line_y_max = max(y_coords)
            line_y_center = (line_y_min + line_y_max) / 2
            
            # 检查是否是新题目的开始
            if self.pattern_matcher.matches_pattern(line_text, 'question'):
                # 检查是否是中文数字大题
                is_current_chinese = self.pattern_matcher.is_chinese_number_question(line_text)
                
                # 如果是中文大题，开始新的大题
                if is_current_chinese:
                    # 结束上一个区域
                    if current_question:
                        zones.append(current_question)
                    if current_answer:
                        zones.append(current_answer)
                        current_answer = None
                    
                    # 开始新的中文大题
                    current_question = {
                        'type': 'question',
                        'x': 0,
                        'y': int(line_y_min),
                        'width': page_width,
                        'height': 0,
                        'text': line_text,
                        'start_line': line_text,
                        'is_chinese_question': True
                    }
                    is_chinese_question = True
                    print(f"    [发现] 发现大题 (行{len(zones)+1}): {line_text[:50]}")
                
                # 如果是阿拉伯数字小题
                elif self.pattern_matcher.is_sub_question(line_text):
                    # 如果当前在中文大题下，不创建新的独立题目，继续累积
                    if is_chinese_question and current_question:
                        # 继续累积到当前大题中，更新高度
                        current_question['height'] = int(line_y_min - current_question['y'])
                        if debug_mode:
                            print(f"    [小题] 继续累积到大题: {line_text[:30]}")
                    else:
                        # 不在大题下，创建新的独立小题
                        if current_question:
                            zones.append(current_question)
                        if current_answer:
                            zones.append(current_answer)
                            current_answer = None
                        
                        current_question = {
                            'type': 'question',
                            'x': 0,
                            'y': int(line_y_min),
                            'width': page_width,
                            'height': 0,
                            'text': line_text,
                            'start_line': line_text,
                            'is_chinese_question': False
                        }
                        is_chinese_question = False
                        print(f"    [发现] 发现小题 (行{len(zones)+1}): {line_text[:50]}")
                
                # 其他类型的题目
                else:
                    # 结束上一个区域
                    if current_question:
                        zones.append(current_question)
                    if current_answer:
                        zones.append(current_answer)
                        current_answer = None
                    
                    current_question = {
                        'type': 'question',
                        'x': 0,
                        'y': int(line_y_min),
                        'width': page_width,
                        'height': 0,
                        'text': line_text,
                        'start_line': line_text,
                        'is_chinese_question': False
                    }
                    is_chinese_question = False
                    print(f"    [发现] 发现题目 (行{len(zones)+1}): {line_text[:50]}")
            
            # 如果是题目区域，继续累积内容
            elif current_question and line_text.strip():
                # 检查是否是下一个题目的开始（防止误判）
                is_current_chinese = self.pattern_matcher.is_chinese_number_question(line_text)
                is_current_sub = self.pattern_matcher.is_sub_question(line_text)
                
                # 如果遇到新的中文大题，结束当前题目
                if is_current_chinese:
                    current_question['height'] = int(line_y_min - current_question['y'])
                    zones.append(current_question)
                    
                    current_question = {
                        'type': 'question',
                        'x': 0,
                        'y': int(line_y_min),
                        'width': page_width,
                        'height': 0,
                        'text': line_text,
                        'start_line': line_text,
                        'is_chinese_question': True
                    }
                    is_chinese_question = True
                    print(f"    [发现] 发现大题 (行{len(zones)+1}): {line_text[:50]}")
                
                # 如果遇到小题且不在大题下，结束当前题目
                elif is_current_sub and not is_chinese_question:
                    current_question['height'] = int(line_y_min - current_question['y'])
                    zones.append(current_question)
                    
                    current_question = {
                        'type': 'question',
                        'x': 0,
                        'y': int(line_y_min),
                        'width': page_width,
                        'height': 0,
                        'text': line_text,
                        'start_line': line_text,
                        'is_chinese_question': False
                    }
                    is_chinese_question = False
                    print(f"    ✓ 发现小题 (行{len(zones)+1}): {line_text[:50]}")
                
                # 其他类型题目格式
                elif self.pattern_matcher.matches_pattern(line_text, 'question'):
                    current_question['height'] = int(line_y_min - current_question['y'])
                    zones.append(current_question)
                    
                    current_question = {
                        'type': 'question',
                        'x': 0,
                        'y': int(line_y_min),
                        'width': page_width,
                        'height': 0,
                        'text': line_text,
                        'start_line': line_text
                    }
                    is_chinese_question = False
                    print(f"    [发现] 发现题目 (行{len(zones)+1}): {line_text[:50]}")
                
                # 如果是普通内容行（不在题目匹配中），更新当前题目高度
                else:
                    # 更新当前题目的高度（累积内容）
                    if current_question:
                        current_question['height'] = int(line_y_min - current_question['y'])
            
            # 检查是否是答案区域
            elif self.pattern_matcher.matches_pattern(line_text, 'answer'):
                # 结束上一个题目区域
                if current_question:
                    # 更新题目的高度
                    current_question['height'] = int(line_y_min - current_question['y'])
                    zones.append(current_question)
                    current_question = None
                
                # 结束上一个答案区域（如果存在）
                if current_answer:
                    zones.append(current_answer)
                
                # 开始新答案区域
                current_answer = {
                    'type': 'answer',
                    'x': 0,
                    'y': int(line_y_min),
                    'width': page_width,
                    'height': 0,
                    'text': line_text,
                    'start_line': line_text
                }
                print(f"    [发现] 发现答案区域: {line_text[:30]}")
        
        # 处理最后一页的剩余区域
        if current_question:
            current_question['height'] = page_height - current_question['y']
            zones.append(current_question)
        
        if current_answer:
            current_answer['height'] = page_height - current_answer['y']
            zones.append(current_answer)
        
        # 合并相邻的同类型区域（可选优化）
        zones = self.merge_adjacent_zones(zones)
        
        return zones
    
    def merge_adjacent_zones(self, zones: List[Dict]) -> List[Dict]:
        """
        合并相邻的同类型区域（简化输出）
        
        Args:
            zones: 区域列表
            
        Returns:
            合并后的区域列表
        """
        if not zones:
            return []
        
        merged = []
        i = 0
        
        while i < len(zones):
            current = zones[i].copy()
            
            # 检查是否可以与后续区域合并
            j = i + 1
            while j < len(zones):
                next_zone = zones[j]
                
                # 只合并相邻的同类区域
                if (next_zone['type'] == current['type'] and
                    next_zone['y'] <= current['y'] + current['height'] + 50):  # 50像素阈值
                    current['height'] = (next_zone['y'] + next_zone['height']) - current['y']
                    j += 1
                else:
                    break
            
            merged.append(current)
            i = j
        
        return merged

