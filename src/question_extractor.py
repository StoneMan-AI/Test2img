#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
题目提取模块
负责从OCR结果中提取题目位置，处理嵌套子题、跨页检测等复杂逻辑
"""

from typing import List, Dict, Tuple, Optional
from src.pattern_matcher import PatternMatcher


class QuestionExtractor:
    """题目提取器"""
    
    def __init__(self, config=None, pattern_matcher: PatternMatcher = None):
        """
        初始化题目提取器
        
        Args:
            config: 配置对象
            pattern_matcher: 模式匹配器（如果提供，将使用它；否则创建新的）
        """
        self.config = config
        self.pattern_matcher = pattern_matcher or PatternMatcher(config)
    
    def extract_question_positions(
        self, 
        ocr_result: List, 
        page_num: int, 
        all_page_ocr_results: Dict = None, 
        total_pages: int = 0
    ) -> List[Dict]:
        """
        从OCR结果中提取所有题目的位置信息
        
        Args:
            ocr_result: OCR识别结果
            page_num: 页码
            all_page_ocr_results: 所有页面的OCR结果字典 {page_num: ocr_result}（用于跨页检测）
            total_pages: 总页数（用于跨页检测）
            
        Returns:
            题目位置列表，包含起始位置、页码等信息
        """
        questions = []
        seen_start_positions = set()  # 记录已见过的题目起始位置，避免重复
        
        # 打印所有识别到的文本（调试用）
        debug_mode = self.config and self.config.mode == 'debug'
        if debug_mode:
            print(f"\n  调试信息 - 第{page_num}页识别到的所有文本:")
            for idx, line_data in enumerate(ocr_result):
                if line_data and len(line_data) > 1:
                    text = line_data[1][0]
                    # 替换特殊字符避免编码错误
                    safe_text = text[:60].replace('\u2212', '-').replace('\uff08', '(').replace('\uff09', ')')
                    print(f"    行{idx}: {safe_text}")
            print()
        
        # 状态跟踪
        current_question = None
        is_chinese_question = False
        nested_sub_numbers = []  # 记录当前题目累积的嵌套子题编号
        primary_type = getattr(self.config, 'question_primary_type', 'chinese') if self.config else 'chinese'
        prefer_chinese_primary = primary_type != 'arabic'
        
        for idx, line_data in enumerate(ocr_result):
            if not line_data:
                continue
            
            # 解析OCR结果
            line_box = line_data[0]
            line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
            
            # 计算行的边界框
            x_coords = [point[0] for point in line_box]
            y_coords = [point[1] for point in line_box]
            line_x_min = min(x_coords)
            line_x_max = max(x_coords)
            line_y_min = min(y_coords)
            line_y_max = max(y_coords)
            
            is_current_chinese = self.pattern_matcher.is_chinese_number_question(line_text)
            is_current_sub = self.pattern_matcher.is_sub_question(line_text)
            is_current_nested = self.pattern_matcher.is_nested_sub_question(line_text)
            
            # 中文大题处理
            if is_current_chinese:
                if prefer_chinese_primary:
                    if current_question:
                        questions.append(current_question)
                    
                    nested_sub_numbers = []
                    
                    y_pos = int(line_y_min)
                    if y_pos in seen_start_positions:
                        print(f"    ⚠ 跳过重复位置: y={y_pos}, 文本={line_text[:30]}")
                        continue
                    
                    seen_start_positions.add(y_pos)
                    
                    text_height = int(line_y_max - line_y_min)
                    
                    current_question = {
                        'type': 'question',
                        'start_y': y_pos,
                        'start_x': 0,
                        'text': line_text,
                        'page': page_num,
                        'is_chinese_question': True,
                        'question_type': 'chinese',
                        'text_height': text_height,
                        'nested_sub_numbers': []
                    }
                    is_chinese_question = True
                    print(f"    [发现] 发现大题 (行{len(questions)+1}): {line_text[:30]} (y={y_pos}, h={text_height})")
                    continue
                else:
                    boundary_y = int(line_y_min)
                    if current_question:
                        question_start_y = current_question.get('start_y', boundary_y)
                        if boundary_y > question_start_y:
                            current_question['early_end_y'] = boundary_y
                            current_question['early_end_text'] = line_text[:30]
                            if debug_mode:
                                print(f"    [边界] 中文大题用作结束坐标: y={boundary_y}, 文本={line_text[:30]}")
                        else:
                            if debug_mode:
                                print(f"    [边界] 中文大题坐标无效 (y={boundary_y} <= start_y={question_start_y})，保持原值")
                        questions.append(current_question)
                        current_question = None
                    nested_sub_numbers = []
                    is_chinese_question = False
                    if debug_mode and not current_question:
                        print(f"    [跳过] 中文大题作为段落标题: {line_text[:30]}")
                    continue
            
            # 阿拉伯数字小题处理
            if is_current_sub:
                if prefer_chinese_primary and is_chinese_question and current_question:
                    if debug_mode:
                        print(f"    [小题] 继续累积到大题: {line_text[:30]}")
                    continue
                
                if current_question:
                    questions.append(current_question)
                
                y_pos = int(line_y_min)
                if y_pos in seen_start_positions:
                    print(f"    ⚠ 跳过重复位置: y={y_pos}, 文本={line_text[:30]}")
                    continue
                
                seen_start_positions.add(y_pos)
                text_height = int(line_y_max - line_y_min)
                nested_sub_numbers = []
                
                current_question = {
                    'type': 'question',
                    'start_y': y_pos,
                    'start_x': 0,
                    'text': line_text,
                    'page': page_num,
                    'is_chinese_question': False,
                    'is_arabic_question': True,
                    'question_type': 'sub',
                    'text_height': text_height,
                    'nested_sub_numbers': []
                }
                is_chinese_question = False
                print(f"    [发现] 发现小题 (行{len(questions)+1}): {line_text[:30]} (y={y_pos}, h={text_height})")
                continue
            
            # 嵌套小题处理
            if is_current_nested:
                if current_question:
                    nested_number = self.pattern_matcher.extract_nested_sub_number(line_text)
                    
                    if nested_number is not None:
                        nested_sub_numbers.append(nested_number)
                        current_question['nested_sub_numbers'] = nested_sub_numbers.copy()
                        
                        early_end_info = self._check_next_nested_sub_question(
                            idx, ocr_result, nested_number, page_num, 
                            all_page_ocr_results, total_pages, debug_mode
                        )
                        
                        if early_end_info.get('cross_page'):
                            current_question['cross_page'] = True
                            current_question['cross_page_to'] = early_end_info.get('cross_page_to')
                        
                        found_next_nested = early_end_info.get('found_next_nested', False)
                        early_end_y = early_end_info.get('early_end_y')
                        early_end_text = early_end_info.get('early_end_text')
                        
                        if not found_next_nested:
                            if current_question.get('cross_page'):
                                if debug_mode:
                                    print(f"    [跨页检测] 题目跨页，将在边界计算时确定结束位置")
                            elif early_end_y is not None:
                                question_start_y = current_question.get('start_y', 0)
                                
                                if early_end_y > question_start_y:
                                    existing_early_end = current_question.get('early_end_y')
                                    if existing_early_end is None:
                                        current_question['early_end_y'] = early_end_y
                                        current_question['early_end_text'] = early_end_text
                                        if debug_mode:
                                            nested_list = current_question.get('nested_sub_numbers', [])
                                            nested_info = f"已累积子题: {nested_list}, " if nested_list else ""
                                            print(f"    [嵌套小题] 3行内未找到下一个子题({nested_number + 1})，提前结束")
                                            print(f"      {nested_info}结束位置: y={early_end_y}, 文本={early_end_text}")
                                    elif early_end_y < existing_early_end and early_end_y > question_start_y:
                                        current_question['early_end_y'] = early_end_y
                                        current_question['early_end_text'] = early_end_text
                                        if debug_mode:
                                            nested_list = current_question.get('nested_sub_numbers', [])
                                            nested_info = f"已累积子题: {nested_list}, " if nested_list else ""
                                            print(f"    [嵌套小题] 更新提前结束位置: y={early_end_y}, 文本={early_end_text}")
                                else:
                                    if debug_mode:
                                        print(f"    [嵌套小题] 提前结束位置无效 (y={early_end_y} <= start_y={question_start_y})，忽略")
                    
                    if debug_mode:
                        question_type = current_question.get('question_type', 'unknown')
                        nested_info = f", 已累积子题: {nested_sub_numbers}" if nested_sub_numbers else ""
                        print(f"    [嵌套小题] 继续累积到题目（类型={question_type}{nested_info}）: {line_text[:30]}")
                    continue
                else:
                    if debug_mode:
                        print(f"    [嵌套小题] 警告：没有当前题目，跳过: {line_text[:30]}")
                    continue
            
            # 检查其他类型的题目模式
            if self.pattern_matcher.matches_pattern(line_text, 'question'):
                # 如果当前题目已经有提前结束位置，检查新题目是否在提前结束位置之前
                if current_question and current_question.get('early_end_y') is not None:
                    y_pos = int(line_y_min)
                    early_end_y = int(current_question['early_end_y'])
                    
                    if y_pos < early_end_y:
                        if debug_mode:
                            print(f"    [调整] 发现新题目在提前结束位置之前，更新结束位置")
                            print(f"      原结束位置: y={early_end_y}")
                            print(f"      新题目位置: y={y_pos}, 文本={line_text[:30]}")
                        current_question['early_end_y'] = y_pos
                        current_question['early_end_text'] = line_text[:30]
                
                # 先保存当前题目
                if current_question:
                    questions.append(current_question)
                
                # 检查是否重复位置
                y_pos = int(line_y_min)
                if y_pos in seen_start_positions:
                    print(f"    ⚠ 跳过重复位置: y={y_pos}, 文本={line_text[:30]}")
                    continue
                
                seen_start_positions.add(y_pos)
                
                # 计算文字高度
                text_height = int(line_y_max - line_y_min)
                
                # 重置嵌套子题编号列表（开始新题目）
                nested_sub_numbers = []
                
                current_question = {
                    'type': 'question',
                    'start_y': y_pos,
                    'start_x': 0,
                    'text': line_text,
                    'page': page_num,
                    'is_chinese_question': False,
                    'question_type': 'other',
                    'text_height': text_height,
                    'nested_sub_numbers': []
                }
                is_chinese_question = False
                print(f"    [发现] 发现题目 (行{len(questions)+1}): {line_text[:30]} (y={y_pos}, h={text_height})")
        
        # 保存最后一题
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _check_next_nested_sub_question(
        self, 
        current_idx: int,
        ocr_result: List,
        nested_number: int,
        page_num: int,
        all_page_ocr_results: Dict = None,
        total_pages: int = 0,
        debug_mode: bool = False
    ) -> Dict:
        """
        检查后续3行（支持跨页）是否按顺序出现下一个嵌套子题
        
        Args:
            current_idx: 当前嵌套子题在ocr_result中的索引
            ocr_result: 当前页的OCR结果
            nested_number: 当前嵌套子题编号
            page_num: 当前页码
            all_page_ocr_results: 所有页面的OCR结果（用于跨页检测）
            total_pages: 总页数
            debug_mode: 是否启用调试模式
            
        Returns:
            检测结果字典: {
                'found_next_nested': bool,
                'early_end_y': int or None,
                'early_end_text': str or None
            }
        """
        next_expected_number = nested_number + 1
        found_next_nested = False
        early_end_y = None
        early_end_text = None
        
        # 检查后续3行（支持跨页检测）
        look_ahead_count = 3
        checked_lines = 0
        current_page_lines_remaining = len(ocr_result) - current_idx - 1
        
        # 先检查当前页的剩余行
        lines_to_check_current_page = min(look_ahead_count, current_page_lines_remaining)
        
        for next_idx in range(current_idx + 1, current_idx + 1 + lines_to_check_current_page):
            if next_idx >= len(ocr_result):
                break
            
            next_line_data = ocr_result[next_idx]
            if not next_line_data or len(next_line_data) < 2:
                continue
            
            next_line_text = next_line_data[1][0] if len(next_line_data[1]) > 0 else ""
            
            # 提取坐标（用于记录early_end_y）
            next_line_box = next_line_data[0]
            next_y_coords = [point[1] for point in next_line_box]
            next_y_min = int(min(next_y_coords))
            
            # 检查是否是嵌套子题
            if self.pattern_matcher.is_nested_sub_question(next_line_text):
                next_nested_num = self.pattern_matcher.extract_nested_sub_number(next_line_text)
                if next_nested_num == next_expected_number:
                    # 找到了按顺序的下一个子题，继续累积
                    found_next_nested = True
                    if debug_mode:
                        print(f"    [嵌套小题] 找到下一个子题 ({next_expected_number})，继续累积")
                    break
                else:
                    # 是嵌套子题但编号不匹配
                    if early_end_y is None:
                        early_end_y = next_y_min
                        early_end_text = next_line_text[:30]
                        if debug_mode:
                            print(f"    [嵌套小题] 遇到非预期的嵌套子题 ({next_nested_num}，期望{next_expected_number})，记录为可能的结束位置")
            else:
                # 不是嵌套子题，记录为可能的结束位置（只记录第一个）
                if early_end_y is None:
                    early_end_y = next_y_min
                    early_end_text = next_line_text[:30]
                    if debug_mode:
                        print(f"    [嵌套小题] 遇到非嵌套子题文本，记录为可能的结束位置: {early_end_text}")
            
            checked_lines += 1
        
        # 如果当前页已检查完但还没检查够3行，且还有下一页，继续检查下一页
        if not found_next_nested and checked_lines < look_ahead_count:
            remaining_lines_to_check = look_ahead_count - checked_lines
            next_page_num = page_num + 1
            
            # 检查是否有下一页且可以访问OCR结果
            if (next_page_num <= total_pages and 
                all_page_ocr_results and 
                next_page_num in all_page_ocr_results):
                next_page_ocr = all_page_ocr_results[next_page_num]
                
                if next_page_ocr and len(next_page_ocr) > 0:
                    if debug_mode:
                        print(f"    [跨页检测] 当前页已结束，检查下一页（第{next_page_num}页）的前{remaining_lines_to_check}行")
                    
                    # 检查下一页的前几行
                    for next_page_idx in range(min(remaining_lines_to_check, len(next_page_ocr))):
                        next_line_data = next_page_ocr[next_page_idx]
                        if not next_line_data or len(next_line_data) < 2:
                            continue
                        
                        next_line_text = next_line_data[1][0] if len(next_line_data[1]) > 0 else ""
                        
                        # 提取坐标（用于跨页合并）
                        next_line_box = next_line_data[0]
                        next_y_coords = [point[1] for point in next_line_box]
                        next_y_min = int(min(next_y_coords))
                        
                        # 检查是否是嵌套子题
                        if self.pattern_matcher.is_nested_sub_question(next_line_text):
                            next_nested_num = self.pattern_matcher.extract_nested_sub_number(next_line_text)
                            if next_nested_num == next_expected_number:
                                # 找到了按顺序的下一个子题（在下一页）
                                found_next_nested = True
                                if debug_mode:
                                    print(f"    [跨页检测] 在下一页找到下一个子题 ({next_expected_number})，标记跨页合并")
                                # 返回跨页信息，调用处会设置跨页标记
                                return {
                                    'found_next_nested': True,
                                    'cross_page': True,
                                    'cross_page_to': next_page_num,
                                    'early_end_y': None,
                                    'early_end_text': None
                                }
                        else:
                            # 不是嵌套子题
                            if early_end_y is None:
                                if debug_mode:
                                    print(f"    [跨页检测] 在下一页遇到非嵌套子题文本: {next_line_text[:30]}")
                    else:
                        # 循环正常结束
                        if debug_mode:
                            print(f"    [跨页检测] 下一页前{remaining_lines_to_check}行未找到下一个子题")
        
        return {
            'found_next_nested': found_next_nested,
            'early_end_y': early_end_y,
            'early_end_text': early_end_text
        }

