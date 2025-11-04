#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
布局分析模块（重构版）
核心模块：识别页面布局，定位题目和答案的位置
已重构为使用多个子模块，提高代码可维护性
"""

from typing import List, Tuple, Dict
from PIL import Image
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: OpenCV未安装，图像预处理功能将不可用")

# 导入图像预处理模块
try:
    from src.image_preprocessor import preprocess_image, correct_coordinates
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("警告: 图像预处理模块未找到，将使用默认处理方式")

# 导入新模块
from src.ocr_engine import OCREngine
from src.ocr_result_parser import OCRResultParser
from src.ocr_text_merger import OCRTextMerger
from src.pattern_matcher import PatternMatcher
from src.question_extractor import QuestionExtractor
from src.zone_extractor import ZoneExtractor
from src.keyword_matcher import KeywordMatcher


class LayoutAnalyzer:
    """页面布局分析类（重构版）"""
    
    def __init__(self, config=None):
        """
        初始化布局分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 初始化各个功能模块
        print("[初始化] 正在初始化各个功能模块...")
        
        # OCR引擎模块
        self.ocr_engine = OCREngine(config)
        self.ocr = self.ocr_engine.ocr
        self.use_ppstructure = self.ocr_engine.use_ppstructure
        
        # OCR结果解析模块
        self.ocr_parser = OCRResultParser(config)
        
        # OCR文本合并模块
        self.text_merger = OCRTextMerger(config)
        
        # 模式匹配模块
        self.pattern_matcher = PatternMatcher(config)
        
        # 题目提取模块
        self.question_extractor = QuestionExtractor(config, self.pattern_matcher)
        
        # 区域提取模块
        self.zone_extractor = ZoneExtractor(config, self.pattern_matcher)
        
        # 关键词匹配模块
        self.keyword_matcher = KeywordMatcher(config)
        
        # 保持兼容性：patterns属性
        if self.config:
            self.patterns = self.config.compile_patterns()
        else:
            self.patterns = {
                'question': [],
                'answer': []
            }
        
        print("[初始化] 所有模块初始化完成")
    
    def analyze_page(self, image: Image.Image, page_num: int = 1) -> tuple[List[Dict], List[Dict]]:
        """
        分析单页布局，识别题目和答案区域
        
        Args:
            image: PIL图片对象
            page_num: 页码（用于调试）
            
        Returns:
            (题目位置列表, OCR结果列表)
        """
        if not self.ocr:
            raise RuntimeError("OCR引擎未初始化，无法分析布局")
        
        # 将PIL图像转换为numpy数组（PaddleOCR需要的格式）
        original_array = np.array(image.convert('RGB'))
        original_shape = original_array.shape[:2]  # (height, width)
        
        # 图像预处理（如果启用）
        crop_info = {'x': 0, 'y': 0, 'scale': 1.0}
        enable_preprocessing = (
            self.config and 
            hasattr(self.config, 'enable_image_preprocessing') and 
            self.config.enable_image_preprocessing and
            PREPROCESSOR_AVAILABLE and
            CV2_AVAILABLE
        )
        
        is_blank_page = False
        if enable_preprocessing:
            print(f"  -> 图像预处理中（提升坐标精度）...")
            try:
                target_height = getattr(self.config, 'preprocess_target_height', 1280)
                auto_crop = getattr(self.config, 'preprocess_auto_crop', True)
                processed_array, crop_info, is_blank_page = preprocess_image(
                    original_array, 
                    target_height=target_height, 
                    auto_crop=auto_crop
                )
                
                # 如果检测到空白页，直接返回，跳过OCR识别
                if is_blank_page:
                    print(f"  [跳过] 空白页，跳过OCR识别，直接进入下一页")
                    return [], []
                
                bottom_crop = crop_info.get('bottom_crop', 0)
                print(f"  [预处理] 完成：裁剪偏移({crop_info['x']}, {crop_info['y']}), 缩放比例={crop_info['scale']:.3f}, 底部裁剪={bottom_crop}px")
            except Exception as e:
                print(f"  警告: 图像预处理失败: {e}，使用原图")
                processed_array = original_array
                crop_info = {'x': 0, 'y': 0, 'scale': 1.0}
        else:
            processed_array = original_array
        
        # 执行OCR
        print(f"  -> 正在执行OCR识别...")
        
        # 使用OCR引擎模块的方法
        ocr_response = self.ocr_engine.predict(processed_array)
        
        if not ocr_response:
            print(f"  警告: 第{page_num}页未识别到任何内容")
            return [], []
        
        # 处理不同引擎的返回格式
        if self.use_ppstructure:
            ocr_result = self.ocr_parser.parse_ppstructure_result(ocr_response)
        else:
            # 标准PaddleOCR返回格式
            first_result = ocr_response[0] if ocr_response else None
            if first_result is None:
                ocr_result = []
            else:
                result_type_name = type(first_result).__name__
                if result_type_name == 'OCRResult':
                    ocr_result = self.ocr_parser._convert_ocr_result_to_list(first_result)
                else:
                    ocr_result = first_result if isinstance(first_result, list) else []
        
        if not ocr_result:
            print(f"  警告: 第{page_num}页未识别到任何内容")
            return [], []
        
        print(f"  [成功] OCR识别完成，共 {len(ocr_result)} 行文本")
        
        # 坐标校准（如果启用了预处理且确实进行了变换）
        if enable_preprocessing and (crop_info['scale'] != 1.0 or crop_info['x'] != 0 or crop_info['y'] != 0):
            print(f"  -> 坐标校准中（映射回原图坐标）...")
            try:
                ocr_result = correct_coordinates(ocr_result, crop_info, original_shape)
                print(f"  [校准] 完成：已将坐标映射回原图")
            except Exception as e:
                print(f"  警告: 坐标校准失败: {e}，使用原始坐标")
        
        # 合并同一行的文本片段
        merged_ocr_result = self.text_merger.merge_same_line_texts(ocr_result)
        if len(merged_ocr_result) != len(ocr_result):
            print(f"  [合并] 行合并完成，从 {len(ocr_result)} 行合并为 {len(merged_ocr_result)} 行")
        
        # 分析OCR结果，提取题目位置
        # 注意：此时all_page_ocr_results还没有完整收集，所以先不传递跨页参数
        questions = self.question_extractor.extract_question_positions(merged_ocr_result, page_num)
        
        # 输出统计信息
        if len(questions) == 0:
            print(f"  [警告] 未识别到任何题目")
            print(f"     请检查:")
            print(f"     1. 识别规则是否匹配您的试卷格式")
            print(f"     2. OCR是否正确识别了文字")
            if self.config and self.config.mode != 'debug':
                print(f"     3. 使用 --mode debug 查看详细识别结果")
        
        # 返回题目位置和OCR结果（使用合并后的结果）
        return questions, merged_ocr_result
    
    def analyze_all_pages(self, images: List[Image.Image]) -> List[Dict]:
        """
        分析所有页面，识别题目位置并处理跨页情况
        
        Args:
            images: 所有页面的图片列表
            
        Returns:
            完整的题目列表（已处理跨页和边界）
        """
        all_questions = []
        page_ocr_results = {}  # 保存每页的OCR结果：page_num -> ocr_result
        
        # 第一步: 收集所有页面的OCR结果（先收集，再分析，以支持跨页检测）
        print("\n--- 收集所有页面的OCR结果 ---")
        all_ocr_results = {}
        for page_idx, image in enumerate(images, 1):
            print(f"OCR识别第 {page_idx} 页...")
            _, ocr_result = self.analyze_page(image, page_idx)
            all_ocr_results[page_idx] = ocr_result
        
        # 第二步: 基于OCR结果提取题目位置（支持跨页检测和合并）
        print("\n--- 收集所有页面的题目位置 ---")
        total_pages = len(images)
        pending_cross_page_question = None  # 待合并的跨页题目
        
        for page_idx, image in enumerate(images, 1):
            print(f"分析第 {page_idx} 页...")
            ocr_result = all_ocr_results[page_idx]
            
            # 如果有待合并的跨页题目，先尝试将当前页的嵌套子题合并到该题目中
            if pending_cross_page_question:
                cross_page_question, cross_page_from_page = pending_cross_page_question
                cross_page_to_page = cross_page_question.get('cross_page_to', page_idx)
                
                # 检查当前页是否是目标页（或者后续页，支持连续跨页）
                if page_idx >= cross_page_to_page:
                    # 在当前页的OCR结果中查找应该合并的嵌套子题
                    merged_sub_numbers = cross_page_question.get('nested_sub_numbers', [])
                    expected_next_number = max(merged_sub_numbers) + 1 if merged_sub_numbers else 1
                    merged_count = 0
                    
                    debug_mode = self.config and self.config.mode == 'debug'
                    
                    # 遍历当前页的OCR结果，查找匹配的嵌套子题
                    for line_idx, line_data in enumerate(ocr_result):
                        if not line_data or len(line_data) < 2:
                            continue
                        
                        line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
                        
                        # 检查是否是嵌套子题
                        if self.pattern_matcher.is_nested_sub_question(line_text):
                            nested_number = self.pattern_matcher.extract_nested_sub_number(line_text)
                            
                            # 检查是否是预期的下一个嵌套子题
                            if nested_number == expected_next_number:
                                # 合并到跨页题目中
                                merged_sub_numbers.append(nested_number)
                                cross_page_question['nested_sub_numbers'] = merged_sub_numbers.copy()
                                
                                # 更新预期下一个编号
                                expected_next_number = nested_number + 1
                                merged_count += 1
                                
                                if debug_mode:
                                    question_text = cross_page_question.get('text', '')[:30]
                                    print(f"    [跨页合并] 将嵌套子题({nested_number})合并到题目（来自第{cross_page_from_page}页）: {question_text}")
                                
                                # 继续查找当前页的下一个子题（不立即退出循环）
                                continue
                    
                    # 如果合并了子题，检查是否还有更多子题需要合并
                    if merged_count > 0:
                        # 检查是否还需要继续跨页（查看下一页是否还有下一个子题）
                        final_merged_sub_numbers = cross_page_question.get('nested_sub_numbers', [])
                        final_expected_next = max(final_merged_sub_numbers) + 1 if final_merged_sub_numbers else 1
                        needs_more_pages = False
                        
                        # 检查后续页面是否还有子题
                        for check_page_num in range(page_idx + 1, min(page_idx + 3, total_pages + 1)):
                            if check_page_num in all_ocr_results:
                                check_page_ocr = all_ocr_results[check_page_num]
                                for check_line_data in check_page_ocr[:3]:  # 只检查前3行
                                    if not check_line_data or len(check_line_data) < 2:
                                        continue
                                    
                                    check_line_text = check_line_data[1][0] if len(check_line_data[1]) > 0 else ""
                                    if self.pattern_matcher.is_nested_sub_question(check_line_text):
                                        check_nested_num = self.pattern_matcher.extract_nested_sub_number(check_line_text)
                                        if check_nested_num == final_expected_next:
                                            # 还有下一个子题在后续页，继续跨页
                                            needs_more_pages = True
                                            cross_page_question['cross_page_to'] = check_page_num
                                            if debug_mode:
                                                print(f"    [跨页合并] 检测到还有子题({final_expected_next})在第{check_page_num}页，继续跨页")
                                            break
                                if needs_more_pages:
                                    break
                        
                        # 根据是否需要继续跨页来决定是否完成合并
                        if needs_more_pages:
                            # 还需要继续跨页，保持pending状态
                            if debug_mode:
                                print(f"    [跨页合并] 已合并{merged_count}个子题，继续等待第{cross_page_question.get('cross_page_to')}页")
                        else:
                            # 跨页合并完成，添加到结果中
                            # 清除cross_page标记，因为合并已经完成，不再需要跨页处理
                            cross_page_question.pop('cross_page', None)
                            cross_page_question.pop('cross_page_to', None)
                            
                            # 计算跨页题目的实际结束位置（应该包括跨页部分的最后一行）
                            # 查找合并后的最后一个嵌套子题在目标页的位置
                            final_page_ocr = all_ocr_results.get(page_idx, [])
                            if final_page_ocr:
                                # 查找最后一个嵌套子题的位置
                                final_nested_numbers = cross_page_question.get('nested_sub_numbers', [])
                                if final_nested_numbers:
                                    last_nested_num = max(final_nested_numbers)
                                    for line_data in final_page_ocr:
                                        if not line_data or len(line_data) < 2:
                                            continue
                                        
                                        line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
                                        if self.pattern_matcher.is_nested_sub_question(line_text):
                                            nested_num = self.pattern_matcher.extract_nested_sub_number(line_text)
                                            if nested_num == last_nested_num:
                                                # 找到最后一个嵌套子题，记录其结束位置
                                                line_box = line_data[0]
                                                y_coords = [point[1] for point in line_box]
                                                line_y_max = max(y_coords)
                                                cross_page_question['cross_page_end_y'] = int(line_y_max)
                                                if debug_mode:
                                                    print(f"    [跨页合并] 记录最后一个嵌套子题({last_nested_num})的结束位置: y={line_y_max}")
                                                break
                            
                            cross_page_question['image_height'] = images[cross_page_from_page - 1].height
                            cross_page_question['image_width'] = images[cross_page_from_page - 1].width
                            cross_page_question['cross_page_merged'] = True  # 标记已完成跨页合并
                            cross_page_question['cross_page_end_page'] = page_idx  # 记录跨页结束页
                            all_questions.append((cross_page_from_page, cross_page_question, images[cross_page_from_page - 1]))
                            pending_cross_page_question = None
                            question_text = cross_page_question.get('text', '')[:30]
                            print(f"    [跨页合并] 完成跨页合并（第{cross_page_from_page}页 → 第{page_idx}页），共合并{merged_count}个子题: {question_text}")
            
            # 提取当前页的题目位置
            questions = self.question_extractor.extract_question_positions(
                ocr_result, 
                page_idx, 
                all_page_ocr_results=all_ocr_results,
                total_pages=total_pages
            )
            
            # 保存OCR结果（用于后续边界计算）
            page_ocr_results[page_idx] = ocr_result
            
            # 处理当前页的题目
            for q in questions:
                # 检查是否有跨页标记
                if q.get('cross_page'):
                    # 标记为待合并的跨页题目（推迟到目标页处理）
                    pending_cross_page_question = (q, page_idx)
                    print(f"    [跨页检测] 发现跨页题目，将在第{q.get('cross_page_to', page_idx + 1)}页合并嵌套子题")
                else:
                    # 正常题目，直接添加
                    q['image_height'] = image.height
                    q['image_width'] = image.width
                    all_questions.append((page_idx, q, image))
        
        # 如果还有待合并的跨页题目（可能目标页不存在），直接添加
        if pending_cross_page_question:
            cross_page_question, cross_page_from_page = pending_cross_page_question
            cross_page_question['image_height'] = images[cross_page_from_page - 1].height
            cross_page_question['image_width'] = images[cross_page_from_page - 1].width
            all_questions.append((cross_page_from_page, cross_page_question, images[cross_page_from_page - 1]))
            print(f"    [跨页合并] 跨页题目已添加（目标页可能不存在）")
        
        print(f"  共识别到 {len(all_questions)} 个题目位置")
        
        # 第三步: 计算每个题目区域内所有行的最小文字高度
        for i, (page_num, question, image) in enumerate(all_questions):
            question_start_y = question['start_y']
            question_end_y = question.get('end_y')  # 可能有也可能没有
            
            # 获取该题目所在页面的OCR结果
            ocr_result = page_ocr_results.get(page_num, [])
            
            # 收集该题目区域内所有行的text_height
            line_heights = []
            for line_data in ocr_result:
                if not line_data:
                    continue
                
                line_box = line_data[0]
                y_coords = [point[1] for point in line_box]
                line_y_min = min(y_coords)
                
                # 检查是否在当前题目区域内
                if line_y_min >= question_start_y:
                    # 如果有明确的end_y，检查是否超过
                    if question_end_y and line_y_min > question_end_y:
                        break  # 超过题目区域，停止收集
                    
                    # 计算该行的高度
                    line_y_max = max(y_coords)
                    text_height = int(line_y_max - line_y_min)
                    if text_height > 0:
                        line_heights.append(text_height)
            
            # 如果收集到多个行高，使用最小值
            if line_heights:
                min_text_height = min(line_heights)
                question['text_height'] = min_text_height
                if len(line_heights) > 1:
                    print(f"    题目{i+1}: 更新text_height为{min_text_height} (收集到{len(line_heights)}行)")
        
        # 第四步: 确定每个题目的边界
        print("\n--- 确定题目边界 ---")
        final_questions = []
        
        for i, (page_num, question, image) in enumerate(all_questions):
            # 获取下一个题目
            has_next = i + 1 < len(all_questions)
            next_question_info = all_questions[i + 1] if has_next else None
            
            # 确定结束位置
            # 优先处理已完成跨页合并的题目
            if question.get('cross_page_merged'):
                # 已完成跨页合并的题目：使用跨页结束位置
                cross_page_end_page = question.get('cross_page_end_page', page_num)
                cross_page_end_y = question.get('cross_page_end_y')
                
                if cross_page_end_y is not None:
                    # 使用记录的跨页结束位置
                    question['end_y'] = cross_page_end_y
                    question['end_page'] = cross_page_end_page
                    print(f"    题目{i+1}: 跨页合并题目，结束于第{cross_page_end_page}页的y={cross_page_end_y}")
                else:
                    # 如果没有记录结束位置，使用跨页结束页的页尾
                    end_page_image = images[cross_page_end_page - 1] if cross_page_end_page <= len(images) else image
                    question['end_y'] = int(end_page_image.height * 0.97)
                    question['end_page'] = cross_page_end_page
                    print(f"    题目{i+1}: 跨页合并题目，结束于第{cross_page_end_page}页末尾")
            # 处理尚未完成的跨页题目（这种情况应该很少，因为合并逻辑会在提取阶段完成）
            elif question.get('cross_page'):
                # 跨页题目：结束于当前页末尾，在边界计算后会合并到下一页
                question['end_y'] = int(image.height * 0.97)
                question['end_page'] = page_num
                cross_page_to = question.get('cross_page_to', page_num + 1)
                print(f"    题目{i+1}: 跨页题目，当前页结束于y={question['end_y']}，将合并到第{cross_page_to}页")
            # 优先使用提前确定的结束位置（如果累积嵌套子题时检测到）
            elif question.get('early_end_y') is not None:
                # 确保提前结束位置是整数
                early_end_y = int(question['early_end_y'])
                question_start_y = question.get('start_y', 0)
                
                # 验证提前结束位置的有效性：必须大于起始位置
                if early_end_y > question_start_y:
                    margin_offset = 0  # 提前结束位置已经是精确位置，不需要额外margin
                    question['end_y'] = early_end_y - margin_offset
                    question['end_page'] = page_num
                    early_end_text = question.get('early_end_text', '未知')
                    print(f"    题目{i+1}: 使用提前确定的结束位置 y={question['end_y']} (虚拟下一题: {early_end_text})")
                else:
                    # 提前结束位置无效（小于或等于起始位置），使用常规方式
                    print(f"    题目{i+1}: 提前结束位置无效 (y={early_end_y} <= start_y={question_start_y})，使用常规边界计算")
                    question['early_end_y'] = None  # 清除无效的提前结束位置
            
            # 如果没有提前结束位置或提前结束位置无效，使用常规边界计算
            if question.get('early_end_y') is None and not question.get('cross_page') and not question.get('cross_page_merged'):
                if has_next and next_question_info:
                    next_page, next_q, next_img = next_question_info
                    
                    # 判断是否同页
                    if next_page == page_num:
                        # 同页: 使用新的裁剪规则
                        next_question_y = next_q['start_y']
                        next_text_height = next_q.get('text_height', 0)
                        margin_offset = int(next_text_height * 0.3)
                        question['end_y'] = next_question_y - margin_offset
                        question['end_page'] = page_num
                        question['next_question_y'] = next_question_y
                        print(f"    题目{i+1}: 结束于第{page_num}页的y={question['end_y']} (下一题在y={next_question_y}, 高度={next_text_height})")
                    else:
                        # 可能跨页: 暂时标记为页尾，后面会处理
                        question['end_y'] = int(image.height * 0.97)
                        question['end_page'] = page_num
                        question['potential_cross_page'] = True
                        print(f"    题目{i+1}: 到第{page_num}页结尾")
                else:
                    # 最后一题: 到最后一页的页尾
                    question['end_y'] = int(image.height * 0.97)
                    question['end_page'] = len(images)
                    print(f"    题目{i+1}: 最后一页结尾")
            
            # 添加页面信息用于后续裁剪
            question['page_image'] = image
            question['pages_data'] = images
            
            final_questions.append(question)
        
        return final_questions
    
    # 以下方法保持向后兼容，委托给对应模块
    def _parse_ppstructure_result(self, ppstructure_response) -> List:
        """解析PPStructure结果（委托给OCRResultParser）"""
        return self.ocr_parser.parse_ppstructure_result(ppstructure_response)
    
    def _convert_ocr_result_to_list(self, ocr_result_obj) -> List:
        """转换OCRResult对象（委托给OCRResultParser）"""
        return self.ocr_parser._convert_ocr_result_to_list(ocr_result_obj)
    
    def _merge_same_line_texts(self, ocr_result: List) -> List:
        """合并同一行文本（委托给OCRTextMerger）"""
        return self.text_merger.merge_same_line_texts(ocr_result)
    
    def _extract_question_positions(self, ocr_result: List, page_num: int, 
                                    all_page_ocr_results: Dict = None, 
                                    total_pages: int = 0) -> List[Dict]:
        """提取题目位置（委托给QuestionExtractor）"""
        return self.question_extractor.extract_question_positions(
            ocr_result, page_num, all_page_ocr_results, total_pages
        )
    
    def _extract_zones(self, ocr_result: List, page_size: Tuple[int, int], page_num: int) -> List[Dict]:
        """提取区域（委托给ZoneExtractor）"""
        return self.zone_extractor.extract_zones(ocr_result, page_size, page_num)
    
    def _matches_pattern(self, text: str, pattern_type: str) -> bool:
        """匹配模式（委托给PatternMatcher）"""
        return self.pattern_matcher.matches_pattern(text, pattern_type)
    
    def _is_sub_question(self, text: str) -> bool:
        """检查是否是小题（委托给PatternMatcher）"""
        return self.pattern_matcher.is_sub_question(text)
    
    def _is_nested_sub_question(self, text: str) -> bool:
        """检查是否是嵌套小题（委托给PatternMatcher）"""
        return self.pattern_matcher.is_nested_sub_question(text)
    
    def _extract_nested_sub_number(self, text: str) -> int:
        """提取嵌套子题编号（委托给PatternMatcher）"""
        return self.pattern_matcher.extract_nested_sub_number(text)
    
    def _merge_adjacent_zones(self, zones: List[Dict]) -> List[Dict]:
        """合并相邻区域（委托给ZoneExtractor）"""
        return self.zone_extractor.merge_adjacent_zones(zones)
    
    def extract_keyword_following_texts(self, paragraphs: List[str]) -> List[str]:
        """提取关键词后续文本（委托给KeywordMatcher）"""
        return self.keyword_matcher.extract_keyword_following_texts(paragraphs)
    
    def fuzzy_match_keywords_in_ocr(self, keyword_texts: List[str], ocr_result: List) -> List[Dict]:
        """模糊匹配关键词（委托给KeywordMatcher）"""
        return self.keyword_matcher.fuzzy_match_keywords_in_ocr(keyword_texts, ocr_result)
    
    def _is_keyword_text(self, text: str) -> bool:
        """检查是否是关键词文本（委托给KeywordMatcher）"""
        return self.keyword_matcher.is_keyword_text(text)
    
    # 注意：_is_same_line 和 _merge_line_texts 是 OCRTextMerger 的内部方法
    # 如果需要外部调用，可以通过 text_merger 直接访问
    # 这里不提供委托方法，因为它们的实现依赖于OCRTextMerger的内部逻辑

