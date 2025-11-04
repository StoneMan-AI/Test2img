#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
布局分析模块
核心模块：识别页面布局，定位题目和答案的位置
"""

from typing import List, Tuple, Dict
from PIL import Image
import numpy as np
import re

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("警告: OpenCV未安装，图像预处理功能将不可用")

# 尝试导入PPStructure（可能是PPStructure或PPStructureV3）
PPSTRUCTURE_AVAILABLE = False
PPStructure = None

try:
    from paddleocr import PPStructureV3 as PPStructure
    PPSTRUCTURE_AVAILABLE = True
except ImportError:
    try:
        from paddleocr import PPStructure
        PPSTRUCTURE_AVAILABLE = True
    except ImportError as e:
        PPSTRUCTURE_AVAILABLE = False
        print(f"[提示] PPStructure未导入成功 ({str(e)})，将使用标准PaddleOCR")
        PPStructure = None

# 备用：标准PaddleOCR（如果PPStructure不可用）
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None

# 导入图像预处理模块
try:
    from src.image_preprocessor import preprocess_image, correct_coordinates
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("警告: 图像预处理模块未找到，将使用默认处理方式")


class LayoutAnalyzer:
    """页面布局分析类"""
    
    def __init__(self, config=None):
        """
        初始化布局分析器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 初始化OCR引擎（优先使用PPStructure，否则使用标准PaddleOCR）
        self.use_ppstructure = False
        
        # 优先使用PPStructure（专为文档优化）
        # 检查配置是否启用PPStructure
        use_ppstructure_config = (
            self.config and 
            hasattr(self.config, 'use_ppstructure') and 
            self.config.use_ppstructure
        ) or (not self.config)  # 如果没有配置，默认启用
        
        if use_ppstructure_config and PPSTRUCTURE_AVAILABLE and PPStructure:
            try:
                print("[初始化] 尝试使用PPStructure（文档优化版本）...")
                ppstructure_params = {
                    'lang': self.config.ocr_lang if self.config else "ch",
                    # 默认只启用文本检测和识别，不启用表格等其他功能（加快速度）
                    'use_table_recognition': False,
                    'use_formula_recognition': False,
                    'use_chart_recognition': False,
                    'use_seal_recognition': False,
                    'use_doc_orientation_classify': False,
                    'use_doc_unwarping': False,
                }
                
                # PPStructureV3使用的参数名与标准PaddleOCR不同
                # 如果配置了模型路径，则添加（使用PPStructureV3的参数名）
                if self.config and self.config.det_model_dir:
                    ppstructure_params['text_detection_model_dir'] = self.config.det_model_dir
                if self.config and self.config.rec_model_dir:
                    ppstructure_params['text_recognition_model_dir'] = self.config.rec_model_dir
                
                # 添加文本检测相关参数（如果配置了）
                if self.config:
                    if hasattr(self.config, 'det_db_box_thresh'):
                        ppstructure_params['text_det_box_thresh'] = self.config.det_db_box_thresh
                    if hasattr(self.config, 'det_limit_side_len'):
                        ppstructure_params['text_det_limit_side_len'] = self.config.det_limit_side_len
                    if hasattr(self.config, 'det_limit_type'):
                        ppstructure_params['text_det_limit_type'] = self.config.det_limit_type
                
                self.ocr = PPStructure(**ppstructure_params)
                self.use_ppstructure = True
                print("[成功] PPStructure引擎初始化成功（文档优化版本）")
                if self.config and (self.config.det_model_dir or self.config.rec_model_dir):
                    print(f"  -> 使用自定义模型路径")
                print(f"  -> 语言: {ppstructure_params['lang']}")
            except Exception as e:
                error_msg = str(e)
                print(f"[警告] PPStructure初始化失败")
                
                # 检查是否是依赖问题
                if "dependency" in error_msg.lower() or "DependencyError" in error_msg:
                    print(f"  -> 原因: PPStructureV3需要额外的依赖包")
                    print(f"  -> 解决方案: 运行以下命令安装依赖")
                    print(f"     pip install \"paddlex[ocr]\"")
                    print(f"  -> 或者继续使用标准PaddleOCR（当前降级方案）")
                else:
                    print(f"  -> 错误详情: {error_msg}")
                
                import traceback
                if self.config and self.config.mode == 'debug':
                    traceback.print_exc()
                
                print(f"[降级] 将尝试使用标准PaddleOCR作为备用...")
                self.use_ppstructure = False
                self.ocr = None
        else:
            # PPStructure不可用，说明原因
            if not PPSTRUCTURE_AVAILABLE:
                print("[提示] PPStructure未导入成功，将使用标准PaddleOCR")
            elif not PPStructure:
                print("[提示] PPStructure为None，将使用标准PaddleOCR")
        
        # 如果PPStructure不可用，使用标准PaddleOCR作为备用
        if not self.use_ppstructure:
            if PADDLEOCR_AVAILABLE and PaddleOCR:
                try:
                    print("[初始化] 使用标准PaddleOCR...")
                    # 构建PaddleOCR初始化参数
                    ocr_params = {
                        'use_angle_cls': self.config.use_angle_cls if self.config else True,
                        'lang': self.config.ocr_lang if self.config else "ch",
                        'det_db_box_thresh': self.config.det_db_box_thresh if self.config else 0.45,
                        'det_db_unclip_ratio': self.config.det_db_unclip_ratio if self.config else 1.6
                    }
                    
                    # 添加图像预处理相关参数（用于提升坐标精度）
                    if self.config:
                        # 尝试使用新版本参数名，如果失败则使用旧参数名
                        try:
                            ocr_params['text_det_limit_side_len'] = self.config.det_limit_side_len
                            ocr_params['text_det_limit_type'] = self.config.det_limit_type
                        except AttributeError:
                            # 如果没有配置这些参数，使用默认值
                            try:
                                ocr_params['det_limit_side_len'] = self.config.det_limit_side_len if hasattr(self.config, 'det_limit_side_len') else 2048
                                ocr_params['det_limit_type'] = self.config.det_limit_type if hasattr(self.config, 'det_limit_type') else 'max'
                            except:
                                pass
                    
                    # 如果配置了模型路径，则添加
                    if self.config and self.config.det_model_dir:
                        ocr_params['det_model_dir'] = self.config.det_model_dir
                    if self.config and self.config.rec_model_dir:
                        ocr_params['rec_model_dir'] = self.config.rec_model_dir
                    
                    self.ocr = PaddleOCR(**ocr_params)
                    print("[成功] PaddleOCR引擎初始化成功（标准版本）")
                    if self.config and (self.config.det_model_dir or self.config.rec_model_dir):
                        print(f"  -> 使用自定义模型路径")
                    print(f"  -> det_db_box_thresh={ocr_params['det_db_box_thresh']}, det_db_unclip_ratio={ocr_params['det_db_unclip_ratio']}")
                except Exception as e:
                    print(f"[错误] OCR引擎初始化失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.ocr = None
            else:
                print("[错误] PaddleOCR不可用，无法初始化OCR引擎")
                self.ocr = None
        
        # 编译正则表达式模式（提升性能）
        if self.config:
            self.patterns = self.config.compile_patterns()
        else:
            self.patterns = {
                'question': [],
                'answer': []
            }
    
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
        
        # PPStructure 和标准 PaddleOCR 的调用方式不同
        if self.use_ppstructure:
            # PPStructureV3 使用 .predict() 方法
            # predict 方法返回的是布局分析结果，格式为列表: [{'type': 'text', 'bbox': [...], 'res': [...]}, ...]
            ocr_response = self.ocr.predict(processed_array)
        else:
            # 标准 PaddleOCR 使用 .ocr() 方法
            ocr_response = self.ocr.ocr(processed_array)
        
        if not ocr_response:
            print(f"  警告: 第{page_num}页未识别到任何内容")
            return [], []
        
        # 处理不同引擎的返回格式
        if self.use_ppstructure:
            # PPStructureV3 的 predict 返回格式: 列表，每个元素是 LayoutParsingResultV2 对象
            # 每个对象包含：rec_texts, rec_polys, rec_scores 等属性
            if isinstance(ocr_response, list):
                # 调试信息
                if self.config and self.config.mode == 'debug':
                    print(f"  调试: PPStructure返回类型={type(ocr_response)}, 长度={len(ocr_response)}")
                    if len(ocr_response) > 0:
                        first_item = ocr_response[0]
                        print(f"  调试: 第一个元素类型={type(first_item)}")
                        if hasattr(first_item, '__getitem__'):
                            try:
                                keys = list(first_item.keys()) if hasattr(first_item, 'keys') else []
                                print(f"  调试: 可用键: {keys[:10]}...")  # 只显示前10个
                                if 'rec_texts' in first_item or 'rec_texts' in keys:
                                    texts = first_item.get('rec_texts', []) if hasattr(first_item, 'get') else first_item['rec_texts']
                                    print(f"  调试: rec_texts数量={len(texts) if texts else 0}")
                            except Exception as e:
                                print(f"  调试: 访问键时出错: {e}")
                
                ocr_result = self._parse_ppstructure_result(ocr_response)
            else:
                # 如果返回的不是列表，尝试包装
                ocr_result = self._parse_ppstructure_result([ocr_response] if ocr_response else [])
        else:
            # 标准 PaddleOCR 返回格式
            first_result = ocr_response[0]
            result_type_name = type(first_result).__name__
            
            if result_type_name == 'OCRResult':
                # 新版本：OCRResult对象
                if self.config and self.config.mode == 'debug':
                    print(f"  调试: 检测到OCRResult类型")
                ocr_result = self._convert_ocr_result_to_list(first_result)
            else:
                # 旧版本：直接是列表
                if self.config and self.config.mode == 'debug':
                    print(f"  调试: 使用旧版本列表格式")
                ocr_result = first_result if isinstance(first_result, list) else []
        
        if not ocr_result:
            print(f"  警告: 第{page_num}页未识别到任何内容")
            if self.config and self.config.mode == 'debug':
                if self.use_ppstructure:
                    print(f"  调试: PPStructure返回类型={type(ocr_response)}, 长度={len(ocr_response) if ocr_response else 0}")
                else:
                    print(f"  调试: first_result类型={type(first_result) if 'first_result' in locals() else 'N/A'}")
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
        merged_ocr_result = self._merge_same_line_texts(ocr_result)
        if len(merged_ocr_result) != len(ocr_result):
            print(f"  [合并] 行合并完成，从 {len(ocr_result)} 行合并为 {len(merged_ocr_result)} 行")
        
        # 分析OCR结果，提取题目位置
        # 注意：此时all_page_ocr_results还没有完整收集，所以先不传递跨页参数
        # 跨页检测将在analyze_all_pages中统一处理
        questions = self._extract_question_positions(merged_ocr_result, page_num)
        
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
        
        # 第二步: 基于OCR结果提取题目位置（支持跨页检测）
        print("\n--- 收集所有页面的题目位置 ---")
        total_pages = len(images)
        for page_idx, image in enumerate(images, 1):
            print(f"分析第 {page_idx} 页...")
            ocr_result = all_ocr_results[page_idx]
            questions = self._extract_question_positions(
                ocr_result, 
                page_idx, 
                all_page_ocr_results=all_ocr_results,
                total_pages=total_pages
            )
            
            # 保存OCR结果（用于后续边界计算）
            page_ocr_results[page_idx] = ocr_result
            
            for q in questions:
                q['image_height'] = image.height
                q['image_width'] = image.width
                all_questions.append((page_idx, q, image))
        
        print(f"  共识别到 {len(all_questions)} 个题目位置")
        
        # 第二步: 计算每个题目区域内所有行的最小文字高度
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
        
        # 第三步: 确定每个题目的边界
        print("\n--- 确定题目边界 ---")
        final_questions = []
        
        for i, (page_num, question, image) in enumerate(all_questions):
            # 获取下一个题目
            has_next = i + 1 < len(all_questions)
            next_question_info = all_questions[i + 1] if has_next else None
            
            # 确定结束位置
            # 优先处理跨页情况
            if question.get('cross_page'):
                # 跨页题目：结束于当前页末尾，在边界计算后会合并到下一页
                question['end_y'] = int(image.height * 0.97)
                question['end_page'] = page_num
                cross_page_to = question.get('cross_page_to', page_num + 1)
                print(f"    题目{i+1}: 跨页题目，当前页结束于y={question['end_y']}，将合并到第{cross_page_to}页")
            # 优先使用提前确定的结束位置（如果累积嵌套子题时检测到）
            # 提前结束位置应该被当作虚拟的下一题处理
            elif question.get('early_end_y') is not None:
                # 确保提前结束位置是整数
                early_end_y = int(question['early_end_y'])
                question_start_y = question.get('start_y', 0)
                
                # 验证提前结束位置的有效性：必须大于起始位置
                if early_end_y > question_start_y:
                    # 将提前结束位置视为虚拟下一题，应用margin处理
                    # 使用一个合理的margin（类似下一题的处理方式）
                    # 如果提前结束的文本本身是题目，应该留出一些间距
                    margin_offset = 0  # 提前结束位置已经是精确位置，不需要额外margin
                    question['end_y'] = early_end_y - margin_offset
                    question['end_page'] = page_num
                    early_end_text = question.get('early_end_text', '未知')
                    print(f"    题目{i+1}: 使用提前确定的结束位置 y={question['end_y']} (虚拟下一题: {early_end_text})")
                else:
                    # 提前结束位置无效（小于或等于起始位置），使用常规方式
                    print(f"    题目{i+1}: 提前结束位置无效 (y={early_end_y} <= start_y={question_start_y})，使用常规边界计算")
                    question['early_end_y'] = None  # 清除无效的提前结束位置
                    # 继续使用常规边界计算逻辑（fall through）
            
            # 如果没有提前结束位置或提前结束位置无效，使用常规边界计算
            if question.get('early_end_y') is None and not question.get('cross_page'):
                if has_next and next_question_info:
                    next_page, next_q, next_img = next_question_info
                    
                    # 判断是否同页
                    if next_page == page_num:
                        # 同页: 使用新的裁剪规则
                        # 裁剪结束位置 = next_question_y - next_text_height * 0.3
                        next_question_y = next_q['start_y']  # 下一题的y_min
                        next_text_height = next_q.get('text_height', 0)  # 下一题的文字高度
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
    
    def _parse_ppstructure_result(self, ppstructure_response) -> List:
        """
        解析PPStructure的返回结果，转换为标准OCR格式
        
        Args:
            ppstructure_response: PPStructureV3返回的结果（LayoutParsingResultV2对象列表）
        
        Returns:
            标准格式的OCR结果列表: [[坐标框, (文本, 置信度)], ...]
        """
        ocr_result = []
        
        if not ppstructure_response:
            return ocr_result
        
        # PPStructureV3返回格式: [LayoutParsingResultV2对象, ...]
        # 每个对象包含 overall_ocr_res 属性，类型为 OCRResult
        for page_result in ppstructure_response:
            # 检查是否是字典或支持字典访问的对象
            if not (isinstance(page_result, dict) or hasattr(page_result, '__getitem__')):
                continue
            
            # PPStructureV3的关键：从 overall_ocr_res 获取OCR结果
            overall_ocr_res = None
            
            # 尝试多种方式访问 overall_ocr_res
            if isinstance(page_result, dict):
                overall_ocr_res = page_result.get('overall_ocr_res', None)
            elif hasattr(page_result, '__getitem__'):
                try:
                    overall_ocr_res = page_result.get('overall_ocr_res', None) if hasattr(page_result, 'get') else page_result['overall_ocr_res']
                except (KeyError, TypeError):
                    overall_ocr_res = getattr(page_result, 'overall_ocr_res', None)
            
            # 如果还没有获取到，尝试属性访问
            if overall_ocr_res is None:
                overall_ocr_res = getattr(page_result, 'overall_ocr_res', None)
            
            # 调试信息
            if self.config and self.config.mode == 'debug':
                print(f"  调试: overall_ocr_res类型={type(overall_ocr_res) if overall_ocr_res else None}")
                if overall_ocr_res:
                    print(f"  调试: overall_ocr_res长度={len(overall_ocr_res) if hasattr(overall_ocr_res, '__len__') else 'N/A'}")
            
            # overall_ocr_res 是 OCRResult 对象，使用已有的转换方法
            if overall_ocr_res is not None:
                # 检查是否是OCRResult对象或列表
                if hasattr(overall_ocr_res, 'rec_texts') or isinstance(overall_ocr_res, dict):
                    # 使用已有的转换方法
                    converted = self._convert_ocr_result_to_list(overall_ocr_res)
                    ocr_result.extend(converted)
                    if self.config and self.config.mode == 'debug':
                        print(f"  调试: 从overall_ocr_res转换得到 {len(converted)} 条OCR结果")
                elif isinstance(overall_ocr_res, list):
                    # 如果是列表，每个元素可能是OCRResult或标准格式
                    for item in overall_ocr_res:
                        if hasattr(item, 'rec_texts') or isinstance(item, dict):
                            converted = self._convert_ocr_result_to_list(item)
                            ocr_result.extend(converted)
                        elif isinstance(item, list) and len(item) >= 2:
                            # 标准格式: [坐标框, (文本, 置信度)]
                            ocr_result.append(item)
        
        return ocr_result
    
    def _convert_ocr_result_to_list(self, ocr_result_obj) -> List:
        """
        将新版本OCRResult对象转换为旧版本格式的列表
        
        Args:
            ocr_result_obj: OCRResult对象
            
        Returns:
            转换后的列表
        """
        result_list = []
        
        # 调试信息
        debug_mode = self.config and self.config.mode == 'debug'
        if debug_mode:
            attrs = dir(ocr_result_obj)
            print(f"  调试: OCRResult属性数量={len(attrs)}")
        
        # 尝试多种方式获取文本和坐标
        texts = None
        polys = None
        scores = None
        
        # 方式1：尝试直接访问
        if hasattr(ocr_result_obj, 'rec_texts'):
            texts = ocr_result_obj.rec_texts
        # 方式2：尝试字典访问
        elif isinstance(ocr_result_obj, dict) and 'rec_texts' in ocr_result_obj:
            texts = ocr_result_obj['rec_texts']
        
        if hasattr(ocr_result_obj, 'rec_polys'):
            polys = ocr_result_obj.rec_polys
        elif isinstance(ocr_result_obj, dict) and 'rec_polys' in ocr_result_obj:
            polys = ocr_result_obj['rec_polys']
        
        if hasattr(ocr_result_obj, 'rec_scores'):
            scores = ocr_result_obj.rec_scores
        elif isinstance(ocr_result_obj, dict) and 'rec_scores' in ocr_result_obj:
            scores = ocr_result_obj['rec_scores']
        
        if debug_mode:
            print(f"  调试: texts={texts is not None}, polys={polys is not None}, scores={scores is not None}")
            if texts:
                print(f"  调试: 文本数量={len(texts)}, 前3个={texts[:3] if len(texts) > 3 else texts}")
        
        if texts is not None and polys is not None:
            scores = scores if scores is not None else [1.0] * len(texts)
            
            for i, (text, poly) in enumerate(zip(texts, polys)):
                # 转换坐标为旧格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                coord_list = poly.tolist()
                
                # 获取置信度
                conf = scores[i] if i < len(scores) else 1.0
                
                # 构建旧格式：[坐标, (文本, 置信度)]
                result_list.append([coord_list, (text, conf)])
        
        if debug_mode:
            print(f"  调试: 转换后列表长度={len(result_list)}")
        
        return result_list
    
    def _merge_same_line_texts(self, ocr_result: List) -> List:
        """
        合并同一行的文本片段
        使用基于关键词的定位策略
        
        Args:
            ocr_result: OCR结果列表
            
        Returns:
            合并后的OCR结果列表
        """
        if not ocr_result:
            return ocr_result
        
        # 计算每个文本的中心点和高度
        text_info_list = []
        for item in ocr_result:
            bbox = item[0]
            text = item[1][0] if len(item[1]) > 0 else ""
            conf = item[1][1] if len(item[1]) > 1 else 1.0
            
            # 计算中心点和尺寸
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            height = max(y_coords) - min(y_coords)
            
            # 判断是否包含题目关键词
            is_keyword = self._is_keyword_text(text)
            
            text_info_list.append({
                'text': text,
                'bbox': bbox,
                'conf': conf,
                'center': (center_x, center_y),
                'height': height,
                'x_min': min(x_coords),
                'x_max': max(x_coords),
                'y_min': min(y_coords),
                'y_max': max(y_coords),
                'is_keyword': is_keyword
            })
        
        # 计算平均文字高度
        avg_height = np.mean([info['height'] for info in text_info_list])
        
        # 使用自适应阈值进行行聚类
        threshold = avg_height * 0.8  # 同一行的Y轴差异不超过平均高度的80%
        
        # 基于关键词的合并策略
        lines = []
        used_indices = set()
        
        # 第一步：先处理包含关键词的文本
        for i, info in enumerate(text_info_list):
            if info['is_keyword'] and i not in used_indices:
                current_line = [i]
                used_indices.add(i)
                
                # 寻找同一行的其他文本（以关键词为基准）
                for j, other_info in enumerate(text_info_list):
                    if j in used_indices or i == j:
                        continue
                    
                    # 判断是否在同一行
                    if self._is_same_line(info, other_info, threshold):
                        current_line.append(j)
                        used_indices.add(j)
                
                lines.append([text_info_list[idx] for idx in current_line])
        
        # 第二步：处理剩余的非关键词文本
        for i, info in enumerate(text_info_list):
            if i in used_indices:
                continue
            
            current_line = [i]
            used_indices.add(i)
            
            # 寻找同一行的其他文本
            for j, other_info in enumerate(text_info_list):
                if j in used_indices or i == j:
                    continue
                
                # 判断是否在同一行
                if self._is_same_line(info, other_info, threshold):
                    current_line.append(j)
                    used_indices.add(j)
            
            lines.append([text_info_list[idx] for idx in current_line])
        
        # 合并同一行的文本
        merged_result = []
        for line in lines:
            if len(line) == 1:
                # 单独文本，直接添加
                info = line[0]
                merged_result.append([info['bbox'], (info['text'], info['conf'])])
            else:
                # 多个文本片段，需要合并
                merged_item = self._merge_line_texts(line)
                merged_result.append(merged_item)
        
        return merged_result
    
    def extract_keyword_following_texts(self, paragraphs: List[str]) -> List[str]:
        """
        从文本段落中提取关键词后面的1-5个中文字符
        
        Args:
            paragraphs: DOCX提取的段落文本列表
            
        Returns:
            关键词后续文本列表（1-5个字符）
        """
        if not self.patterns:
            return []
        
        keyword_texts = []
        
        for para in paragraphs:
            text = para.strip()
            if not text:
                continue
            
            # 匹配中文大题模式
            chinese_pattern = self.patterns.get('chinese_number')
            if chinese_pattern:
                match = chinese_pattern.match(text)
                if match:
                    # 提取关键词后面的文本
                    remaining = text[match.end():].strip()
                    # 提取1-5个中文字符
                    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', remaining)
                    if 1 <= len(chinese_chars) <= 5:
                        keyword_text = ''.join(chinese_chars)
                        keyword_texts.append(keyword_text)
                        continue
            
            # 匹配阿拉伯数字小题模式
            sub_patterns = self.patterns.get('sub_question', [])
            for pattern in sub_patterns:
                match = pattern.match(text)
                if match:
                    # 对于阿拉伯数字模式，只匹配数字和点，不包含后续内容
                    # 所以需要特殊处理
                    # 查找"数字."的位置
                    digit_dot_match = re.match(r'^\d+\.', text)
                    if digit_dot_match:
                        remaining = text[digit_dot_match.end():].strip()
                    else:
                        remaining = text[match.end():].strip()
                    
                    chinese_chars = re.findall(r'[\u4e00-\u9fa5]', remaining)
                    if 1 <= len(chinese_chars) <= 5:
                        keyword_text = ''.join(chinese_chars)
                        keyword_texts.append(keyword_text)
                        break
        
        return keyword_texts
    
    def fuzzy_match_keywords_in_ocr(self, keyword_texts: List[str], ocr_result: List) -> List[Dict]:
        """
        在OCR结果中模糊匹配关键词后续文本
        
        Args:
            keyword_texts: 关键词后续文本列表
            ocr_result: OCR识别结果
            
        Returns:
            匹配到的题目位置信息
        """
        questions = []
        
        for line_data in ocr_result:
            if not line_data:
                continue
            
            line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
            line_box = line_data[0]
            
            # 计算y坐标
            y_coords = [point[1] for point in line_box]
            line_y_min = min(y_coords)
            line_y_max = max(y_coords)
            
            # 检查是否包含任一关键词文本
            for keyword_text in keyword_texts:
                if keyword_text in line_text:
                    # 找到匹配
                    question_info = {
                        'text': line_text,
                        'y_min': int(line_y_min),
                        'y_max': int(line_y_max),
                        'matched_keyword': keyword_text,
                        'bbox': line_box
                    }
                    questions.append(question_info)
                    break  # 每个文本只匹配一次
        
        return questions
    
    def _is_keyword_text(self, text: str) -> bool:
        """
        判断文本是否包含题目关键词
        
        Args:
            text: 待判断文本
            
        Returns:
            是否包含关键词
        """
        if not self.patterns:
            return False
        
        text_stripped = text.strip()
        
        # 检查是否匹配中文大题模式
        chinese_pattern = self.patterns.get('chinese_number')
        if chinese_pattern and chinese_pattern.match(text_stripped):
            return True
        
        # 检查是否匹配阿拉伯数字小题模式
        sub_patterns = self.patterns.get('sub_question', [])
        for pattern in sub_patterns:
            if pattern.match(text_stripped):
                return True
        
        # 检查是否匹配其他题目模式
        question_patterns = self.patterns.get('question', [])
        for pattern in question_patterns:
            if pattern.match(text_stripped):
                return True
        
        return False
    
    def _is_same_line(self, info1: Dict, info2: Dict, threshold: float) -> bool:
        """
        判断两个文本是否在同一行
        
        Args:
            info1: 文本1的信息
            info2: 文本2的信息
            threshold: Y坐标差异阈值
            
        Returns:
            是否在同一行
        """
        # 计算Y中心点的差异
        y_diff = abs(info1['center'][1] - info2['center'][1])
        
        # 如果Y差异小于阈值，认为在同一行
        if y_diff <= threshold:
            return True
        
        # 如果两个文本的y_min和y_max有重叠，也可能在同一行
        overlap_height = min(info1['y_max'], info2['y_max']) - max(info1['y_min'], info2['y_min'])
        if overlap_height > 0 and overlap_height / max(info1['height'], info2['height']) > 0.3:
            return True
        
        return False
    
    def _merge_line_texts(self, line_texts: List[Dict]) -> List:
        """
        合并同一行的多个文本片段
        
        Args:
            line_texts: 同一行的文本信息列表
            
        Returns:
            合并后的OCR结果项
        """
        if not line_texts:
            return []
        
        if len(line_texts) == 1:
            info = line_texts[0]
            return [info['bbox'], (info['text'], info['conf'])]
        
        # 按X坐标排序
        line_texts.sort(key=lambda x: x['x_min'])
        
        # 合并文本
        merged_text = ' '.join([info['text'] for info in line_texts])
        
        # 合并置信度（取平均值）
        avg_conf = np.mean([info['conf'] for info in line_texts])
        
        # 提取关键词后面的1-2个中文字
        chinese_chars = re.findall(r'[\u4e00-\u9fa5]', merged_text)
        target_chars = chinese_chars[:2] if len(chinese_chars) >= 2 else chinese_chars
        
        # 找到包含这些中文字的文本片段
        bbox_texts = []
        if target_chars:
            for target_char in target_chars:
                for info in line_texts:
                    if target_char in info['text'] and info not in bbox_texts:
                        bbox_texts.append(info)
                        break
        else:
            # 如果没有中文字，使用第一个文本片段
            bbox_texts = [line_texts[0]]
        
        # 计算关键词后面的1-2个中文字的边界框
        if bbox_texts:
            x_min = min([info['x_min'] for info in bbox_texts])
            x_max = max([info['x_max'] for info in bbox_texts])
            y_min = min([info['y_min'] for info in bbox_texts])
            y_max = max([info['y_max'] for info in bbox_texts])
        else:
            # 如果没有找到，使用整个行的边界
            x_min = min([info['x_min'] for info in line_texts])
            x_max = max([info['x_max'] for info in line_texts])
            y_min = min([info['y_min'] for info in line_texts])
            y_max = max([info['y_max'] for info in line_texts])
        
        # 构造新的bbox
        merged_bbox = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        return [merged_bbox, (merged_text, avg_conf)]
    
    def _extract_question_positions(self, ocr_result: List, page_num: int, 
                                    all_page_ocr_results: Dict = None, 
                                    total_pages: int = 0) -> List[Dict]:
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
            
            # 检查是否是中文数字大题
            chinese_num_pattern = self.patterns.get('chinese_number')
            is_current_chinese = chinese_num_pattern and chinese_num_pattern.match(line_text.strip())
            
            # 如果是中文大题
            if is_current_chinese:
                # 先保存当前题目
                if current_question:
                    questions.append(current_question)
                
                # 重置嵌套子题编号列表（开始新题目）
                nested_sub_numbers = []
                
                # 检查是否重复位置
                y_pos = int(line_y_min)
                if y_pos in seen_start_positions:
                    print(f"    ⚠ 跳过重复位置: y={y_pos}, 文本={line_text[:30]}")
                    continue
                
                seen_start_positions.add(y_pos)
                
                # 计算文字高度
                text_height = int(line_y_max - line_y_min)
                
                # 开始新的中文大题
                current_question = {
                    'type': 'question',
                    'start_y': y_pos,
                    'start_x': 0,
                    'text': line_text,
                    'page': page_num,
                    'is_chinese_question': True,
                    'question_type': 'chinese',
                    'text_height': text_height,  # 保存文字高度
                    'nested_sub_numbers': []  # 初始化嵌套子题编号列表
                }
                is_chinese_question = True
                print(f"    [发现] 发现大题 (行{len(questions)+1}): {line_text[:30]} (y={y_pos}, h={text_height})")
            
            # 检查是否是阿拉伯数字小题
            elif self._is_sub_question(line_text):
                # 如果当前在中文大题下，不创建新的独立题目，继续累积到大题中
                if is_chinese_question and current_question:
                    # 继续累积到当前大题中
                    if debug_mode:
                        print(f"    [小题] 继续累积到大题: {line_text[:30]}")
                    # 不创建新题目，继续累积
                    continue
                else:
                    # 不在大题下，创建新的独立小题
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
                        'is_arabic_question': True,  # 标记为阿拉伯数字小题
                        'question_type': 'sub',
                        'text_height': text_height,  # 保存文字高度
                        'nested_sub_numbers': []  # 初始化嵌套子题编号列表
                    }
                    is_chinese_question = False
                    print(f"    [发现] 发现小题 (行{len(questions)+1}): {line_text[:30]} (y={y_pos}, h={text_height})")
            
            # 检查是否是嵌套小题（(1)、（2）、[3]等，应累积到上一级题目中）
            elif self._is_nested_sub_question(line_text):
                # 如果当前有任何题目（阿拉伯数字小题或普通题目），继续累积
                if current_question:
                    # 提取嵌套子题编号
                    nested_number = self._extract_nested_sub_number(line_text)
                    
                    if nested_number is not None:
                        # 记录嵌套子题编号
                        nested_sub_numbers.append(nested_number)
                        current_question['nested_sub_numbers'] = nested_sub_numbers.copy()
                        
                        # 检查后续3行是否按顺序出现下一个数字子题
                        next_expected_number = nested_number + 1
                        found_next_nested = False
                        early_end_y = None
                        early_end_text = None
                        
                        # 检查后续3行（支持跨页检测）
                        look_ahead_count = 3
                        checked_lines = 0
                        current_page_lines_remaining = len(ocr_result) - idx - 1
                        
                        # 先检查当前页的剩余行
                        lines_to_check_current_page = min(look_ahead_count, current_page_lines_remaining)
                        
                        for next_idx in range(idx + 1, idx + 1 + lines_to_check_current_page):
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
                            if self._is_nested_sub_question(next_line_text):
                                next_nested_num = self._extract_nested_sub_number(next_line_text)
                                if next_nested_num == next_expected_number:
                                    # 找到了按顺序的下一个子题，继续累积
                                    found_next_nested = True
                                    if debug_mode:
                                        print(f"    [嵌套小题] 找到下一个子题 ({next_expected_number})，继续累积")
                                    break
                                else:
                                    # 是嵌套子题但编号不匹配（比如期望(5)但遇到(1)）
                                    # 如果还没有记录early_end_y，记录这个文本
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
                                        if self._is_nested_sub_question(next_line_text):
                                            next_nested_num = self._extract_nested_sub_number(next_line_text)
                                            if next_nested_num == next_expected_number:
                                                # 找到了按顺序的下一个子题（在下一页），标记跨页
                                                found_next_nested = True
                                                current_question['cross_page'] = True
                                                current_question['cross_page_to'] = next_page_num
                                                if debug_mode:
                                                    print(f"    [跨页检测] 在下一页找到下一个子题 ({next_expected_number})，标记跨页合并")
                                                break
                                        else:
                                            # 不是嵌套子题，如果还没记录early_end_y，记录（使用当前页的最后位置作为结束）
                                            if early_end_y is None:
                                                # 注意：跨页时，early_end_y应该使用当前页的最后位置
                                                # 这里先不设置early_end_y，因为它在下一页
                                                # 我们会在边界计算时处理跨页情况
                                                if debug_mode:
                                                    print(f"    [跨页检测] 在下一页遇到非嵌套子题文本: {next_line_text[:30]}")
                                    else:
                                        # 循环正常结束，说明下一页的前几行都没有找到下一个子题
                                        if debug_mode:
                                            print(f"    [跨页检测] 下一页前{remaining_lines_to_check}行未找到下一个子题")
                        
                        # 如果3行内没有找到下一个子题（包括跨页检查），提前确定结束位置
                        # 注意：提前结束位置必须大于题目的起始位置才有效
                        # 如果是跨页情况，需要特殊处理
                        if not found_next_nested:
                            # 如果是跨页情况，early_end_y可能为None（因为下一个文本在下一页）
                            # 此时应该标记跨页，但不确定结束位置（在边界计算时处理）
                            if current_question.get('cross_page'):
                                if debug_mode:
                                    print(f"    [跨页检测] 题目跨页，将在边界计算时确定结束位置")
                            elif early_end_y is not None:
                                question_start_y = current_question.get('start_y', 0)
                                
                                # 验证提前结束位置的有效性：必须大于起始位置
                                if early_end_y > question_start_y:
                                    existing_early_end = current_question.get('early_end_y')
                                    # 如果已经有提前结束位置，只有当新的位置更早（更小）且仍然大于起始位置时才更新
                                    if existing_early_end is None:
                                        current_question['early_end_y'] = early_end_y
                                        current_question['early_end_text'] = early_end_text
                                        if debug_mode:
                                            nested_list = current_question.get('nested_sub_numbers', [])
                                            nested_info = f"已累积子题: {nested_list}, " if nested_list else ""
                                            print(f"    [嵌套小题] 3行内未找到下一个子题({next_expected_number})，提前结束")
                                            print(f"      {nested_info}结束位置: y={early_end_y}, 文本={early_end_text}")
                                    elif early_end_y < existing_early_end and early_end_y > question_start_y:
                                        # 新的结束位置更靠前，且仍然有效
                                        current_question['early_end_y'] = early_end_y
                                        current_question['early_end_text'] = early_end_text
                                        if debug_mode:
                                            nested_list = current_question.get('nested_sub_numbers', [])
                                            nested_info = f"已累积子题: {nested_list}, " if nested_list else ""
                                            print(f"    [嵌套小题] 更新提前结束位置: y={early_end_y}, 文本={early_end_text}")
                                else:
                                    # 提前结束位置无效，不设置
                                    if debug_mode:
                                        print(f"    [嵌套小题] 提前结束位置无效 (y={early_end_y} <= start_y={question_start_y})，忽略")
                            # 继续累积当前嵌套子题，但标记了提前结束位置
                        
                    # 继续累积到当前题目中
                    if debug_mode:
                        question_type = current_question.get('question_type', 'unknown')
                        nested_info = f", 已累积子题: {nested_sub_numbers}" if nested_sub_numbers else ""
                        print(f"    [嵌套小题] 继续累积到题目（类型={question_type}{nested_info}）: {line_text[:30]}")
                    # 不创建新题目，继续累积
                    continue
                else:
                    # 没有当前题目，嵌套子题不应该独立存在
                    # 作为兜底，也继续累积（虽然这不应该发生）
                    if debug_mode:
                        print(f"    [嵌套小题] 警告：没有当前题目，跳过: {line_text[:30]}")
                    continue
            
            # 检查其他类型的题目模式
            elif self._matches_pattern(line_text, 'question'):
                # 如果当前题目已经有提前结束位置，检查新题目是否在提前结束位置之前
                # 如果在之前，说明提前结束位置设置错误，需要调整
                if current_question and current_question.get('early_end_y') is not None:
                    y_pos = int(line_y_min)
                    # 确保 early_end_y 是整数类型
                    early_end_y = int(current_question['early_end_y'])
                    
                    # 如果新题目在提前结束位置之前，说明提前结束位置设置有问题
                    # 应该使用新题目的位置作为结束位置
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
                    'text_height': text_height,  # 保存文字高度
                    'nested_sub_numbers': []  # 初始化嵌套子题编号列表
                }
                is_chinese_question = False
                print(f"    [发现] 发现题目 (行{len(questions)+1}): {line_text[:30]} (y={y_pos}, h={text_height})")
        
        # 保存最后一题
        if current_question:
            questions.append(current_question)
        
        return questions
    
    def _extract_zones(self, ocr_result: List, page_size: Tuple[int, int], page_num: int) -> List[Dict]:
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
        if self.config and self.config.mode == 'debug':
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
            if self._matches_pattern(line_text, 'question'):
                # 检查是否是中文数字大题
                chinese_num_pattern = self.patterns.get('chinese_number')
                is_current_chinese = chinese_num_pattern and chinese_num_pattern.match(line_text.strip())
                
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
                elif self._is_sub_question(line_text):
                    # 如果当前在中文大题下，不创建新的独立题目，继续累积
                    if is_chinese_question and current_question:
                        # 继续累积到当前大题中，更新高度
                        # 更新高度到当前行的位置
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
                # 需要区分是中文大题还是阿拉伯小题
                chinese_num_pattern = self.patterns.get('chinese_number')
                debug_mode = self.config and self.config.mode == 'debug'
                is_current_chinese = chinese_num_pattern and chinese_num_pattern.match(line_text.strip())
                is_current_sub = self._is_sub_question(line_text)
                
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
                elif any(pattern.match(line_text.strip()) for pattern in self.patterns.get('question', [])):
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
            elif self._matches_pattern(line_text, 'answer'):
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
        zones = self._merge_adjacent_zones(zones)
        
        return zones
    
    def _matches_pattern(self, text: str, pattern_type: str) -> bool:
        """
        检查文本是否匹配指定类型的模式
        
        Args:
            text: 待检查文本
            pattern_type: 模式类型 ('question' 或 'answer')
            
        Returns:
            是否匹配
        """
        if not self.config or pattern_type not in self.patterns:
            return False
        
        patterns = self.patterns[pattern_type]
        text_stripped = text.strip()
        
        for pattern in patterns:
            if pattern.match(text_stripped):
                # 调试模式下显示匹配的详细信息
                if self.config and self.config.mode == 'debug':
                    print(f"    [模式匹配] '{text_stripped}' 匹配规则: {pattern.pattern}")
                return True
        
        return False
    
    def _is_sub_question(self, text: str) -> bool:
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
    
    def _extract_nested_sub_number(self, text: str) -> int:
        """
        从嵌套子题文本中提取数字编号
        
        Args:
            text: 嵌套子题文本，如 "(1)", "（2）", "[3]"
            
        Returns:
            数字编号，如果无法提取则返回None
        """
        if not text:
            return None
        
        import re
        # 匹配括号或方括号中的数字
        match = re.search(r'[（(）)\[\]]([0-9]+)[）)\]）]', text.strip())
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return None
        return None
    
    def _is_nested_sub_question(self, text: str) -> bool:
        """
        检查文本是否是嵌套小题（(1)、【1】、(1)、[1]等）
        
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
    
    def _merge_adjacent_zones(self, zones: List[Dict]) -> List[Dict]:
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

