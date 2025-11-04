#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR结果解析模块
负责解析不同OCR引擎的返回结果并转换为统一格式
"""

from typing import List, Dict, Any


class OCRResultParser:
    """OCR结果解析器"""
    
    def __init__(self, config=None):
        """
        初始化解析器
        
        Args:
            config: 配置对象
        """
        self.config = config
    
    def parse_ppstructure_result(self, ppstructure_response) -> List:
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
                if hasattr(overall_ocr_res, 'rec_texts') or hasattr(overall_ocr_res, 'rec_polys') or isinstance(overall_ocr_res, dict):
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
        
        # 尝试获取文本、坐标和置信度
        try:
            # 新版本PPStructure的OCRResult对象可能有以下属性：
            # - rec_texts: 文本列表
            # - rec_polys: 坐标列表（多边形）
            # - rec_scores: 置信度列表
            
            # 方法1: 尝试属性访问
            if hasattr(ocr_result_obj, 'rec_texts'):
                texts = ocr_result_obj.rec_texts if not isinstance(ocr_result_obj.rec_texts, dict) else ocr_result_obj.rec_texts.get('texts', [])
            elif isinstance(ocr_result_obj, dict):
                texts = ocr_result_obj.get('rec_texts', [])
            else:
                texts = []
            
            if hasattr(ocr_result_obj, 'rec_polys'):
                polys = ocr_result_obj.rec_polys if not isinstance(ocr_result_obj.rec_polys, dict) else ocr_result_obj.rec_polys.get('polys', [])
            elif isinstance(ocr_result_obj, dict):
                polys = ocr_result_obj.get('rec_polys', [])
            else:
                polys = []
            
            if hasattr(ocr_result_obj, 'rec_scores'):
                scores = ocr_result_obj.rec_scores if not isinstance(ocr_result_obj.rec_scores, dict) else ocr_result_obj.rec_scores.get('scores', [])
            elif isinstance(ocr_result_obj, dict):
                scores = ocr_result_obj.get('rec_scores', [])
            else:
                scores = []
            
            # 如果texts是空列表，尝试其他方法
            if not texts:
                # 方法2: 尝试作为列表访问（旧版本格式）
                if isinstance(ocr_result_obj, list):
                    return ocr_result_obj
                # 方法3: 尝试字典访问
                elif isinstance(ocr_result_obj, dict):
                    # 可能是一个包含多个结果的字典
                    if 'texts' in ocr_result_obj:
                        texts = ocr_result_obj['texts']
                    if 'polys' in ocr_result_obj:
                        polys = ocr_result_obj['polys']
                    if 'scores' in ocr_result_obj:
                        scores = ocr_result_obj['scores']
            
            # 组合结果
            max_len = max(len(texts), len(polys))
            for i in range(max_len):
                text = texts[i] if i < len(texts) else ""
                poly = polys[i] if i < len(polys) else []
                score = scores[i] if i < len(scores) else 1.0
                
                # 确保poly是列表格式
                if hasattr(poly, 'tolist'):
                    box = poly.tolist()
                elif isinstance(poly, (list, tuple)):
                    box = list(poly)
                else:
                    continue  # 跳过无效的poly格式
                
                result_list.append([box, (text, float(score))])
            
            if debug_mode:
                print(f"  调试: 转换后列表长度={len(result_list)}")
            
        except Exception as e:
            if debug_mode:
                print(f"  调试: 转换OCRResult对象时出错: {e}")
                import traceback
                traceback.print_exc()
            # 如果转换失败，返回空列表
            return []
        
        return result_list

