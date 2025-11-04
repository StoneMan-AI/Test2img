#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR文本合并模块
负责合并同一行的文本片段，计算文本边界框
"""

from typing import List, Dict
import numpy as np


class OCRTextMerger:
    """OCR文本合并器"""
    
    def __init__(self, config=None):
        """
        初始化文本合并器
        
        Args:
            config: 配置对象
        """
        self.config = config
        # 行合并的Y坐标阈值（同一行的Y坐标差小于此值时认为是同一行）
        self.line_y_threshold = 10.0  # 可以根据需要调整
    
    def merge_same_line_texts(self, ocr_result: List) -> List:
        """
        合并同一行的文本片段
        
        Args:
            ocr_result: OCR识别结果列表，格式: [[坐标框, (文本, 置信度)], ...]
        
        Returns:
            合并后的OCR结果列表
        """
        if not ocr_result:
            return []
        
        # 按Y坐标排序（从上到下）
        sorted_result = sorted(ocr_result, key=lambda x: min([point[1] for point in x[0]]))
        
        merged_result = []
        current_line_texts = []
        
        for line_data in sorted_result:
            if not line_data or len(line_data) < 2:
                continue
            
            line_box = line_data[0]
            line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
            confidence = line_data[1][1] if len(line_data[1]) > 1 else 0
            
            # 计算当前行的Y坐标中心
            y_coords = [point[1] for point in line_box]
            y_center = np.mean(y_coords)
            
            if not current_line_texts:
                # 第一行，直接添加
                current_line_texts.append({
                    'text': line_text,
                    'confidence': confidence,
                    'box': line_box,
                    'y_center': y_center
                })
            else:
                # 检查是否与当前行在同一行
                last_y_center = current_line_texts[-1]['y_center']
                if abs(y_center - last_y_center) <= self.line_y_threshold:
                    # 同一行，添加到当前行
                    current_line_texts.append({
                        'text': line_text,
                        'confidence': confidence,
                        'box': line_box,
                        'y_center': y_center
                    })
                else:
                    # 新的一行，先合并上一行
                    merged_line = self._merge_line_texts(current_line_texts)
                    merged_result.append(merged_line)
                    # 开始新的一行
                    current_line_texts = [{
                        'text': line_text,
                        'confidence': confidence,
                        'box': line_box,
                        'y_center': y_center
                    }]
        
        # 处理最后一行
        if current_line_texts:
            merged_line = self._merge_line_texts(current_line_texts)
            merged_result.append(merged_line)
        
        return merged_result
    
    def _is_same_line(self, info1: Dict, info2: Dict, threshold: float = None) -> bool:
        """
        判断两个文本信息是否在同一行
        
        Args:
            info1: 第一个文本信息（包含'x_min', 'x_max', 'y_min', 'y_max'）
            info2: 第二个文本信息
            threshold: Y坐标阈值，默认使用self.line_y_threshold
        
        Returns:
            是否在同一行
        """
        if threshold is None:
            threshold = self.line_y_threshold
        
        y1_center = (info1['y_min'] + info1['y_max']) / 2
        y2_center = (info2['y_min'] + info2['y_max']) / 2
        
        return abs(y1_center - y2_center) <= threshold
    
    def _merge_line_texts(self, line_texts: List[Dict]) -> List:
        """
        合并同一行的多个文本片段
        
        Args:
            line_texts: 同一行的文本信息列表
        
        Returns:
            合并后的OCR结果: [坐标框, (文本, 置信度)]
        """
        if not line_texts:
            return None
        
        if len(line_texts) == 1:
            # 只有一个文本，直接返回
            text_info = line_texts[0]
            return [text_info['box'], (text_info['text'], text_info['confidence'])]
        
        # 按X坐标排序（从左到右）
        sorted_texts = sorted(line_texts, key=lambda x: min([point[0] for point in x['box']]))
        
        # 合并文本
        merged_text = "".join([info['text'] for info in sorted_texts])
        
        # 计算平均置信度
        avg_conf = np.mean([info['confidence'] for info in sorted_texts])
        
        # 计算合并后的边界框（取所有框的最小/最大值）
        all_points = []
        for info in sorted_texts:
            all_points.extend(info['box'])
        
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]
        
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        
        # 构建合并后的边界框 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        merged_bbox = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max]
        ]
        
        return [merged_bbox, (merged_text, avg_conf)]

