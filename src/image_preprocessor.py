#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像预处理模块
用于优化OCR识别的坐标精度：裁剪空白区域、统一缩放、坐标校准
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Tuple, Optional


def preprocess_image(img_array: np.ndarray, target_height: int = 1280, auto_crop: bool = True) -> Tuple[np.ndarray, Dict, bool]:
    """
    图像预处理：裁剪空白区域、统一缩放尺寸
    
    Args:
        img_array: numpy数组格式的图像 (RGB)
        target_height: 目标高度（像素），默认1280
        auto_crop: 是否自动裁剪空白区域，默认True
    
    Returns:
        processed_img: 处理后的图像 (RGB格式的numpy数组)
        crop_info: 裁剪信息字典 {'x': x偏移, 'y': y偏移, 'scale': 缩放比例}
        is_blank_page: 是否检测到空白页（True表示空白页，应跳过OCR）
    """
    # 转换为BGR格式（OpenCV格式）
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_bgr.shape[:2]
    
    crop_info = {'x': 0, 'y': 0, 'scale': 1.0, 'bottom_crop': 0}
    
    # 步骤1: 自动裁剪空白区域
    # 底部固定裁剪高度：图片总高度的8.75%（0.0875）
    bottom_crop = int(orig_h * 0.0875)
    
    if auto_crop:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 使用阈值检测文本区域
        _, th = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # 找到非零像素的边界框
        coords = cv2.findNonZero(th)
        if coords is not None and len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords)
            
            # 添加边距（避免裁剪过紧）
            margin = 20
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(orig_w - x, w + margin * 2)
            
            # 底部固定裁剪：图片总高度的8.75%
            # 裁剪区域从 y 开始，到 (orig_h - bottom_crop) 结束
            # 所以最大可用高度 = orig_h - y - bottom_crop
            max_h = orig_h - y - bottom_crop
            h = min(max_h, h + margin * 2)
            h = max(h, 0)  # 确保高度不为负
            
            # 如果高度为0或负数，可能是空白页，使用原图
            if h <= 0:
                cropped = img_bgr
                crop_info['x'] = 0
                crop_info['y'] = 0
                crop_info['bottom_crop'] = 0
            else:
                # 实际裁剪区域：[y : y+h, x : x+w]
                # 底部边界 = y + h，应该 ≤ orig_h - bottom_crop
                cropped = img_bgr[y:y+h, x:x+w]
                crop_info['x'] = x
                crop_info['y'] = y
                crop_info['bottom_crop'] = bottom_crop  # 记录底部裁剪量，用于调试和验证
        else:
            # 未检测到可裁剪区域，但从底部固定裁剪
            if orig_h > bottom_crop:
                h = orig_h - bottom_crop
                # 确保高度有效
                if h > 0:
                    cropped = img_bgr[0:h, :]
                    crop_info['x'] = 0
                    crop_info['y'] = 0
                    crop_info['bottom_crop'] = bottom_crop
                else:
                    # 高度无效，可能是空白页，使用原图
                    cropped = img_bgr
                    crop_info['bottom_crop'] = 0
            else:
                # 图像高度小于底部裁剪量，可能是很小的空白页
                cropped = img_bgr
                crop_info['bottom_crop'] = 0
    else:
        # 即使不自动裁剪，也从底部固定裁剪
        if orig_h > bottom_crop:
            h = orig_h - bottom_crop
            # 确保高度有效
            if h > 0:
                cropped = img_bgr[0:h, :]
                crop_info['x'] = 0
                crop_info['y'] = 0
                crop_info['bottom_crop'] = bottom_crop
            else:
                # 高度无效，使用原图
                cropped = img_bgr
                crop_info['bottom_crop'] = 0
        else:
            cropped = img_bgr
            crop_info['bottom_crop'] = 0
    
    # 步骤2: 统一缩放尺寸（防止推理失真）
    # 检查裁剪后的图像是否为空（空白页处理）
    if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        # 空白页或裁剪后为空，标记为空白页
        print(f"  [空白页] 检测到空白页或裁剪后图像为空，将跳过OCR识别")
        processed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop_info['x'] = 0
        crop_info['y'] = 0
        crop_info['scale'] = 1.0
        crop_info['bottom_crop'] = 0
        return processed_img, crop_info, True  # 返回True表示空白页
    
    new_h, new_w = cropped.shape[:2]
    
    # 再次检查尺寸是否有效
    if new_h <= 0 or new_w <= 0:
        print(f"  [空白页] 裁剪后图像尺寸无效 ({new_w}x{new_h})，将跳过OCR识别")
        processed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop_info['x'] = 0
        crop_info['y'] = 0
        crop_info['scale'] = 1.0
        crop_info['bottom_crop'] = 0
        return processed_img, crop_info, True  # 返回True表示空白页
    
    # 计算缩放比例（保持长宽比）
    scale = target_height / new_h if new_h > 0 else 1.0
    
    # 如果缩放后宽度太大，以宽度为准
    new_width = int(new_w * scale)
    max_width = 2048
    if new_width > max_width:
        scale = max_width / new_w
        target_height = int(new_h * scale)
    
    # 确保缩放后的尺寸有效
    if target_height <= 0:
        target_height = new_h
    if new_width <= 0:
        new_width = new_w
    
    try:
        resized = cv2.resize(cropped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    except cv2.error as e:
        # 如果resize失败，可能是空白页，标记为空白页
        print(f"  [空白页] 图像缩放失败 ({e})，将跳过OCR识别")
        processed_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        crop_info['x'] = 0
        crop_info['y'] = 0
        crop_info['scale'] = 1.0
        crop_info['bottom_crop'] = 0
        return processed_img, crop_info, True  # 返回True表示空白页
    
    crop_info['scale'] = scale
    
    # 转换回RGB格式
    processed_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return processed_img, crop_info, False  # 返回False表示非空白页


def correct_coordinates(ocr_result: list, crop_info: Dict, original_shape: Tuple[int, int]) -> list:
    """
    将OCR坐标映射回原图坐标
    
    Args:
        ocr_result: OCR识别结果列表，格式: [[坐标框, (文本, 置信度)], ...]
        crop_info: 裁剪信息字典 {'x': x偏移, 'y': y偏移, 'scale': 缩放比例}
        original_shape: 原图尺寸 (height, width)
    
    Returns:
        校正后的OCR结果列表（坐标已映射回原图）
    """
    corrected_result = []
    
    x_offset = crop_info['x']
    y_offset = crop_info['y']
    scale = crop_info['scale']
    
    # 如果图像被裁剪和缩放了，需要逆变换
    # 坐标需要：1. 除以缩放比例 2. 加上裁剪偏移
    
    for line_data in ocr_result:
        if not line_data:
            continue
        
        box = line_data[0]
        text_info = line_data[1]
        
        # 校正坐标框
        corrected_box = []
        for point in box:
            # 先除以缩放比例
            orig_x = point[0] / scale
            orig_y = point[1] / scale
            
            # 加上裁剪偏移
            orig_x += x_offset
            orig_y += y_offset
            
            # 确保坐标在图像范围内
            orig_x = max(0, min(orig_x, original_shape[1]))
            orig_y = max(0, min(orig_y, original_shape[0]))
            
            corrected_box.append([orig_x, orig_y])
        
        corrected_result.append([corrected_box, text_info])
    
    return corrected_result

