#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片裁剪模块
根据识别出的区域坐标裁剪图片并保存
"""

from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import os


class ImageCropper:
    """图片裁剪类"""
    
    def __init__(self, config=None):
        """
        初始化图片裁剪器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 确保输出目录存在
        if self.config and self.config.output_dir:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def crop_and_save(self, questions: List[Dict], base_filename: str) -> Dict:
        """
        裁剪并保存图片
        
        Args:
            questions: 题目列表（已包含边界信息）
            base_filename: 基础文件名（不含扩展名）
            
        Returns:
            保存结果统计信息
        """
        output_dir = self.config.output_dir if self.config else Path("./output")
        
        # 计数器
        question_count = 0
        
        print("\n--- 开始裁剪并保存 ---")
        
        for i, question in enumerate(questions, 1):
            try:
                # 裁剪题目
                cropped = self._crop_question(question)
                
                if cropped:
                    question_count += 1
                    
                    # 生成文件名
                    if question.get('cross_page_merged'):
                        # 跨页合并题目使用multipage文件名
                        filename = output_dir / f"{base_filename}_Q{question_count}_multipage.png"
                    elif question.get('potential_cross_page'):
                        filename = output_dir / f"{base_filename}_Q{question_count}_multipage.png"
                    else:
                        page_num = question.get('page', 1)
                        filename = output_dir / f"{base_filename}_Q{question_count}_p{page_num}.png"
                    
                    # 保存图片
                    cropped.save(str(filename))
                    print(f"    [成功] 保存: {filename.name}")
                else:
                    print(f"    警告: 题目{i}裁剪失败")
                    
            except Exception as e:
                print(f"    错误: 处理题目{i}时出错: {e}")
                continue
        
        return {
            'total_saved': question_count,
            'questions': question_count,
            'answers': 0,
            'output_dir': str(output_dir)
        }
    
    def _crop_question(self, question: Dict) -> Image.Image:
        """
        裁剪单个题目区域
        
        Args:
            question: 题目信息
            
        Returns:
            裁剪后的图片
        """
        # 获取基本信息
        original_start_y = question['start_y']
        original_end_y = question['end_y']
        start_x = question.get('start_x', 0)
        page_num = question['page']
        image = question['page_image']
        image_width = image.width
        image_height = question.get('image_height', image.height)
        
        # 使用新的裁剪规则计算起始位置
        # 裁剪开始位置 = line_y_min - text_height * 0.5
        text_height = question.get('text_height', 0)
        debug_mode = self.config and self.config.mode == 'debug'
        
        if text_height > 0:
            start_y_offset = int(text_height * 0.5)
            start_y = original_start_y - start_y_offset
            if debug_mode:
                print(f"      裁剪计算: 题目原y={original_start_y}, 高度={text_height}, 偏移={start_y_offset}, 最终y={start_y}")
        else:
            start_y = original_start_y
        
        # end_y 已经在 layout_analyzer 中按照新规则计算了，直接使用
        end_y = original_end_y
        
        if debug_mode:
            print(f"      裁剪区域: y1={start_y}, y2={end_y}, 高度={end_y-start_y}")
        
        # 应用水平边距
        margin_h = self.config.horizontal_margin if self.config else 5
        
        # 确定裁剪区域
        x1 = max(0, start_x - margin_h)
        y1 = max(0, start_y)
        x2 = image_width  # 使用完整宽度
        y2 = min(image_height, end_y)  # 到结束位置，但不超过页面高度
        
        # 确保y2不超过页面高度
        if y2 > image_height:
            y2 = image_height
        
        # 检查是否是跨页合并的题目
        if question.get('cross_page_merged'):
            # 跨页合并题目：需要合并多页
            return self._crop_cross_page_merged(question)
        # 检查是否在同页内
        elif end_y <= image_height:
            # 同页: 直接裁剪
            crop_box = (x1, y1, x2, y2)
            
            # 验证有效性（只检查坐标顺序和边界，不做尺寸限制）
            if x1 >= x2 or y1 >= y2:
                print(f"      无效裁剪框: {crop_box}")
                return None
            
            if y2 > image_height:
                print(f"      超出页面高度: y2={y2} > {image_height}")
                y2 = image_height
                crop_box = (x1, y1, x2, y2)
            
            try:
                return image.crop(crop_box)
            except Exception as e:
                print(f"      裁剪失败: {e}, Box: {crop_box}")
                return None
        else:
            # 跨页: 需要合并多页
            return self._crop_cross_page(question)
    
    def _crop_cross_page_merged(self, question: Dict) -> Image.Image:
        """
        处理已完成跨页合并的题目裁剪
        
        Args:
            question: 题目信息（已完成跨页合并）
            
        Returns:
            合并后的图片
        """
        # 第一页部分（起始页）
        original_start_y = question['start_y']
        page1 = question['page_image']
        start_page_num = question['page']
        
        # 使用新的裁剪规则计算起始位置
        text_height = question.get('text_height', 0)
        if text_height > 0:
            start_y_offset = int(text_height * 0.5)
            start_y = original_start_y - start_y_offset
        else:
            start_y = original_start_y
        
        # 应用水平边距
        margin_h = self.config.horizontal_margin if self.config else 5
        
        # 裁剪第一页（从题目开始到页尾）
        crop1 = page1.crop((max(0, 0 - margin_h), max(0, start_y), page1.width, page1.height))
        
        # 获取跨页结束页
        pages_data = question.get('pages_data', [])
        cross_page_end_page = question.get('cross_page_end_page', start_page_num + 1)
        
        if cross_page_end_page > start_page_num and cross_page_end_page <= len(pages_data):
            # 有跨页结束页，裁剪结束页
            page2 = pages_data[cross_page_end_page - 1]
            cross_page_end_y = question.get('cross_page_end_y')
            
            if cross_page_end_y is not None:
                # 使用记录的跨页结束位置
                end_y = cross_page_end_y
            else:
                # 如果没有记录结束位置，使用页尾
                end_y = page2.height
            
            # 裁剪第二页（从页首到结束位置）
            crop2 = page2.crop((max(0, 0 - margin_h), 0, page2.width, min(page2.height, end_y)))
            
            # 合并两张图片
            merged = self._merge_images_vertically(crop1, crop2)
            
            debug_mode = self.config and self.config.mode == 'debug'
            if debug_mode:
                print(f"      [跨页合并裁剪] 第{start_page_num}页: {start_y}→{page1.height}, 第{cross_page_end_page}页: 0→{end_y}")
            
            return merged
        else:
            # 没有跨页结束页，只返回第一页
            return crop1
    
    def _crop_cross_page(self, question: Dict) -> Image.Image:
        """
        处理跨页题目的裁剪
        
        Args:
            question: 题目信息
            
        Returns:
            合并后的图片
        """
        # 第一页部分
        original_start_y = question['start_y']
        page1 = question['page_image']
        
        # 使用新的裁剪规则计算起始位置
        text_height = question.get('text_height', 0)
        if text_height > 0:
            start_y_offset = int(text_height * 0.5)
            start_y = original_start_y - start_y_offset
        else:
            start_y = original_start_y
        
        # 裁剪第一页（从题目开始到页尾，不添加边距）
        margin_h = self.config.horizontal_margin if self.config else 5
        
        crop1 = page1.crop((0, max(0, start_y), page1.width, page1.height))
        
        # 检查是否有下一页
        pages_data = question.get('pages_data', [])
        page_num = question['page']
        
        if page_num < len(pages_data):
            # 有下一页，裁剪下一页
            page2 = pages_data[page_num]
            next_question_y = question.get('next_question_y')
            
            if next_question_y:
                # 有明确的下一题位置，使用 end_y（已经在 layout_analyzer 中按新规则计算）
                end_y = question.get('end_y', next_question_y)
            else:
                # 到页尾
                end_y = page2.height
            
            crop2 = page2.crop((0, 0, page2.width, min(page2.height, end_y)))
            
            # 合并两张图片
            merged = self._merge_images_vertically(crop1, crop2)
            return merged
        else:
            # 没有下一页，只返回第一页
            return crop1
    
    def _merge_images_vertically(self, img1: Image.Image, img2: Image.Image) -> Image.Image:
        """
        垂直拼接两张图片
        
        Args:
            img1: 上方图片
            img2: 下方图片
            
        Returns:
            合并后的图片
        """
        width = max(img1.width, img2.width)
        height = img1.height + img2.height
        
        merged = Image.new('RGB', (width, height), color='white')
        merged.paste(img1, (0, 0))
        merged.paste(img2, (0, img1.height))
        
        return merged
    
    def _apply_margin(self, zone: Dict, image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        应用边距到裁剪框
        
        Args:
            zone: 区域字典，包含x, y, width, height
            image_size: 图片尺寸 (width, height)
            
        Returns:
            裁剪框坐标 (left, top, right, bottom)
        """
        margin_v = self.config.vertical_margin if self.config else 10
        margin_h = self.config.horizontal_margin if self.config else 5
        
        x1 = max(0, zone['x'] - margin_h)
        y1 = max(0, zone['y'] - margin_v)
        x2 = min(image_size[0], zone['x'] + zone['width'] + margin_h)
        y2 = min(image_size[1], zone['y'] + zone['height'] + margin_v)
        
        return (x1, y1, x2, y2)
    
    def _is_valid_box(self, box: Tuple[int, int, int, int], image_size: Tuple[int, int], zone: Dict) -> bool:
        """
        验证裁剪框是否有效
        
        Args:
            box: 裁剪框 (left, top, right, bottom)
            image_size: 图片尺寸
            zone: 区域信息（用于调试）
            
        Returns:
            是否有效
        """
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        
        # 检查坐标顺序
        if left >= right or top >= bottom:
            print(f"    警告: 跳过无效区域 {zone.get('type', 'unknown')} - 坐标顺序错误: {box}")
            return False
        
        # 检查是否在图片范围内
        if left < 0 or top < 0 or right > image_size[0] or bottom > image_size[1]:
            print(f"    警告: 跳过无效区域 {zone.get('type', 'unknown')} - 超出图片范围: {box}, 图片尺寸: {image_size}")
            return False
        
        # 检查最小尺寸
        min_height = self.config.min_zone_height if self.config else 50
        min_width = self.config.min_zone_width if self.config else 200
        
        if width < min_width:
            print(f"    警告: 跳过无效区域 {zone.get('type', 'unknown')} - 宽度不足: {width} < {min_width}, Box: {box}")
            return False
        
        if height < min_height:
            print(f"    警告: 跳过无效区域 {zone.get('type', 'unknown')} - 高度不足: {height} < {min_height}, Box: {box}")
            return False
        
        return True
