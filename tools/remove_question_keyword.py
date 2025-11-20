#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立工具：删除题目编码关键词
使用黑色正方形遮挡关键词（"一、"、"1."等），包含数字和标点
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import re
from typing import Optional, Tuple, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.ocr_engine import OCREngine
from src.ocr_result_parser import OCRResultParser


def extract_keyword(text: str) -> Optional[Tuple[str, int]]:
    """
    从文本中提取题目编码关键词（包含数字和标点）
    
    Args:
        text: 完整文本
    
    Returns:
        (关键词完整文本, 关键词长度) 或 None
        例如: ("一、", 2) 表示"一、"
              ("1.", 2) 表示"1."
    """
    if not text:
        return None
    
    text_stripped = text.strip()
    
    # 匹配中文数字+顿号：一、二、三、...
    chinese_match = re.match(r'^([一二三四五六七八九十]+[、])', text_stripped)
    if chinese_match:
        keyword = chinese_match.group(1)
        return (keyword, len(keyword))
    
    # 匹配阿拉伯数字+点号/顿号：1. 2. 10. 或 1、2、10、
    # 注意：需要同时匹配半角点号"."和全角点号"．"（U+FF0E）
    # 允许数字后面有空格，如 "9. " 或 "9 "
    # 匹配半角点号或全角点号
    arabic_match = re.match(r'^(\d+[\.．、]\s*)', text_stripped)
    if arabic_match:
        keyword = arabic_match.group(1).rstrip()  # 去掉末尾空格
        return (keyword, len(keyword))
    
    # 如果上面没匹配到，尝试更宽松的匹配：数字后面可能有空格再跟点号（半角或全角）
    arabic_match2 = re.match(r'^(\d+)\s*([\.．、])', text_stripped)
    if arabic_match2:
        number = arabic_match2.group(1)
        punct = arabic_match2.group(2)
        keyword = number + punct
        return (keyword, len(keyword))
    
    return None


def detect_keyword_position(ocr_result: List, image_width: int) -> Optional[Tuple[int, int, int, int]]:
    """
    检测OCR结果中第一行的题目编码关键词位置
    
    Args:
        ocr_result: OCR识别结果列表
        image_width: 图片宽度（用于计算x坐标）
    
    Returns:
        关键词边界框 (x_min, y_min, x_max, y_max) 或 None
    """
    first_keyword = None
    min_y = float('inf')
    
    for line_data in ocr_result:
        if not line_data:
            continue
        
        # 解析OCR结果格式: [box, (text, confidence)]
        if not isinstance(line_data, list) or len(line_data) < 2:
            continue
        
        box = line_data[0]
        text_info = line_data[1]
        text = text_info[0] if isinstance(text_info, (list, tuple)) and len(text_info) > 0 else ""
        
        # 计算整行的边界框（先计算，用于调试判断）
        if not isinstance(box, list) or len(box) == 0:
            continue
        
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        line_x_min = min(x_coords)
        line_x_max = max(x_coords)
        line_y_min = min(y_coords)
        line_y_max = max(y_coords)
        line_width = line_x_max - line_x_min
        
        # 提取关键词（包含数字和标点）
        keyword_info = extract_keyword(text)
        if not keyword_info:
            # 调试：如果是最上面的行，打印为什么没匹配
            if line_y_min < min_y + 10:  # 只对最上面的行进行调试
                print(f"    调试：文本 '{text[:30]}' 未匹配到关键词")
            continue
        
        keyword_text, keyword_length = keyword_info
        
        text_length = len(text)
        
        # 判断关键词类型：数字关键词（如"1."）还是中文数字关键词（如"一、"）
        is_arabic_keyword = keyword_text[0].isdigit() if keyword_text else False
        
        if is_arabic_keyword:
            # 数字关键词（如"1."、"2."）：只使用数字部分（不带"."），以数字的结束位置作为遮挡的结束坐标
            # 提取数字部分（去掉标点）
            import re
            number_match = re.match(r'^(\d+)', keyword_text)
            if number_match:
                number_text = number_match.group(1)
                number_length = len(number_text)
                # 计算数字部分的宽度（根据文本比例）
                number_ratio = number_length / text_length if text_length > 0 else 0
                number_width = int(line_width * number_ratio)
                
                # 关键词的起始x坐标（从行首开始）
                keyword_x_min = line_x_min
                # 结束x坐标：数字部分的结束位置
                keyword_x_max = line_x_min + number_width
            else:
                # 如果无法提取数字，使用默认方式
                keyword_ratio = keyword_length / text_length if text_length > 0 else 0
                keyword_width = int(line_width * keyword_ratio)
                keyword_x_min = line_x_min
                keyword_x_max = line_x_min + keyword_width
        else:
            # 中文数字关键词（如"一、"、"二、"）：使用关键词结束位置再向左平移10px
            # 计算关键词的宽度（根据文本比例）
            keyword_ratio = keyword_length / text_length if text_length > 0 else 0
            keyword_width = int(line_width * keyword_ratio)
            
            # 计算关键词的起始x坐标（从行首开始）
            keyword_x_min = line_x_min
            # 结束x坐标：关键词结束位置再向左平移10px，避免遮挡题目内容
            keyword_x_max = line_x_min + keyword_width - 10
        
        # 确保x_max不小于x_min
        if keyword_x_max <= keyword_x_min:
            keyword_x_max = keyword_x_min + 5  # 至少保留5px宽度
        
        # 关键词的y坐标使用整行的y坐标
        keyword_y_min = line_y_min
        keyword_y_max = line_y_max
        
        if line_y_min < min_y:
            min_y = line_y_min
            first_keyword = (keyword_x_min, keyword_y_min, keyword_x_max, keyword_y_max)
    
    return first_keyword


def remove_question_keyword(image_path: str, output_path: str = None, config: Config = None, ocr_engine: OCREngine = None) -> bool:
    """
    删除图片第一行的题目编码关键词（使用黑色正方形遮挡）
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径（如果为None，则覆盖原文件）
        config: 配置对象（如果为None则创建默认配置）
        ocr_engine: OCR引擎（如果为None则创建新引擎，用于批量处理时复用引擎）
    
    Returns:
        是否成功处理
    """
    if not Path(image_path).exists():
        print(f"错误: 图片不存在: {image_path}")
        return False
    
    if config is None:
        config = Config()
    
    # 使用传入的OCR引擎，或创建新引擎
    print(f"处理图片: {image_path}")
    if ocr_engine is None:
        print("  初始化OCR引擎（PPStructure）...")
        ocr_engine = OCREngine(config=config)
        if not ocr_engine.use_ppstructure:
            print("  警告: PPStructure不可用，将使用标准PaddleOCR")
    else:
        print("  使用已初始化的OCR引擎")
    
    # 读取图片
    image = Image.open(image_path)
    image_array = np.array(image.convert('RGB'))
    image_width = image.width
    image_height = image.height
    
    # 使用OCR引擎识别
    print("  执行OCR识别...")
    try:
        ocr_response = ocr_engine.predict(image_array)
    except Exception as e:
        print(f"  错误: OCR识别失败: {e}")
        return False
    
    # 解析OCR结果
    parser = OCRResultParser(config)
    
    if ocr_engine.use_ppstructure:
        ocr_result = parser.parse_ppstructure_result(ocr_response)
    else:
        first_result = ocr_response[0] if ocr_response else None
        if first_result is None:
            print("  错误: 未获取到OCR结果")
            return False
        result_type_name = type(first_result).__name__
        if result_type_name == 'OCRResult':
            ocr_result = parser._convert_ocr_result_to_list(first_result)
        else:
            ocr_result = first_result if isinstance(first_result, list) else []
    
    if not ocr_result:
        print("  未找到OCR识别结果")
        return False
    
    print(f"  识别到 {len(ocr_result)} 行文本")
    
    # 调试：打印前几行OCR结果
    if len(ocr_result) > 0:
        print("  调试：前3行OCR文本内容：")
        for i, line_data in enumerate(ocr_result[:3]):
            if line_data and len(line_data) >= 2:
                text_info = line_data[1]
                text = text_info[0] if isinstance(text_info, (list, tuple)) and len(text_info) > 0 else ""
                print(f"    行{i+1}: {text[:50]}")
    
    # 找到第一行的题目编码关键词位置
    keyword_bbox = detect_keyword_position(ocr_result, image_width)
    if not keyword_bbox:
        print("  未找到题目编码关键词，跳过")
        # 调试：尝试手动检查第一行
        if len(ocr_result) > 0:
            first_line = ocr_result[0]
            if first_line and len(first_line) >= 2:
                text_info = first_line[1]
                first_text = text_info[0] if isinstance(text_info, (list, tuple)) and len(text_info) > 0 else ""
                print(f"  调试：第一行文本: '{first_text}'")
                keyword_info = extract_keyword(first_text)
                if keyword_info:
                    print(f"  调试：extract_keyword返回: {keyword_info}")
                else:
                    print(f"  调试：extract_keyword未匹配到关键词")
        return False
    
    x_min, y_min, x_max, y_max = keyword_bbox
    print(f"  找到关键词位置: ({x_min}, {y_min}, {x_max}, {y_max})")
    
    # 使用黑色正方形遮挡
    draw = ImageDraw.Draw(image)
    
    # 计算正方形尺寸（使用关键词的高度，确保完全覆盖）
    keyword_height = y_max - y_min
    square_size = max(keyword_height, x_max - x_min)  # 取高度和宽度的较大值
    
    # 计算正方形中心点（关键词的中心）
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # 绘制浅灰色正方形
    half_size = square_size / 2
    left = center_x - half_size
    top = center_y - half_size
    right = center_x + half_size
    bottom = center_y + half_size
    
    # 确保不超出图片边界
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)
    
    # 使用浅灰色 (220, 220, 220)
    light_gray = (255, 255, 255)
    draw.rectangle(
        [left, top, right, bottom],
        fill=light_gray,
        outline=light_gray
    )
    
    print(f"  已用白色正方形遮挡关键词（尺寸: {square_size:.1f}px）")
    
    # 左侧对齐处理：使关键词距离左边缘固定300px
    target_left_margin = 150  # 目标左边距（px）
    keyword_center_x = (x_min + x_max) / 2
    
    if keyword_center_x < target_left_margin:
        # 关键词太靠左，需要在左侧补齐白色区域，使关键词向右移动到目标位置
        padding_left = target_left_margin - keyword_center_x
        new_width = int(image_width + padding_left)
        new_height = int(image_height)
        
        # 创建新图片，左侧填充白色
        white = (255, 255, 255)
        new_image = Image.new('RGB', (new_width, new_height), white)
        new_image.paste(image, (int(padding_left), 0))
        image = new_image
        print(f"  已在左侧补齐 {padding_left:.1f}px 白色区域，使关键词距离左边缘 {target_left_margin}px")
    elif keyword_center_x > target_left_margin:
        # 关键词太靠右，需要裁剪左侧，使关键词向左移动到目标位置
        crop_x = keyword_center_x - target_left_margin
        new_width = image_width - crop_x
        if new_width > 0:
            image = image.crop((int(crop_x), 0, image_width, image_height))
            print(f"  已裁剪左侧 {crop_x:.1f}px，使关键词距离左边缘 {target_left_margin}px")
        else:
            print(f"  警告: 裁剪后宽度为0，跳过裁剪")
    else:
        # 关键词位置正好，无需调整
        print(f"  关键词位置已符合要求（距离左边缘 {target_left_margin}px）")
    
    # 保存图片（如果未指定输出路径，则覆盖原文件）
    if output_path is None:
        output_path = image_path
    
    image.save(output_path)
    if output_path == image_path:
        print(f"  已覆盖原文件: {output_path}")
    else:
        print(f"  已保存到: {output_path}")
    
    return True


def process_directory(input_dir: str, output_dir: str = None, pattern: str = '*.png'):
    """
    批量处理目录中的图片
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录（如果为None，则覆盖原文件）
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 处理输出目录
    if output_dir is None:
        output_path = None  # None表示覆盖原文件
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化配置和OCR引擎（只初始化一次，提高效率）
    print("初始化OCR引擎（PPStructure）...")
    config = Config()
    config.use_ppstructure = True
    ocr_engine = OCREngine(config=config)
    print("OCR引擎初始化完成\n")
    
    # 查找所有图片
    image_files = list(input_path.glob(pattern))
    if not image_files:
        print(f"未找到匹配的图片文件: {pattern}")
        return
    
    print(f"找到 {len(image_files)} 张图片\n")
    
    # 处理每张图片
    success_count = 0
    skip_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] ", end='')
        # 如果指定了输出目录，保存到输出目录；否则覆盖原文件
        if output_path and output_path != input_path:
            output_file = output_path / image_file.name
        else:
            output_file = None  # None表示覆盖原文件
        
        # 传递已初始化的OCR引擎，避免重复初始化
        if remove_question_keyword(str(image_file), str(output_file) if output_file else None, config, ocr_engine):
            success_count += 1
        else:
            skip_count += 1
        print()
    
    print(f"\n处理完成:")
    print(f"  成功: {success_count} 张")
    print(f"  跳过: {skip_count} 张")
    if output_path:
        print(f"  输出目录: {output_path}")
    else:
        print(f"  已覆盖原文件")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  单张图片: python tools/remove_question_keyword.py <图片路径> [输出路径]")
        print("  批量处理: python tools/remove_question_keyword.py <目录路径> [输出目录]")
        print()
        print("说明:")
        print("  - 如果不指定输出路径，将覆盖原文件")
        print("  - 使用浅灰色正方形遮挡关键词")
        print()
        print("示例:")
        print("  python tools/remove_question_keyword.py exam_Q1_p1.png  # 覆盖原文件")
        print("  python tools/remove_question_keyword.py exam_Q1_p1.png output.png  # 保存到新文件")
        print("  python tools/remove_question_keyword.py ./output  # 覆盖原目录中的文件")
        print("  python tools/remove_question_keyword.py ./output ./output_masked  # 保存到新目录")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    input_path_obj = Path(input_path)
    
    if input_path_obj.is_file():
        # 单张图片处理
        config = Config()
        config.use_ppstructure = True
        remove_question_keyword(input_path, output_path, config)
    elif input_path_obj.is_dir():
        # 批量处理
        process_directory(input_path, output_path)
    else:
        print(f"错误: 路径不存在: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

