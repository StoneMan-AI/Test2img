#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR调试工具
用于可视化OCR识别结果，帮助调试识别规则
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("错误: 需要安装 PaddleOCR")
    PADDLEOCR_AVAILABLE = False


def debug_ocr(input_image_path: str, output_path: str = "debug_output.png"):
    """
    对图片进行OCR并在图片上标记识别结果
    
    Args:
        input_image_path: 输入图片路径
        output_path: 输出标记后的图片路径
    """
    if not PADDLEOCR_AVAILABLE:
        return
    
    print("正在初始化OCR引擎...")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    
    print("正在加载图片...")
    image = Image.open(input_image_path)
    img_array = np.array(image.convert('RGB'))
    
    print("正在执行OCR识别...")
    result = ocr.ocr(img_array)[0]
    
    if not result:
        print("未识别到任何内容")
        return
    
    print(f"识别到 {len(result)} 行文本")
    
    # 创建可绘制的图片副本
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试加载字体（如果失败则使用默认字体）
    try:
        # Windows字体
        font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
    except:
        try:
            # Linux字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # 绘制每个识别结果
    for idx, line_data in enumerate(result):
        if not line_data:
            continue
        
        line_box = line_data[0]
        line_text = line_data[1][0]
        confidence = line_data[1][1]
        
        # 计算边界框
        x_coords = [point[0] for point in line_box]
        y_coords = [point[1] for point in line_box]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # 绘制边界框（绿色）
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=2)
        
        # 绘制识别文本（红色，显示在左上角）
        text_display = f"{idx}: {line_text[:30]}"
        # 绘制文本背景
        bbox = draw.textbbox((x_min, y_min), text_display, font=font)
        draw.rectangle(bbox, fill="yellow", outline="red")
        draw.text((x_min, y_min), text_display, fill="red", font=font)
        
        print(f"{idx:3d}: {confidence:.2f} - {line_text}")
    
    # 保存标记后的图片
    draw_image.save(output_path)
    print(f"\n标记结果已保存到: {output_path}")
    print(f"共识别 {len(result)} 行文本")


def list_pattern_matches(input_image_path: str):
    """
    列出与各种模式匹配的文本行
    
    Args:
        input_image_path: 输入图片路径
    """
    import re
    
    if not PADDLEOCR_AVAILABLE:
        return
    
    patterns = {
        "数字题号": re.compile(r'^\d+\.'),
        "中文题号": re.compile(r'^[一二三四五六七八九十]+[、．]'),
        "括号题号": re.compile(r'^[（(）)\[\]][0-9]+[）)\]）]'),
        "答案标识": re.compile(r'^【答案】|^【解析】'),
        "题型": re.compile(r'^(选择题|填空题|简答题|计算题|阅读理解)'),
    }
    
    print("正在初始化OCR引擎...")
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")
    
    print("正在加载图片...")
    image = Image.open(input_image_path)
    img_array = np.array(image.convert('RGB'))
    
    print("正在执行OCR识别...")
    result = ocr.ocr(img_array)[0]
    
    if not result:
        print("未识别到任何内容")
        return
    
    print("\n模式匹配结果:")
    print("="*60)
    
    matches_found = {name: [] for name in patterns.keys()}
    
    for idx, line_data in enumerate(result):
        if not line_data:
            continue
        
        line_text = line_data[1][0]
        
        # 检查每个模式
        for pattern_name, pattern in patterns.items():
            if pattern.match(line_text.strip()):
                matches_found[pattern_name].append((idx, line_text))
    
    # 输出匹配结果
    for pattern_name, matches in matches_found.items():
        if matches:
            print(f"\n{pattern_name}:")
            for idx, text in matches:
                print(f"  [行{idx}] {text}")
    
    print("\n" + "="*60)
    print("提示：如果没有任何匹配，说明需要调整识别规则")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OCR调试工具")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", default="debug_output.png",
                       help="输出标记图片路径 (默认: debug_output.png)")
    parser.add_argument("--list", action="store_true",
                       help="列出模式匹配结果")
    
    args = parser.parse_args()
    
    if args.list:
        list_pattern_matches(args.input)
    else:
        debug_ocr(args.input, args.output)
        print("\n提示：运行 'python tools/debug_ocr.py <图片> --list' 查看模式匹配结果")

