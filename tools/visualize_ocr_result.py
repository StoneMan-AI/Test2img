#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleOCR识别结果可视化工具
用于可视化OCR识别文档时的真实情况，包括识别错误
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

# 导入项目模块
from src.file_handler import FileHandler
from src.config import Config

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    print("错误: 需要安装 PaddleOCR")
    PADDLEOCR_AVAILABLE = False


def visualize_ocr_result(docx_path: str, output_path: str = "ocr_visualization.png", page_num: int = 1):
    """
    对DOCX文档进行OCR识别并可视化结果
    
    Args:
        docx_path: DOCX文件路径
        output_path: 输出图片路径
        page_num: 要可视化的页码（从1开始）
    """
    if not PADDLEOCR_AVAILABLE:
        print("错误: PaddleOCR 未安装")
        return
    
    # 加载配置（使用与主程序相同的配置）
    config = Config()
    
    # 步骤1: 将DOCX转换为图片
    print("=" * 60)
    print("步骤1: 将DOCX转换为图片")
    print("=" * 60)
    file_handler = FileHandler(config)
    
    try:
        images = file_handler.convert_to_images(docx_path)
        if not images:
            print("错误: 未能从DOCX提取图片")
            return
        
        if page_num > len(images):
            print(f"警告: 文档只有 {len(images)} 页，使用第1页")
            page_num = 1
        
        image = images[page_num - 1]
        print(f"成功: 已转换第 {page_num} 页 (共 {len(images)} 页)")
        print(f"图片尺寸: {image.width} x {image.height}")
    except Exception as e:
        print(f"错误: DOCX转换失败: {e}")
        return
    
    # 步骤2: 使用与主程序相同的OCR配置
    print("\n" + "=" * 60)
    print("步骤2: 初始化OCR引擎（使用项目配置）")
    print("=" * 60)
    
    try:
        # 构建PaddleOCR初始化参数（与layout_analyzer.py保持一致）
        ocr_params = {
            'use_angle_cls': config.use_angle_cls,
            'lang': config.ocr_lang,
            'det_db_box_thresh': config.det_db_box_thresh,
            'det_db_unclip_ratio': config.det_db_unclip_ratio
        }
        
        # 如果配置了模型路径，则添加
        if config.det_model_dir:
            ocr_params['det_model_dir'] = config.det_model_dir
        if config.rec_model_dir:
            ocr_params['rec_model_dir'] = config.rec_model_dir
        
        print(f"OCR配置:")
        print(f"  - use_angle_cls: {ocr_params['use_angle_cls']}")
        print(f"  - lang: {ocr_params['lang']}")
        print(f"  - det_db_box_thresh: {ocr_params['det_db_box_thresh']}")
        print(f"  - det_db_unclip_ratio: {ocr_params['det_db_unclip_ratio']}")
        if config.det_model_dir:
            print(f"  - det_model_dir: {config.det_model_dir}")
        if config.rec_model_dir:
            print(f"  - rec_model_dir: {config.rec_model_dir}")
        
        ocr = PaddleOCR(**ocr_params)
        print("OCR引擎初始化成功")
    except Exception as e:
        print(f"错误: OCR引擎初始化失败: {e}")
        return
    
    # 步骤3: 执行OCR识别
    print("\n" + "=" * 60)
    print("步骤3: 执行OCR识别")
    print("=" * 60)
    
    img_array = np.array(image.convert('RGB'))
    ocr_response = ocr.ocr(img_array)
    
    if not ocr_response:
        print("警告: OCR未识别到任何内容")
        return
    
    # 处理OCR返回结果（兼容新旧版本）
    first_result = ocr_response[0]
    result_type_name = type(first_result).__name__
    
    if result_type_name == 'OCRResult':
        # 新版本：OCRResult对象
        print("检测到新版本OCRResult格式，正在转换...")
        ocr_result = _convert_ocr_result_to_list(first_result)
    else:
        # 旧版本：直接是列表
        ocr_result = first_result if isinstance(first_result, list) else []
    
    if not ocr_result:
        print("警告: 未能解析OCR结果")
        return
    
    print(f"成功: 识别到 {len(ocr_result)} 行文本")
    
    # 步骤4: 可视化识别结果
    print("\n" + "=" * 60)
    print("步骤4: 生成可视化图片")
    print("=" * 60)
    
    # 创建可绘制的图片副本（放大以便看清细节）
    scale = 1.0  # 可以调整放大倍数
    if scale > 1.0:
        new_size = (int(image.width * scale), int(image.height * scale))
        draw_image = image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        draw_image = image.copy()
    
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试加载字体
    try:
        if os.name == 'nt':  # Windows
            font_small = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 12)
            font_large = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
        else:  # Linux/macOS
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_small = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    # 绘制每个识别结果
    colors = {
        'high_confidence': 'green',      # 高置信度（>=0.8）
        'medium_confidence': 'orange',   # 中等置信度（>=0.5）
        'low_confidence': 'red'           # 低置信度（<0.5）
    }
    
    print("\n识别结果详情:")
    print("-" * 60)
    
    for idx, line_data in enumerate(ocr_result):
        if not line_data:
            continue
        
        line_box = line_data[0]
        line_text = line_data[1][0] if len(line_data[1]) > 0 else ""
        confidence = line_data[1][1] if len(line_data[1]) > 1 else 0.0
        
        # 缩放坐标（如果图片被放大）
        scaled_box = [[int(p[0] * scale), int(p[1] * scale)] for p in line_box]
        
        # 计算边界框
        x_coords = [point[0] for point in scaled_box]
        y_coords = [point[1] for point in scaled_box]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # 根据置信度选择颜色
        if confidence >= 0.8:
            color = colors['high_confidence']
        elif confidence >= 0.5:
            color = colors['medium_confidence']
        else:
            color = colors['low_confidence']
        
        # 绘制边界框（四边形，支持倾斜）
        if len(scaled_box) == 4:
            # 绘制四边形的四条边
            for i in range(4):
                start_point = tuple(scaled_box[i])
                end_point = tuple(scaled_box[(i + 1) % 4])
                draw.line([start_point, end_point], fill=color, width=2)
        else:
            # 如果格式不对，绘制矩形
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=2)
        
        # 显示识别文本和置信度
        text_display = f"[{idx}] {line_text[:40]}"  # 限制长度
        conf_text = f"{confidence:.2f}"
        
        # 计算文本背景框
        try:
            bbox = draw.textbbox((x_min, y_min - 20), text_display, font=font_small)
            # 绘制文本背景（半透明效果用白色背景代替）
            draw.rectangle(bbox, fill="white", outline=color)
            # 绘制文本
            draw.text((x_min, y_min - 20), text_display, fill=color, font=font_small)
            
            # 绘制置信度
            conf_bbox = draw.textbbox((x_max - 60, y_min - 20), conf_text, font=font_small)
            draw.rectangle(conf_bbox, fill="yellow", outline=color)
            draw.text((x_max - 60, y_min - 20), conf_text, fill="black", font=font_small)
        except:
            # 如果字体渲染失败，至少绘制一个标记
            draw.ellipse([x_min - 5, y_min - 5, x_min + 5, y_min + 5], fill=color)
        
        # 打印到控制台
        status = "✓" if confidence >= 0.8 else ("?" if confidence >= 0.5 else "✗")
        print(f"{idx:3d} {status} [{confidence:.3f}] {line_text}")
    
    # 添加图例
    legend_y = 10
    legend_x = 10
    draw.text((legend_x, legend_y), "图例:", fill="black", font=font_large)
    legend_y += 20
    draw.text((legend_x, legend_y), "绿色: 高置信度(>=0.8)", fill=colors['high_confidence'], font=font_small)
    legend_y += 15
    draw.text((legend_x, legend_y), "橙色: 中置信度(0.5-0.8)", fill=colors['medium_confidence'], font=font_small)
    legend_y += 15
    draw.text((legend_x, legend_y), "红色: 低置信度(<0.5)", fill=colors['low_confidence'], font=font_small)
    
    # 添加统计信息
    high_conf = sum(1 for item in ocr_result if item and len(item) > 1 and item[1][1] >= 0.8)
    medium_conf = sum(1 for item in ocr_result if item and len(item) > 1 and 0.5 <= item[1][1] < 0.8)
    low_conf = sum(1 for item in ocr_result if item and len(item) > 1 and item[1][1] < 0.5)
    
    stats_y = legend_y + 30
    draw.text((legend_x, stats_y), f"统计: 总计{len(ocr_result)}行", fill="black", font=font_small)
    stats_y += 15
    draw.text((legend_x, stats_y), f"  高置信度: {high_conf}行", fill=colors['high_confidence'], font=font_small)
    stats_y += 15
    draw.text((legend_x, stats_y), f"  中置信度: {medium_conf}行", fill=colors['medium_confidence'], font=font_small)
    stats_y += 15
    draw.text((legend_x, stats_y), f"  低置信度: {low_conf}行", fill=colors['low_confidence'], font=font_small)
    
    # 保存结果
    draw_image.save(output_path)
    
    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"可视化结果已保存到: {output_path}")
    print(f"共识别 {len(ocr_result)} 行文本")
    print(f"  - 高置信度: {high_conf} 行")
    print(f"  - 中置信度: {medium_conf} 行")
    print(f"  - 低置信度: {low_conf} 行")
    print("\n说明:")
    print("  - 绿色框: 识别置信度高的文本")
    print("  - 橙色框: 识别置信度中等的文本")
    print("  - 红色框: 识别置信度低的文本（可能识别错误）")
    print("  - 每行显示: [序号] 识别文本")
    print("  - 右上角黄色框显示置信度分数")


def _convert_ocr_result_to_list(ocr_result_obj):
    """
    将新版本OCRResult对象转换为旧版本格式的列表
    """
    result_list = []
    
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
    
    if texts is not None and polys is not None:
        scores = scores if scores is not None else [1.0] * len(texts)
        
        for i, (text, poly) in enumerate(zip(texts, polys)):
            # 转换坐标为旧格式 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            coord_list = poly.tolist() if hasattr(poly, 'tolist') else poly
            
            # 获取置信度
            conf = scores[i] if i < len(scores) else 1.0
            
            # 构建旧格式：[坐标, (文本, 置信度)]
            result_list.append([coord_list, (text, conf)])
    
    return result_list


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PaddleOCR识别结果可视化工具")
    parser.add_argument("input", help="输入DOCX文件路径")
    parser.add_argument("-o", "--output", default="ocr_visualization.png",
                       help="输出可视化图片路径 (默认: ocr_visualization.png)")
    parser.add_argument("-p", "--page", type=int, default=1,
                       help="要可视化的页码（从1开始，默认: 1）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    visualize_ocr_result(args.input, args.output, args.page)

