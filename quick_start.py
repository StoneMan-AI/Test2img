#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速开始脚本
用于测试和演示试卷分割功能
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.file_handler import FileHandler
from src.layout_analyzer import LayoutAnalyzer
from src.cropper import ImageCropper


def quick_test():
    """快速测试函数"""
    print("="*60)
    print("试卷题目分割工具 - 快速测试")
    print("="*60)
    
    # 检查是否有测试文件
    test_files = [
        "sample.pdf",
        "sample.jpg",
        "sample.png",
        "test.pdf",
        "test.jpg",
        "exam.pdf",
    ]
    
    input_file = None
    for f in test_files:
        if os.path.exists(f):
            input_file = f
            break
    
    if not input_file:
        print("\n❌ 错误: 当前目录中未找到测试文件")
        print("\n请将测试用的PDF或图片文件放在当前目录，命名为以下之一：")
        for f in test_files:
            print(f"  - {f}")
        print("\n然后重新运行此脚本。")
        return
    
    print(f"\n✓ 找到测试文件: {input_file}")
    print(f"✓ 输出目录: ./output")
    
    # 创建配置
    config = Config()
    config.output_dir = Path("./output")
    
    # 创建输出目录
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化各个组件
    print("\n--- 初始化系统 ---")
    file_handler = FileHandler(config=config)
    layout_analyzer = LayoutAnalyzer(config=config)
    cropper = ImageCropper(config=config)
    
    # 处理文件
    print("\n--- 开始处理 ---")
    
    try:
        # 1. 转换为图片
        images = file_handler.convert_to_images(input_file)
        print(f"✓ 转换完成，共 {len(images)} 页")
        
        # 2. 分析布局
        all_zones = []
        for page_idx, image in enumerate(images, 1):
            print(f"\n分析第 {page_idx} 页...")
            try:
                zones = layout_analyzer.analyze_page(image, page_idx)
                all_zones.append({
                    "page": page_idx,
                    "image": image,
                    "zones": zones
                })
                print(f"✓ 识别出 {len(zones)} 个区域")
            except Exception as e:
                print(f"✗ 分析失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 3. 裁剪保存
        base_name = Path(input_file).stem
        output_info = cropper.crop_and_save(all_zones, base_name)
        
        # 输出结果
        print("\n" + "="*60)
        print("✅ 处理完成！")
        print("="*60)
        print(f"输出目录: {output_info['output_dir']}")
        print(f"题目数: {output_info['questions']}")
        print(f"答案数: {output_info['answers']}")
        print(f"总计: {output_info['total_saved']} 个文件")
        
        # 列出生成的文件
        output_path = Path(output_info['output_dir'])
        files = sorted(output_path.glob(f"{base_name}_*.png"))
        if files:
            print(f"\n生成的文件:")
            for f in files[:10]:  # 只显示前10个
                print(f"  - {f.name}")
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n提示：如果识别结果不理想，请检查以下内容：")
    print("1. 输入文件是否清晰（建议至少200 DPI）")
    print("2. 题目编号格式是否匹配默认规则")
    print("3. 可以修改 src/config.py 中的识别规则")


if __name__ == "__main__":
    quick_test()

