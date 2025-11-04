#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
试卷题目分割工具 - 主程序
将PDF/图片格式的试卷自动分割成独立的题目和答案图片

作者: Auto
创建时间: 2024
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import re

# 项目模块导入
from src.file_handler import FileHandler
from src.layout_analyzer import LayoutAnalyzer
from src.cropper import ImageCropper
from src.config import Config


class ExamSplitter:
    """试卷分割主类"""
    
    def __init__(self, config: Config = None):
        """
        初始化试卷分割器
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or Config()
        self.file_handler = FileHandler(config=self.config)
        self.layout_analyzer = LayoutAnalyzer(config=self.config)
        self.cropper = ImageCropper(config=self.config)
        
        print("[成功] 试卷分割系统初始化完成")
    
    def process_file(self, input_path: str) -> Dict:
        """
        处理单个文件
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            处理结果字典
        """
        print(f"\n{'='*60}")
        print(f"开始处理文件: {os.path.basename(input_path)}")
        print(f"{'='*60}")
        
        # 1. 转换文件格式为图片
        try:
            images = self.file_handler.convert_to_images(input_path)
            if not images:
                return {"status": "failed", "message": "无法转换文件为图片"}
            print(f"[成功] 成功转换，共 {len(images)} 页")
        except Exception as e:
            print(f"[失败] 文件转换失败: {e}")
            return {"status": "failed", "message": str(e)}
        
        # 2. 全局分析所有页面，识别题目位置和边界
        try:
            questions = self.layout_analyzer.analyze_all_pages(images)
            print(f"[成功] 共识别出 {len(questions)} 个题目")
        except Exception as e:
            print(f"[失败] 页面分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "message": f"分析失败: {e}"}
        
        # 3. 裁剪并保存图片
        base_name = Path(input_path).stem
        output_info = self.cropper.crop_and_save(questions, base_name)
        
        result = {
            "status": "success",
            "input_file": input_path,
            "total_pages": len(images),
            "output_info": output_info
        }
        
        print(f"\n{'='*60}")
        print(f"处理完成！共生成 {output_info['total_saved']} 个文件")
        print(f"{'='*60}")
        
        return result
    
    def process_directory(self, input_dir: str) -> List[Dict]:
        """
        批量处理目录中的所有文件
        
        Args:
            input_dir: 输入目录路径
            
        Returns:
            处理结果列表
        """
        supported_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.bmp']
        files = []
        
        for ext in supported_extensions:
            files.extend(Path(input_dir).glob(f"*{ext}"))
            files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not files:
            print(f"目录中没有找到支持的文件格式: {supported_extensions}")
            return []
        
        results = []
        for file_path in files:
            result = self.process_file(str(file_path))
            results.append(result)
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='试卷题目分割工具 - 将PDF/图片自动分割成独立的题目和答案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  # 处理单个PDF文件
  python main.py -i sample.pdf -o ./output
  
  # 批量处理目录中的所有文件
  python main.py -i ./exam_papers/ -o ./output
  
  # 使用自定义配置
  python main.py -i sample.pdf --config custom_config.json
        '''
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入文件或目录路径')
    parser.add_argument('-o', '--output', default='./output',
                       help='输出目录 (默认: ./output)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (可选)')
    parser.add_argument('--mode', choices=['debug', 'normal', 'fast'], 
                       default='normal',
                       help='运行模式 (默认: normal)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config.from_file(args.config) if args.config else Config()
    config.output_dir = Path(args.output)
    config.mode = args.mode
    
    # 确保输出目录存在
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建分割器并处理
    splitter = ExamSplitter(config=config)
    
    input_path = Path(args.input)
    if input_path.is_file():
        splitter.process_file(str(input_path))
    elif input_path.is_dir():
        splitter.process_directory(str(input_path))
    else:
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()

