#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件模块
定义试卷分割的配置参数和规则
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import re


class Config:
    """试卷分割配置类"""
    
    def __init__(self):
        # 路径配置
        self.output_dir = Path("./output")
        
        # OCR配置
        self.ocr_lang = "ch"  # 使用中文模型（支持中英文）
        self.use_angle_cls = True  # 使用角度分类器
        self.det_model_dir = None  # 检测模型路径，None则自动下载
        self.rec_model_dir = None  # 识别模型路径，None则自动下载
        self.det_db_box_thresh = 0.6  # 过滤低置信度框
        self.det_db_unclip_ratio = 1  # 调小一点，框更贴近文字
        
        # 图像预处理配置（用于提升坐标精度）
        self.enable_image_preprocessing = True  # 是否启用图像预处理（裁剪空白、统一缩放）
        self.preprocess_target_height = 1280  # 预处理目标高度（像素）
        self.preprocess_auto_crop = True  # 是否自动裁剪空白区域
        self.det_limit_side_len = 2048  # 检测输入限制（避免过度压缩）
        self.det_limit_type = 'max'  # 限制类型：'max'或'min'
        
        # PPStructure配置
        self.use_ppstructure = True  # 是否尝试使用PPStructure（如果依赖可用）
        # 注意：PPStructure需要额外依赖，运行: pip install "paddlex[ocr]"
        # 如果未安装或初始化失败，会自动降级使用标准PaddleOCR
        
        # 运行模式
        self.mode = "normal"  # debug, normal, fast
        
        # 题目识别规则（正则表达式模式）
        # 这些模式用于识别题目的开始标记
        self.question_patterns = [
            # 中文数字大题（如 "一、", "二、"）- 最高优先级
            r'^[一二三四五六七八九十]+[、．]',
            # 数字编号（如 "1.", "2. ", "10."）- 必须跟中文或较长文本
            r'^\d+\.\S+',  # 数字+点+非空白字符
            # 注意：带括号数字（如 "(1)", "（2）", "[3]"）不再作为独立题目识别
            # 它们被定义为嵌套子题，会累积到上一级题目中
            # 题型标题（如 "选择题", "填空题"）
            r'^(选择题|填空题|简答题|计算题|阅读理解|解答题)',
            # 注意：不包含 A., B., C., D. 这些通常用作选项编号
        ]
        
        # 中文数字大题模式（用于识别大题级别）
        self.chinese_number_pattern = re.compile(r'^[一二三四五六七八九十]+[、．]')
        
        # 阿拉伯数字小题模式（用于识别小题级别，这些不应该在中文大题下独立识别）
        self.sub_question_patterns = [
            r'^\d+\.\S+',
        ]
        
        # 嵌套子题模式（(1)、（2）、[3]等，应该累积到上一级题目中）
        self.nested_sub_question_patterns = [
            r'^[（(）)\[\]][0-9]+[）)\]）]',  # (1), （2）, [3]等
        ]
        
        # 答案识别规则
        self.answer_patterns = [
            r'^【答案】',
            r'^【解析】',
            r'^【详解】',
            r'^参考答案',
            r'^Answer\s*:?',
            r'^答案\s*:?',
        ]
        
        # 区域识别配置
        # 注意: 根据新的需求，不应以固定尺寸限制题目区域
        # 这些值仅作为最后的兜底检查，应该设置得很小
        self.min_zone_height = 10  # 最小区域高度（像素）- 已放宽
        self.min_zone_width = 50   # 最小区域宽度（像素）- 已放宽
        self.vertical_margin = 10  # 垂直边距（像素）
        self.horizontal_margin = 5  # 水平边距（像素）
        
        # 区域类型定义
        self.zone_types = {
            "question": "题目",
            "answer": "答案",
            "option": "选项",
            "other": "其他"
        }
    
    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """
        从JSON文件加载配置
        
        Args:
            file_path: 配置文件路径
            
        Returns:
            配置对象
        """
        config = cls()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 更新配置
            for key, value in data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        except Exception as e:
            print(f"警告: 加载配置文件失败: {e}，使用默认配置")
        
        return config
    
    def save_to_file(self, file_path: str):
        """
        保存配置到JSON文件
        
        Args:
            file_path: 配置文件路径
        """
        data = {
            "output_dir": str(self.output_dir),
            "ocr_lang": self.ocr_lang,
            "use_angle_cls": self.use_angle_cls,
            "det_model_dir": self.det_model_dir,
            "rec_model_dir": self.rec_model_dir,
            "det_db_box_thresh": self.det_db_box_thresh,
            "det_db_unclip_ratio": self.det_db_unclip_ratio,
            "enable_image_preprocessing": self.enable_image_preprocessing,
            "preprocess_target_height": self.preprocess_target_height,
            "preprocess_auto_crop": self.preprocess_auto_crop,
            "det_limit_side_len": self.det_limit_side_len,
            "det_limit_type": self.det_limit_type,
            "mode": self.mode,
            "question_patterns": self.question_patterns,
            "answer_patterns": self.answer_patterns,
            "min_zone_height": self.min_zone_height,
            "min_zone_width": self.min_zone_width,
            "vertical_margin": self.vertical_margin,
            "horizontal_margin": self.horizontal_margin,
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def compile_patterns(self):
        """
        编译正则表达式模式（优化性能）
        
        Returns:
            编译后的模式字典
        """
        return {
            'question': [re.compile(p, re.IGNORECASE) for p in self.question_patterns],
            'answer': [re.compile(p, re.IGNORECASE) for p in self.answer_patterns],
            'chinese_number': self.chinese_number_pattern,
            'sub_question': [re.compile(p, re.IGNORECASE) for p in self.sub_question_patterns],
            'nested_sub_question': [re.compile(p, re.IGNORECASE) for p in self.nested_sub_question_patterns]
        }

