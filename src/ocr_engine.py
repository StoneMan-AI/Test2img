#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OCR引擎管理模块
负责OCR引擎的初始化、选择和配置
"""

# 尝试导入PPStructure（可能是PPStructure或PPStructureV3）
PPSTRUCTURE_AVAILABLE = False
PPStructure = None

try:
    from paddleocr import PPStructureV3 as PPStructure
    PPSTRUCTURE_AVAILABLE = True
except ImportError:
    try:
        from paddleocr import PPStructure
        PPSTRUCTURE_AVAILABLE = True
    except ImportError as e:
        PPSTRUCTURE_AVAILABLE = False
        print(f"[提示] PPStructure未导入成功 ({str(e)})，将使用标准PaddleOCR")
        PPStructure = None

# 备用：标准PaddleOCR（如果PPStructure不可用）
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    PaddleOCR = None


class OCREngine:
    """OCR引擎管理类"""
    
    def __init__(self, config=None):
        """
        初始化OCR引擎
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.ocr = None
        self.use_ppstructure = False
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化OCR引擎（优先使用PPStructure，否则使用标准PaddleOCR）"""
        # 检查配置是否启用PPStructure
        use_ppstructure_config = (
            self.config and 
            hasattr(self.config, 'use_ppstructure') and 
            self.config.use_ppstructure
        ) or (not self.config)  # 如果没有配置，默认启用
        
        if use_ppstructure_config and PPSTRUCTURE_AVAILABLE and PPStructure:
            self._init_ppstructure()
        
        # 如果PPStructure不可用，使用标准PaddleOCR作为备用
        if not self.use_ppstructure:
            self._init_paddleocr()
        
        if self.ocr is None:
            raise RuntimeError("无法初始化OCR引擎，请检查依赖安装")
    
    def _init_ppstructure(self):
        """初始化PPStructure引擎"""
        try:
            print("[初始化] 尝试使用PPStructure（文档优化版本）...")
            ppstructure_params = {
                'lang': self.config.ocr_lang if self.config else "ch",
                # 默认只启用文本检测和识别，不启用表格等其他功能（加快速度）
                'use_table_recognition': False,
                'use_formula_recognition': False,
                'use_chart_recognition': False,
                'use_seal_recognition': False,
                'use_doc_orientation_classify': False,
                'use_doc_unwarping': False,
            }
            
            # PPStructureV3使用的参数名与标准PaddleOCR不同
            # 如果配置了模型路径，则添加（使用PPStructureV3的参数名）
            if self.config and self.config.det_model_dir:
                ppstructure_params['text_detection_model_dir'] = self.config.det_model_dir
            if self.config and self.config.rec_model_dir:
                ppstructure_params['text_recognition_model_dir'] = self.config.rec_model_dir
            
            # 添加文本检测相关参数（如果配置了）
            if self.config:
                if hasattr(self.config, 'det_db_box_thresh'):
                    ppstructure_params['text_det_box_thresh'] = self.config.det_db_box_thresh
                if hasattr(self.config, 'det_limit_side_len'):
                    ppstructure_params['text_det_limit_side_len'] = self.config.det_limit_side_len
                if hasattr(self.config, 'det_limit_type'):
                    ppstructure_params['text_det_limit_type'] = self.config.det_limit_type
            
            self.ocr = PPStructure(**ppstructure_params)
            self.use_ppstructure = True
            print("[成功] PPStructure引擎初始化成功（文档优化版本）")
            if self.config and (self.config.det_model_dir or self.config.rec_model_dir):
                print(f"  -> 使用自定义模型路径")
            print(f"  -> 语言: {ppstructure_params['lang']}")
        except Exception as e:
            error_msg = str(e)
            print(f"[警告] PPStructure初始化失败")
            
            # 检查是否是依赖问题
            if "dependency" in error_msg.lower() or "DependencyError" in error_msg:
                print(f"  -> 原因: PPStructureV3需要额外的依赖包")
                print(f"  -> 解决方案: 运行以下命令安装依赖")
                print(f"     pip install \"paddlex[ocr]\"")
                print(f"  -> 或者继续使用标准PaddleOCR（当前降级方案）")
            else:
                print(f"  -> 错误详情: {error_msg}")
            
            import traceback
            if self.config and self.config.mode == 'debug':
                traceback.print_exc()
            
            print(f"[降级] 将尝试使用标准PaddleOCR作为备用...")
            self.use_ppstructure = False
            self.ocr = None
        else:
            # PPStructure不可用，说明原因
            if not PPSTRUCTURE_AVAILABLE:
                print("[提示] PPStructure未导入成功，将使用标准PaddleOCR")
            elif not PPStructure:
                print("[提示] PPStructure为None，将使用标准PaddleOCR")
    
    def _init_paddleocr(self):
        """初始化标准PaddleOCR引擎"""
        if not PADDLEOCR_AVAILABLE or not PaddleOCR:
            print("[错误] PaddleOCR不可用，无法初始化OCR引擎")
            return
        
        try:
            print("[初始化] 使用标准PaddleOCR...")
            # 构建PaddleOCR初始化参数
            ocr_params = {
                'use_angle_cls': self.config.use_angle_cls if self.config else True,
                'lang': self.config.ocr_lang if self.config else "ch",
                'det_db_box_thresh': self.config.det_db_box_thresh if self.config else 0.45,
                'det_db_unclip_ratio': self.config.det_db_unclip_ratio if self.config else 1.6
            }
            
            # 添加图像预处理相关参数（用于提升坐标精度）
            if self.config:
                # 尝试使用新版本参数名，如果失败则使用旧参数名
                try:
                    ocr_params['text_det_limit_side_len'] = self.config.det_limit_side_len
                    ocr_params['text_det_limit_type'] = self.config.det_limit_type
                except AttributeError:
                    # 如果没有配置这些参数，使用默认值
                    try:
                        ocr_params['det_limit_side_len'] = self.config.det_limit_side_len if hasattr(self.config, 'det_limit_side_len') else 2048
                        ocr_params['det_limit_type'] = self.config.det_limit_type if hasattr(self.config, 'det_limit_type') else 'max'
                    except:
                        pass
            
            # 如果配置了模型路径，则添加
            if self.config and self.config.det_model_dir:
                ocr_params['det_model_dir'] = self.config.det_model_dir
            if self.config and self.config.rec_model_dir:
                ocr_params['rec_model_dir'] = self.config.rec_model_dir
            
            self.ocr = PaddleOCR(**ocr_params)
            print("[成功] PaddleOCR引擎初始化成功（标准版本）")
            if self.config and (self.config.det_model_dir or self.config.rec_model_dir):
                print(f"  -> 使用自定义模型路径")
            print(f"  -> det_db_box_thresh={ocr_params['det_db_box_thresh']}, det_db_unclip_ratio={ocr_params['det_db_unclip_ratio']}")
        except Exception as e:
            print(f"[错误] OCR引擎初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.ocr = None
    
    def predict(self, image):
        """
        执行OCR识别（PPStructure使用predict方法）
        
        Args:
            image: 图像数据
            
        Returns:
            OCR识别结果
        """
        if self.use_ppstructure:
            return self.ocr.predict(image)
        else:
            return self.ocr.ocr(image)

