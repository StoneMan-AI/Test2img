@echo off
chcp 65001 >nul
echo ========================================
echo 试卷题目分割工具 - 快速示例
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo ✓ Python已安装
echo.

REM 检查是否已安装依赖
python -c "import paddleocr" >nul 2>&1
if errorlevel 1 (
    echo 首次运行，正在安装依赖...
    echo 这可能需要几分钟时间，请耐心等待...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖安装失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo ✓ 依赖已就绪
echo.

REM 运行快速测试
python quick_start.py

pause

