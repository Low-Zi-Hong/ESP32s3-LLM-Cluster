@echo off
chcp 65001 > nul
echo ========================================
echo     ESP32-S3 Node 节点全自动烧录流水线
echo ========================================

:LOOP
echo.
echo [等待操作] 请用 Type-C 线连接下一块 Node 板子...
echo (插好后，直接按回车键开始烧录)
pause > nul

echo [烧录中] 正在自动寻找 COM 口并烧录...
idf.py flash

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [成功] 烧录完成！请拔下线缆。
) else (
    echo.
    echo [失败] 烧录出错，请检查连接或重新按回车尝试。
)

goto LOOP