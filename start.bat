@echo off

path=%path%;C:\Windows\system32

%1 mshta vbscript:CreateObject("Shell.Application").ShellExecute("cmd.exe","/c %~s0 ::","","runas",1)(window.close)&&exit
cd /d "%~dp0"

setlocal
set uac=~uac_permission_tmp_%random%
md "%SystemRoot%\system32\%uac%" 2>nul
if %errorlevel%==0 ( rd "%SystemRoot%\system32\%uac%" >nul 2>nul ) else (
    echo set uac = CreateObject^("Shell.Application"^)>"%temp%\%uac%.vbs"
    echo uac.ShellExecute "%~s0","","","runas",1 >>"%temp%\%uac%.vbs"
    echo WScript.Quit >>"%temp%\%uac%.vbs"
    "%temp%\%uac%.vbs" /f
    del /f /q "%temp%\%uac%.vbs" & exit )
endlocal

:: 进入当前脚本所在目录
cd /d "%~dp0"

:: 激活虚拟环境并使用 pythonw.exe 隐藏控制台窗口运行 Python 脚本
call ".\venv\Scripts\activate.bat"
start "" ".\venv\Scripts\pythonw.exe" ".\ByPass.py"
rem start "" ".\venv\Scripts\python.exe" ".\ByPass.py"
:: 退出
exit