#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易系统守护进程
功能：监控交易脚本，崩溃/断网时自动重启
"""

import os
import sys
import time
import subprocess
import signal
from datetime import datetime

SCRIPT_PATH = r"D:\TradingSystem\ai_jiaoyi_mt5.py"
MAX_RESTARTS = 10  # 最大重启次数
RESTART_DELAY = 10  # 重启延迟（秒）
CHECK_INTERVAL = 30  # 检查间隔（秒）

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}")
    with open(r"D:\TradingSystem\guardian.log", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")

def main():
    log("="*50)
    log("交易系统守护进程启动")
    log(f"监控脚本: {SCRIPT_PATH}")
    log("="*50)
    
    restart_count = 0
    process = None
    
    while restart_count < MAX_RESTARTS:
        try:
            # 启动交易脚本
            log(f"启动交易系统 (第{restart_count + 1}次)...")
            process = subprocess.Popen(
                [sys.executable, SCRIPT_PATH],
                cwd=os.path.dirname(SCRIPT_PATH),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # 等待进程结束
            log(f"交易系统进程 PID: {process.pid}")
            return_code = process.wait()
            
            # 进程结束，检查原因
            if return_code == 0:
                log("交易系统正常退出")
                break
            elif return_code == -signal.SIGINT:
                log("用户中断，守护进程退出")
                break
            else:
                log(f"交易系统异常退出，返回码: {return_code}")
                restart_count += 1
                
        except KeyboardInterrupt:
            log("守护进程被用户中断")
            if process:
                process.terminate()
            break
        except Exception as e:
            log(f"守护进程异常: {e}")
            restart_count += 1
        
        # 重启前等待
        if restart_count < MAX_RESTARTS:
            log(f"{RESTART_DELAY}秒后尝试重启...")
            time.sleep(RESTART_DELAY)
    
    if restart_count >= MAX_RESTARTS:
        log(f"达到最大重启次数 ({MAX_RESTARTS})，守护进程退出")
    
    log("守护进程已退出")

if __name__ == "__main__":
    main()
