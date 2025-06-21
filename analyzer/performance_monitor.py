#!/usr/bin/env python3
"""
性能監控腳本
用於追蹤個人色彩分析器的性能表現
"""

import time
import psutil
import json
import os
from datetime import datetime
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = None
        self.log_file = "performance_log.json"
        
    def start_monitoring(self, operation_name):
        """開始監控操作"""
        self.start_time = time.time()
        self.current_operation = operation_name
        print(f"🔍 開始監控: {operation_name}")
        
    def end_monitoring(self, success=True, error_msg=None):
        """結束監控操作"""
        if self.start_time is None:
            return
            
        duration = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        metric = {
            "operation": self.current_operation,
            "duration": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": round(memory.used / 1024 / 1024, 2),
            "success": success,
            "error": error_msg
        }
        
        self.metrics[self.current_operation].append(metric)
        
        # 即時輸出
        status = "✅" if success else "❌"
        print(f"{status} {self.current_operation} 完成，耗時: {duration:.3f}秒")
        print(f"   CPU: {cpu_percent}% | 記憶體: {memory.percent}% ({memory.used / 1024 / 1024:.1f}MB)")
        
        if error_msg:
            print(f"   錯誤: {error_msg}")
            
        self.start_time = None
        
    def get_average_performance(self, operation_name=None):
        """獲取平均性能數據"""
        if operation_name:
            metrics = self.metrics.get(operation_name, [])
        else:
            # 合併所有操作
            metrics = []
            for op_metrics in self.metrics.values():
                metrics.extend(op_metrics)
                
        if not metrics:
            return {}
            
        durations = [m["duration"] for m in metrics if m["success"]]
        cpu_percents = [m["cpu_percent"] for m in metrics if m["success"]]
        memory_percents = [m["memory_percent"] for m in metrics if m["success"]]
        
        return {
            "total_operations": len(metrics),
            "successful_operations": len([m for m in metrics if m["success"]]),
            "average_duration": round(sum(durations) / len(durations), 3) if durations else 0,
            "min_duration": round(min(durations), 3) if durations else 0,
            "max_duration": round(max(durations), 3) if durations else 0,
            "average_cpu": round(sum(cpu_percents) / len(cpu_percents), 1) if cpu_percents else 0,
            "average_memory": round(sum(memory_percents) / len(memory_percents), 1) if memory_percents else 0
        }
        
    def save_logs(self):
        """保存性能日誌"""
        log_data = {
            "summary": self.get_average_performance(),
            "operations": dict(self.metrics),
            "exported_at": datetime.now().isoformat()
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
            
        print(f"📊 性能日誌已保存至: {self.log_file}")
        
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_average_performance()
        
        print("\n" + "="*50)
        print("📈 性能監控摘要")
        print("="*50)
        print(f"總操作次數: {summary['total_operations']}")
        print(f"成功操作次數: {summary['successful_operations']}")
        print(f"平均處理時間: {summary['average_duration']}秒")
        print(f"最快處理時間: {summary['min_duration']}秒")
        print(f"最慢處理時間: {summary['max_duration']}秒")
        print(f"平均 CPU 使用率: {summary['average_cpu']}%")
        print(f"平均記憶體使用率: {summary['average_memory']}%")
        print("="*50)
        
        # 各操作詳細統計
        for operation in self.metrics:
            op_summary = self.get_average_performance(operation)
            print(f"\n🔍 {operation}:")
            print(f"   次數: {op_summary['total_operations']}")
            print(f"   平均時間: {op_summary['average_duration']}秒")
            print(f"   時間範圍: {op_summary['min_duration']} - {op_summary['max_duration']}秒")

# 全域監控實例
monitor = PerformanceMonitor()

def monitor_function(func):
    """裝飾器：自動監控函數性能"""
    def wrapper(*args, **kwargs):
        monitor.start_monitoring(func.__name__)
        try:
            result = func(*args, **kwargs)
            monitor.end_monitoring(success=True)
            return result
        except Exception as e:
            monitor.end_monitoring(success=False, error_msg=str(e))
            raise
    return wrapper

if __name__ == "__main__":
    # 測試監控功能
    print("🧪 測試性能監控功能...")
    
    @monitor_function
    def test_operation():
        time.sleep(0.5)  # 模擬操作
        return "success"
    
    # 執行測試
    for i in range(3):
        test_operation()
        
    # 打印摘要
    monitor.print_summary()
    monitor.save_logs() 