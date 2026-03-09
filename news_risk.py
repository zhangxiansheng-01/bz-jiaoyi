# 金十数据新闻风控模块 v4 - 自适应版
# 功能：根据危机等级自动调整交易参数

import os
import time
import pickle
from datetime import datetime

# 危机等级配置
CRISIS_LEVELS = {
    0: {"name": "正常", "lot_mult": 1.0, "desc": "无危机"},
    1: {"name": "低危机", "lot_mult": 0.9, "desc": "一般新闻"},
    2: {"name": "高危机", "lot_mult": 0.7, "desc": "重大事件"},
    3: {"name": "极端危机", "lot_mult": 0, "desc": "全面战争/恐袭"},
}

# 危机关键词及等级
CRISIS_KEYWORDS = {
    # 极端危机 (Level 3) - 全面战争、大规模恐袭
    3: ["第三次世界大战", "核战争", "核弹", "全面战争", "大规模恐袭", 
        "哈梅内伊", "苏莱曼尼", "伊朗最高领袖", "导弹袭击美军基地",
        "伊朗向以色列发射导弹", "以色列发动全面战争"],
    
    # 高危机 (Level 2) - 地区冲突升级
    2: ["战争", "空袭", "轰炸", "导弹", "冲突升级", "局势紧张",
        "以色列", "伊朗", "中东", "美军基地", "袭击", "爆炸",
        "无人机", "紧张", "紧急", "避险", "恐慌"],
    
    # 低危机 (Level 1) - 一般财经新闻
    1: ["下跌", "崩盘", "暴跌", "恐慌", "风险", "警告",
        "衰退", "裁员", "制裁", "争端"],
}

class NewsRiskManager:
    def __init__(self, state_file="news_risk_state.pkl"):
        self.state_file = state_file
        self.last_check_time = 0
        self.crisis_level = 0
        self.last_news = []
        self.crisis_count = 0
        self.load_state()
    
    def save_state(self):
        state = {
            "last_check_time": self.last_check_time,
            "crisis_level": self.crisis_level,
            "crisis_count": self.crisis_count,
        }
        try:
            with open(self.state_file, "wb") as f:
                pickle.dump(state, f)
        except:
            pass
    
    def load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "rb") as f:
                    state = pickle.load(f)
                    self.last_check_time = state.get("last_check_time", 0)
                    self.crisis_level = state.get("crisis_level", 0)
                    self.crisis_count = state.get("crisis_count", 0)
        except:
            pass
    
    def parse_news(self, text):
        """解析新闻"""
        import re
        news_list = []
        lines = text.split('\n')
        for line in lines:
            match = re.match(r'(\d{2}:\d{2}:\d{2})\s+(.+)', line)
            if match:
                time_str, content = match.groups()
                if len(content) > 5:
                    news_list.append({"time": time_str, "content": content.strip()})
        return news_list
    
    def check_crisis_level(self, news_list):
        """检测危机等级"""
        max_level = 0
        crisis_news = []
        
        for news in news_list:
            content = news.get("content", "")
            
            # 从高到低检查
            for level in [3, 2, 1]:
                for keyword in CRISIS_KEYWORDS.get(level, []):
                    if keyword in content:
                        max_level = max(max_level, level)
                        crisis_news.append((level, keyword, content[:60]))
                        break
        
        return max_level, crisis_news[:10]
    
    def check_and_update(self, news_text=None):
        """检查新闻并更新状态"""
        if news_text:
            self.last_news = self.parse_news(news_text)
        
        if not self.last_news:
            return self.crisis_level
        
        level, crisis_news = self.check_crisis_level(self.last_news)
        
        if level > 0:
            self.crisis_count += 1
            self.crisis_level = level
            print(f"[危机等级 {level}] {CRISIS_LEVELS[level]['name']}")
            for lv, kw, content in crisis_news[:3]:
                print(f"  - [{kw}] {content}...")
        else:
            if self.crisis_count > 0:
                self.crisis_count -= 1
            if self.crisis_count <= 1:
                self.crisis_level = 0
                self.crisis_count = 0
                print("[正常] 无危机")
        
        self.last_check_time = time.time()
        self.save_state()
        
        return self.crisis_level
    
    def get_lot_multiplier(self):
        """获取仓位系数"""
        return CRISIS_LEVELS.get(self.crisis_level, {}).get("lot_mult", 1.0)
    
    def get_status(self):
        info = CRISIS_LEVELS.get(self.crisis_level, {})
        return {
            "crisis_level": self.crisis_level,
            "name": info.get("name", "未知"),
            "lot_mult": self.get_lot_multiplier(),
            "last_check": datetime.fromtimestamp(self.last_check_time).strftime("%H:%M:%S") if self.last_check_time else "Never"
        }

# 测试
if __name__ == "__main__":
    test_news = """
15:53:11 伊朗轰炸美国基地，中东局势紧张
15:50:27 阿曼港口遭无人机袭击
15:49:38 伊朗威胁对以色列发动导弹袭击
15:48:41 迪拜两栋房屋遭导弹碎片击中
15:45:31 伊朗空军轰炸美国在中东的军事基地
15:43:16 美国大使馆让公民避难
"""
    
    manager = NewsRiskManager()
    print("=== News Risk Manager v4 - Adaptive ===")
    print(f"Initial: {manager.get_status()}")
    
    level = manager.check_and_update(test_news)
    
    print(f"\nCrisis Level: {level}")
    print(f"Lot Multiplier: {manager.get_lot_multiplier()}")
    print(f"Status: {manager.get_status()}")
