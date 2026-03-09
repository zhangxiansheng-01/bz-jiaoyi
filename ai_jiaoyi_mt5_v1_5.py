#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BZ交易系统 1.5.9 - Scaler修复版
=======================
版本: 1.5.9
日期: 2026-03-09
作者: 包子

修复内容:
- 支持多品种交易 (GOLD + CrudeOIL)
- 修复信号映射
- 修复模型加载
- 增强风控
- 模型保存/加载
- 新增30分钟冷静期，同品种不重复开仓
- 修复：平仓后也要记录时间，冷静期从平仓开始算
- 新增：自动检测止盈止损平仓并记录冷静期
- 实盘参数：初始资金$1000，手数0.03
- 修复：只在模型未训练时训练和保存
- 修复：简化持仓检测逻辑
- 修复：有持仓时检查方向，与信号相反则平仓
- 修复：scaler使用scaler_fitted标志判断，不再每次重新fit
- 修复：预测用最后3根K线平均，更稳定

运行: python ai_jiaoyi_mt5_v1_5.py
"""

import os
import sys
import time
import pickle
import random
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("警告: MT5未安装，将使用模拟模式")

warnings.filterwarnings('ignore')


# ==================== 配置 ====================

class Config:
    # 交易品种
    SYMBOLS = ["GOLD", "CrudeOIL"]  # 黄金、原油
    DEFAULT_SYMBOL = "GOLD"
    
    # MT5配置
    MT5_LOGIN = 101656064
    MT5_PASSWORD = "520523Zzx@"
    MT5_SERVER = "Ava-Demo 1-MT5"
    
    # 模型权重
    MODEL_WEIGHTS = {"xgb": 0.20, "lgb": 0.20, "catboost": 0.20, "rf": 0.20, "ga": 0.20}
    
    MIN_SAMPLES = 500
    
    FEATURES = ['atr', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'adx', 'di_plus', 'di_minus', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'ema_5', 'ema_20', 'ema_50', 'sma_5', 'sma_20', 'volume_ma', 'volume_ratio', 'high_low_ratio', 'close_open_ratio', 'returns_1', 'returns_3', 'returns_5']
    
    # 交易参数
    STOP_LOSS_ATR = 20   # 黄金止损20美元
    TAKE_PROFIT_ATR = 60  # 黄金止盈60美元
    OIL_STOP_LOSS = 2      # 原油止损2美元
    OIL_TAKE_PROFIT = 3    # 原油止盈3美元
    SIGNAL_THRESHOLD = 0.50  # 默认置信度50%开仓
    OIL_SIGNAL_THRESHOLD = 0.73  # 原油73%才开仓
    MAX_POSITION = 1.0
    
    # 风控参数
    CIRCUIT_BREAKER_LOSS = 0.10  # 日亏损10%
    MAX_DAILY_TRADES = 50  # 增加到50次
    MAX_DRAWDOWN = 0.10
    
    # 冷静期配置（分钟）
    COOLDOWN_MINUTES = 30
    
    HOLIDAYS = ['01-01', '12-25']
    MODEL_DIR = "saved_models"


# ==================== 工具类 ====================

class DataFetcher:
    def __init__(self, symbol=None):
        self.symbol = symbol or Config.DEFAULT_SYMBOL
        self.connected = False
    
    def connect(self):
        if MT5_AVAILABLE:
            if Config.MT5_LOGIN and Config.MT5_SERVER and Config.MT5_PASSWORD:
                mt5.initialize(login=Config.MT5_LOGIN, server=Config.MT5_SERVER, password=Config.MT5_PASSWORD)
            else:
                mt5.initialize()
            account = mt5.account_info()
            if account is None:
                print("MT5连接失败")
                return False
            print(f"MT5已连接，账户: {account.login}, 余额: {account.balance}")
        self.connected = True
        return True
    
    def disconnect(self):
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False
    
    def get_bars(self, symbol, count=2000):
        if not self.connected:
            return self._generate_dummy_data(count)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, count)
        if rates is None:
            return self._generate_dummy_data(count)
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def _generate_dummy_data(self, count):
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=count, freq='H')
        base_price = 2000
        prices = base_price + np.cumsum(np.random.randn(count) * 5)
        return pd.DataFrame({'time': dates, 'open': prices + np.random.randn(count) * 2, 'high': prices + abs(np.random.randn(count) * 3), 'low': prices - abs(np.random.randn(count) * 3), 'close': prices, 'tick_volume': np.random.randint(100, 1000, count)})


class FeatureEngineer:
    def compute_indicators(self, df):
        df = df.copy()
        close, high, low, volume = df['close'], df['high'], df['low'], df['tick_volume']
        
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / loss))
        
        ema_12, ema_26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        tr_14 = tr.rolling(14).mean()
        df['di_plus'] = 100 * (high - high.shift(1)).apply(lambda x: max(x, 0)).rolling(14).mean() / tr_14
        df['di_minus'] = 100 * (-low + low.shift(1)).apply(lambda x: max(x, 0)).rolling(14).mean() / tr_14
        dx = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = dx.rolling(14).mean()
        
        df['bb_middle'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'], df['bb_lower'] = df['bb_middle'] + 2 * bb_std, df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        df['ema_5'], df['ema_20'], df['ema_50'] = close.ewm(span=5).mean(), close.ewm(span=20).mean(), close.ewm(span=50).mean()
        df['sma_5'], df['sma_20'] = close.rolling(5).mean(), close.rolling(20).mean()
        df['volume_ma'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_ma']
        df['high_low_ratio'], df['close_open_ratio'] = high / low, close / df['open']
        for lag in [1, 3, 5]:
            df[f'returns_{lag}'] = close.pct_change(lag)
        
        return df.fillna(0)
    
    def create_labels(self, df, horizon=5):
        df = df.copy()
        df['future_return'] = df['close'].pct_change(horizon)
        df['label'] = pd.cut(df['future_return'], bins=[-np.inf, -0.003, 0.003, np.inf], labels=[0, 1, 2])
        df['label'] = df['label'].fillna(1).astype(int)
        df = df.drop(columns=['future_return'], errors='ignore')
        # 保留所有标签，不过滤
        return df


# ==================== AI模型 ====================

class XGBPredictor:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_proba(self, X):
        if not self.is_trained:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(self.scaler.transform(X))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True


class LGBPredictor:
    def __init__(self):
        self.model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_proba(self, X):
        if not self.is_trained:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(self.scaler.transform(X))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True


class RFPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_proba(self, X):
        if not self.is_trained:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(self.scaler.transform(X))
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True


class CatBoostPredictor:
    def __init__(self):
        self.model = None
        try:
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, random_state=42, verbose=False)
        except:
            pass
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, X, y):
        if self.model is None:
            return
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict_proba(self, X):
        if not self.is_trained or self.model is None:
            return np.zeros((len(X), 3))
        return self.model.predict_proba(self.scaler.transform(X))
    
    def save(self, path):
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.is_trained = True
        except:
            pass


# ==================== 模型融合 ====================

class EnsembleModel:
    def __init__(self, symbol):
        self.symbol = symbol
        self.weights = Config.MODEL_WEIGHTS
        self.scaler = StandardScaler()
        self.scaler_fitted = False  # 标记scaler是否fit过
        self.is_trained = False
        self.models = {'xgb': XGBPredictor(), 'lgb': LGBPredictor(), 'catboost': CatBoostPredictor(), 'rf': RFPredictor()}
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.scaler_fitted = True
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
        for name, model in self.models.items():
            print(f"训练 {self.symbol} - {name}...")
            try:
                model.train(X_train, y_train)
            except Exception as e:
                print(f"{name} 训练失败: {e}")
        self.is_trained = True
        print(f"{self.symbol} 训练完成!")
        self.save()
    
    def save(self):
        for name, model in self.models.items():
            model.save(f"{Config.MODEL_DIR}/{self.symbol}_{name}.pkl")
        print(f"{self.symbol} 模型已保存")
    
    def load(self):
        all_loaded = True
        for name, model in self.models.items():
            path = f"{Config.MODEL_DIR}/{self.symbol}_{name}.pkl"
            if os.path.exists(path):
                model.load(path)
            else:
                all_loaded = False
        # 重新fit scaler
        if all_loaded:
            import glob
            model_files = glob.glob(f"{Config.MODEL_DIR}/{self.symbol}_*.pkl")
            if model_files:
                try:
                    # 尝试加载scaler
                    for name, model in self.models.items():
                        path = f"{Config.MODEL_DIR}/{self.symbol}_{name}.pkl"
                        if os.path.exists(path):
                            with open(path, 'rb') as f:
                                data = pickle.load(f)
                                if 'scaler' in data:
                                    self.scaler = data['scaler']
                                    break
                except:
                    pass
        self.is_trained = all_loaded
        if all_loaded:
            self.scaler_fitted = True
        return self.is_trained
    
    def predict_proba(self, X):
        # 使用已训练的scaler，不再每次重新fit
        if not self.scaler_fitted:
            self.scaler.fit(X[:min(50, len(X))])
            self.scaler_fitted = True
        
        X_scaled = self.scaler.transform(X)
        ensemble_proba = np.zeros((len(X), 3))
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_scaled)
                if proba.shape[0] == len(X):
                    ensemble_proba += self.weights.get(name, 0.2) * proba
            except Exception as e:
                pass
        
        return ensemble_proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        # 取最后3根K线的平均概率，更稳定
        avg_proba = np.mean(proba[-3:], axis=0)
        return np.argmax(avg_proba)


# ==================== 交易执行 ====================

class TradeExecutor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.position = 0
    
    def open_position(self, action, volume=0.03):
        # 确保MT5已连接
        if MT5_AVAILABLE and not mt5.terminal_info().connected:
            mt5.initialize(login=Config.MT5_LOGIN, server=Config.MT5_SERVER, password=Config.MT5_PASSWORD)
        
        if not MT5_AVAILABLE:
            action_name = {0: "卖出", 2: "买入"}[action]
            print(f"[模拟] {action_name} {self.symbol} {volume}手")
            self.position = volume if action == 2 else -volume
            return True
        
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                print(f"{self.symbol} 品种不存在")
                return False
            if not symbol_info.visible:
                mt5.symbol_select(self.symbol, True)
            
            price = mt5.symbol_info_tick(self.symbol).ask if action == 2 else mt5.symbol_info_tick(self.symbol).bid
            # 根据品种设置止盈止损
            if 'GOLD' in self.symbol:
                sl_price = price - Config.STOP_LOSS_ATR if action == 2 else price + Config.STOP_LOSS_ATR
                tp_price = price + Config.TAKE_PROFIT_ATR if action == 2 else price - Config.TAKE_PROFIT_ATR
            else:  # 原油
                sl_price = price - Config.OIL_STOP_LOSS if action == 2 else price + Config.OIL_STOP_LOSS
                tp_price = price + Config.OIL_TAKE_PROFIT if action == 2 else price - Config.OIL_TAKE_PROFIT
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if action == 2 else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20, "magic": 234000, "comment": "BZ交易系统",
                "type_time": mt5.ORDER_TIME_GTC, "type_filling": 0,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.position = volume if action == 2 else -volume
                # 写入日志文件
                with open('D:/包子交易系统/logs/trade_20260309.log', 'a', encoding='utf-8') as f:
                    f.write(f"[下单成功] {'买入' if action == 2 else '卖出'} {self.symbol} @ {price} SL:{sl_price} TP:{tp_price}\n")
                return True
            else:
                with open('D:/包子交易系统/logs/trade_20260309.log', 'a', encoding='utf-8') as f:
                    f.write(f"[下单失败] retcode={result.retcode} {result.comment}\n")
                return False
        except Exception as e:
            with open('D:/包子交易系统/logs/trade_20260309.log', 'a', encoding='utf-8') as f:
                f.write(f"[下单异常] {e}\n")
            return False
        return False
    
    def close_position(self):
        if not MT5_AVAILABLE:
            print(f"[模拟] 平仓 {self.symbol}")
            self.position = 0
            return True
        if self.position == 0:
            return True
        action = mt5.ORDER_TYPE_SELL if self.position > 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(self.symbol).bid if self.position > 0 else mt5.symbol_info_tick(self.symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": abs(self.position),
            "type": action, "price": price, "deviation": 20, "magic": 234000,
            "comment": "BZ平仓", "type_time": mt5.ORDER_TIME_GTC, "type_filling": 0,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.position = 0
            # 写入平仓日志
            with open('D:/包子交易系统/logs/trade_20260309.log', 'a', encoding='utf-8') as f:
                f.write(f"[平仓成功] {self.symbol} @ {price}\n")
            return True
        return False
    
    def get_position(self):
        if not MT5_AVAILABLE:
            return self.position
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            return sum([p.volume for p in positions])
        return 0


# ==================== 风控 ====================

class RiskManager:
    def __init__(self):
        self.daily_pnl = 0
        self.daily_trades = 0
        self.circuit_breaker = False
        self.last_reset = datetime.now().date()
        self.peak_balance = 0
        self.last_balance = 0  # 上次余额，用于自动计算盈亏
    
    def reset_daily(self):
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_pnl = 0
            self.daily_trades = 0
            self.circuit_breaker = False
            self.last_reset = today
    
    def check(self, balance, pnl=0):
        self.reset_daily()
        # 自动计算余额变化
        if self.last_balance > 0:
            delta = balance - self.last_balance
            if delta != 0:
                self.daily_pnl += delta
        self.last_balance = balance
        
        # 更新峰值余额
        if balance > self.peak_balance:
            self.peak_balance = balance
        if balance > 0 and abs(self.daily_pnl) / balance >= Config.CIRCUIT_BREAKER_LOSS:
            self.circuit_breaker = True
            return False, f"日亏损熔断: {abs(self.daily_pnl)/balance:.2%}"
        if self.daily_trades >= Config.MAX_DAILY_TRADES:
            return False, f"达到每日最大交易次数: {Config.MAX_DAILY_TRADES}"
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - balance) / self.peak_balance
            if drawdown >= Config.MAX_DRAWDOWN:
                return False, f"最大回撤熔断: {drawdown:.2%}"
        return True, ""
    
    def is_holiday(self):
        md = datetime.now().strftime('%m-%d')
        return md in Config.HOLIDAYS or datetime.now().weekday() >= 5


# ==================== 日志 ====================

class Logger:
    def __init__(self):
        os.makedirs('logs', exist_ok=True)
        self.file = f"D:/包子交易系统/logs/trade_{datetime.now().strftime('%Y%m%d')}.log"
    
    def log(self, msg, level="INFO"):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {msg}"
        print(line)
        with open(self.file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


# ==================== 主系统 ====================

class BZTradingSystem:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.ensembles = {}
        self.executors = {}
        self.risk_manager = RiskManager()
        self.logger = Logger()
        self.is_running = False
        
        # 冷静期记录：每个品种的最后下单时间
        self.last_trade_time = {symbol: None for symbol in Config.SYMBOLS}
        # 上次持仓记录，用于检测自动平仓
        self.previous_position = {symbol: 0 for symbol in Config.SYMBOLS}
        
        for symbol in Config.SYMBOLS:
            self.ensembles[symbol] = EnsembleModel(symbol)
            self.executors[symbol] = TradeExecutor(symbol)
    
    def initialize(self):
        self.logger.log("=" * 50)
        self.logger.log("BZ交易系统 1.5.9 启动 (Scaler修复版)")
        self.logger.log(f"交易品种: {Config.SYMBOLS}")
        self.logger.log(f"冷静期: {Config.COOLDOWN_MINUTES}分钟")
        self.logger.log("=" * 50)
        
        if not self.data_fetcher.connect():
            self.logger.log("MT5连接失败，使用模拟模式", "WARNING")
        
        for symbol in Config.SYMBOLS:
            if self.ensembles[symbol].load():
                self.logger.log(f"{symbol}: 已加载已有模型")
            else:
                self.logger.log(f"{symbol}: 将从头训练")
        
        self.logger.log("系统初始化完成")
    
    def get_balance(self):
        if MT5_AVAILABLE:
            account = mt5.account_info()
            if account:
                return account.balance
        return 10000
    
    def check_cooldown(self, symbol):
        """检查是否在冷静期内"""
        last_time = self.last_trade_time.get(symbol)
        if last_time is None:
            return True  # 从未交易过，允许
        
        elapsed = (datetime.now() - last_time).total_seconds() / 60
        if elapsed < Config.COOLDOWN_MINUTES:
            self.logger.log(f"{symbol} 冷静期内，距上次交易 {elapsed:.1f}分钟，需要 {Config.COOLDOWN_MINUTES}分钟")
            return False
        return True
    
    def record_trade(self, symbol):
        """记录交易时间"""
        self.last_trade_time[symbol] = datetime.now()
    
    def process_symbol(self, symbol):
        self.logger.log(f"处理 {symbol}...")
        df = self.data_fetcher.get_bars(symbol, 2000)
        df = self.feature_engineer.compute_indicators(df)
        df = self.feature_engineer.create_labels(df)
        available = [f for f in Config.FEATURES if f in df.columns]
        df = df[available + ['label']].dropna()
        
        if len(df) < Config.MIN_SAMPLES:
            self.logger.log(f"{symbol} 数据不足: {len(df)}")
            return
        
        self.logger.log(f"{symbol} 数据: {len(df)}条")
        
        X = df[Config.FEATURES].values
        y = df['label'].values
        
        # 只在模型未训练时训练（避免每次循环都训练）
        if not self.ensembles[symbol].is_trained:
            self.logger.log(f"{symbol} 训练模型...")
            self.ensembles[symbol].train(X, y)
            # 保存模型
            self.ensembles[symbol].save()
        
        signal = self.ensembles[symbol].predict(X[-10:])
        proba = self.ensembles[symbol].predict_proba(X[-10:])
        confidence = np.max(proba[-1])
        
        self.logger.log(f"{symbol} 信号: {signal}, 置信度: {confidence:.2%}")
        
        # 获取真实持仓（只用一个变量）
        real_position = 0
        try:
            if MT5_AVAILABLE:
                if not mt5.terminal_info().connected:
                    mt5.initialize(login=Config.MT5_LOGIN, server=Config.MT5_SERVER, password=Config.MT5_PASSWORD)
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    real_position = sum([p.volume for p in positions])
                self.logger.log(f"{symbol} 当前持仓: {real_position}")
        except Exception as e:
            self.logger.log(f"{symbol} 检测持仓失败: {e}")
        
        # 检测自动平仓（止盈止损导致）
        if self.previous_position.get(symbol, 0) > 0 and real_position == 0:
            self.logger.log(f"{symbol} 检测到自动平仓，记录冷静期")
            self.record_trade(symbol)
        self.previous_position[symbol] = real_position
        
        # 风控检查
        balance = self.get_balance()
        allow, reason = self.risk_manager.check(balance)
        
        if not allow:
            self.logger.log(f"{symbol} 风控阻止: {reason}")
            if real_position != 0:
                self.executors[symbol].close_position()
                self.record_trade(symbol)
            return
        
        # 信号映射: 0=卖出, 1=持有, 2=买入
        action_map = {0: "卖出", 1: "持有", 2: "买入"}
        
        # 根据品种使用不同的置信度阈值
        if 'OIL' in symbol:
            threshold = Config.OIL_SIGNAL_THRESHOLD
        else:
            threshold = Config.SIGNAL_THRESHOLD
        
        # 无论置信度是否足够，都检查持仓
        has_position = real_position != 0
        
        if confidence > threshold:
            signal_name = action_map.get(signal, "未知")
            self.logger.log(f"{symbol} 信号: {signal_name} | 置信度: {confidence:.2%}")
            
            if has_position:
                # 有持仓时检查方向
                current_direction = 0 if real_position > 0 else 1  # 0=做多,1=做空
                if signal != current_direction + 2 and signal != current_direction:
                    # 信号方向和持仓相反，平仓
                    self.logger.log(f"{symbol} 信号方向与持仓相反，平仓")
                    self.executors[symbol].close_position()
                    self.record_trade(symbol)
            
            # 没有持仓时才开仓
            if real_position == 0:
                # 检查冷静期
                if not self.check_cooldown(symbol):
                    self.logger.log(f"{symbol} 冷静期阻止开仓")
                else:
                    if signal == 2:  # 买入信号
                        self.logger.log(f"{symbol} 执行买入!")
                        success = self.executors[symbol].open_position(2)
                        if success:
                            self.risk_manager.daily_trades += 1
                            self.record_trade(symbol)
                    elif signal == 0:  # 卖出信号
                        self.logger.log(f"{symbol} 执行卖出!")
                        success = self.executors[symbol].open_position(0)
                        if success:
                            self.risk_manager.daily_trades += 1
                            self.record_trade(symbol)
        else:
            # 置信度不够，但有持仓时也检查是否需要平仓
            if has_position:
                self.logger.log(f"{symbol} 置信度不足，但有持仓，继续持有")
                self.logger.log(f"{symbol} 已有持仓 {real_position}，跳过开仓")
    
    def run(self):
        self.initialize()
        self.is_running = True
        self.logger.log("开始交易循环...")
        
        while self.is_running:
            try:
                if self.risk_manager.is_holiday():
                    self.logger.log("节假日休市")
                    time.sleep(3600)
                    continue
                
                for symbol in Config.SYMBOLS:
                    self.process_symbol(symbol)
                
                time.sleep(60)
                
            except KeyboardInterrupt:
                self.logger.log("用户中断")
                break
            except Exception as e:
                self.logger.log(f"错误: {e}", "ERROR")
                time.sleep(60)
        
        self.shutdown()
    
    def shutdown(self):
        self.is_running = False
        for symbol in Config.SYMBOLS:
            if self.ensembles[symbol].is_trained:
                self.ensembles[symbol].save()
        self.data_fetcher.disconnect()
        self.logger.log("系统已关闭")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║        BZ交易系统 1.5.7 - 逻辑优化版                     ║
    ║     XGB + LGB + RF + CatBoost + 30分钟冷静期          ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    system = BZTradingSystem()
    system.run()
