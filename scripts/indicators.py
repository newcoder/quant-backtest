#!/usr/bin/env python3
"""
完整技术指标库 - 支持参数空间并行搜索
包含20+常用技术指标的多种实现（Pandas/Numba/GPU）
"""

import numpy as np
import pandas as pd
from numba import njit, prange, cuda
from typing import Dict, List, Tuple, Union, Callable
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


# ============== 基础工具函数 ==============

@njit(cache=True)
def rolling_mean_numba(arr, window):
    """Numba滚动平均"""
    n = len(arr)
    result = np.empty(n)
    for i in range(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(arr[i-window+1:i+1])
    return result

@njit(cache=True)
def rolling_std_numba(arr, window):
    """Numba滚动标准差"""
    n = len(arr)
    result = np.empty(n)
    for i in range(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.std(arr[i-window+1:i+1])
    return result

@njit(cache=True)
def ema_numba(arr, span):
    """Numba指数移动平均"""
    n = len(arr)
    result = np.empty(n)
    alpha = 2.0 / (span + 1)
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result


# ============== 趋势指标 ==============

class MovingAverages:
    """移动平均线家族"""
    
    @staticmethod
    @njit(cache=True, parallel=True)
    def sma(close: np.ndarray, window: int) -> np.ndarray:
        """简单移动平均"""
        return rolling_mean_numba(close, window)
    
    @staticmethod
    @njit(cache=True)
    def ema(close: np.ndarray, span: int) -> np.ndarray:
        """指数移动平均"""
        return ema_numba(close, span)
    
    @staticmethod
    @njit(cache=True)
    def wma(close: np.ndarray, window: int) -> np.ndarray:
        """加权移动平均"""
        n = len(close)
        result = np.empty(n)
        weights = np.arange(1, window + 1)
        weight_sum = np.sum(weights)
        
        for i in range(n):
            if i < window - 1:
                result[i] = np.nan
            else:
                result[i] = np.sum(close[i-window+1:i+1] * weights) / weight_sum
        return result
    
    @staticmethod
    @njit(cache=True)
    def dema(close: np.ndarray, span: int) -> np.ndarray:
        """双指数移动平均 DEMA = 2*EMA - EMA(EMA)"""
        ema1 = ema_numba(close, span)
        ema2 = ema_numba(ema1, span)
        return 2 * ema1 - ema2
    
    @staticmethod
    @njit(cache=True)
    def tema(close: np.ndarray, span: int) -> np.ndarray:
        """三指数移动平均 TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))"""
        ema1 = ema_numba(close, span)
        ema2 = ema_numba(ema1, span)
        ema3 = ema_numba(ema2, span)
        return 3 * ema1 - 3 * ema2 + ema3
    
    @staticmethod
    @njit(cache=True)
    def kama(close: np.ndarray, er_period: int = 10, fast_ema: int = 2, slow_ema: int = 30) -> np.ndarray:
        """考夫曼自适应移动平均 KAMA"""
        n = len(close)
        kama = np.empty(n)
        kama[0] = close[0]
        
        for i in range(1, n):
            if i < er_period:
                kama[i] = close[i]
            else:
                change = abs(close[i] - close[i-er_period])
                volatility = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
                er = change / (volatility + 1e-10)
                
                fast_sc = 2.0 / (fast_ema + 1)
                slow_sc = 2.0 / (slow_ema + 1)
                sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                
                kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
        
        return kama


class MACDIndicator:
    """MACD指标家族"""
    
    @staticmethod
    @njit(cache=True)
    def macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """标准MACD"""
        ema_fast = ema_numba(close, fast)
        ema_slow = ema_numba(close, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema_numba(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    @njit(cache=True)
    def macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
        """MACD柱状图"""
        _, _, hist = MACDIndicator.macd(close, fast, slow, signal)
        return hist


# ============== 动量指标 ==============

class MomentumIndicators:
    """动量指标家族"""
    
    @staticmethod
    @njit(cache=True)
    def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        """相对强弱指数 RSI"""
        n = len(close)
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(n)
        avg_losses = np.zeros(n)
        
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, n):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stoch_rsi(close: np.ndarray, period: int = 14, stoch_period: int = 14) -> np.ndarray:
        """随机RSI"""
        rsi_vals = MomentumIndicators.rsi(close, period)
        n = len(rsi_vals)
        stoch_rsi = np.empty(n)
        
        for i in range(n):
            if i < stoch_period - 1:
                stoch_rsi[i] = np.nan
            else:
                window = rsi_vals[i-stoch_period+1:i+1]
                min_rsi = np.min(window)
                max_rsi = np.max(window)
                if max_rsi - min_rsi > 0:
                    stoch_rsi[i] = (rsi_vals[i] - min_rsi) / (max_rsi - min_rsi)
                else:
                    stoch_rsi[i] = 0.5
        
        return stoch_rsi * 100
    
    @staticmethod
    @njit(cache=True)
    def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
        """商品通道指数 CCI"""
        n = len(close)
        tp = (high + low + close) / 3
        cci = np.empty(n)
        
        for i in range(n):
            if i < period - 1:
                cci[i] = np.nan
            else:
                sma_tp = np.mean(tp[i-period+1:i+1])
                mean_dev = np.mean(np.abs(tp[i-period+1:i+1] - sma_tp))
                cci[i] = (tp[i] - sma_tp) / (0.015 * mean_dev + 1e-10)
        
        return cci
    
    @staticmethod
    @njit(cache=True)
    def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """威廉指标 Williams %R"""
        n = len(close)
        wr = np.empty(n)
        
        for i in range(n):
            if i < period - 1:
                wr[i] = np.nan
            else:
                highest_high = np.max(high[i-period+1:i+1])
                lowest_low = np.min(low[i-period+1:i+1])
                wr[i] = (highest_high - close[i]) / (highest_high - lowest_low + 1e-10) * -100
        
        return wr
    
    @staticmethod
    @njit(cache=True)
    def ultimate_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           period1: int = 7, period2: int = 14, period3: int = 28) -> np.ndarray:
        """终极震荡指标 Ultimate Oscillator"""
        n = len(close)
        bp = close - np.minimum(low, np.roll(close, 1))
        tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
        
        uo = np.empty(n)
        for i in range(n):
            if i < period3 - 1:
                uo[i] = np.nan
            else:
                avg1 = np.sum(bp[i-period1+1:i+1]) / (np.sum(tr[i-period1+1:i+1]) + 1e-10)
                avg2 = np.sum(bp[i-period2+1:i+1]) / (np.sum(tr[i-period2+1:i+1]) + 1e-10)
                avg3 = np.sum(bp[i-period3+1:i+1]) / (np.sum(tr[i-period3+1:i+1]) + 1e-10)
                uo[i] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        
        return uo


# ============== 波动率指标 ==============

class VolatilityIndicators:
    """波动率指标家族"""
    
    @staticmethod
    @njit(cache=True)
    def bollinger_bands(close: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """布林带 Bollinger Bands"""
        middle = rolling_mean_numba(close, window)
        std = rolling_std_numba(close, window)
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower
    
    @staticmethod
    @njit(cache=True)
    def bb_width(close: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
        """布林带宽度"""
        upper, middle, lower = VolatilityIndicators.bollinger_bands(close, window, num_std)
        return (upper - lower) / (middle + 1e-10)
    
    @staticmethod
    @njit(cache=True)
    def bb_percent_b(close: np.ndarray, window: int = 20, num_std: float = 2.0) -> np.ndarray:
        """%B指标"""
        upper, middle, lower = VolatilityIndicators.bollinger_bands(close, window, num_std)
        return (close - lower) / (upper - lower + 1e-10)
    
    @staticmethod
    @njit(cache=True)
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """平均真实波幅 ATR"""
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0]
        
        for i in range(1, n):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr[i] = max(tr1, max(tr2, tr3))
        
        return ema_numba(tr, period)
    
    @staticmethod
    def keltner_channels(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                        ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """肯特纳通道 Keltner Channels"""
        middle = ema_numba(close, ema_period)
        atr_val = VolatilityIndicators.atr(high, low, close, atr_period)
        upper = middle + multiplier * atr_val
        lower = middle - multiplier * atr_val
        return upper, middle, lower
    
    @staticmethod
    @njit(cache=True)
    def donchian_channels(high: np.ndarray, low: np.ndarray, period: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """唐奇安通道 Donchian Channels"""
        n = len(high)
        upper = np.empty(n)
        middle = np.empty(n)
        lower = np.empty(n)
        
        for i in range(n):
            if i < period - 1:
                upper[i] = middle[i] = lower[i] = np.nan
            else:
                upper[i] = np.max(high[i-period+1:i+1])
                lower[i] = np.min(low[i-period+1:i+1])
                middle[i] = (upper[i] + lower[i]) / 2
        
        return upper, middle, lower


# ============== 成交量指标 ==============

class VolumeIndicators:
    """成交量指标家族"""
    
    @staticmethod
    @njit(cache=True)
    def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """能量潮 OBV"""
        n = len(close)
        obv = np.empty(n)
        obv[0] = volume[0]
        
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    @staticmethod
    @njit(cache=True)
    def vwma(close: np.ndarray, volume: np.ndarray, window: int = 20) -> np.ndarray:
        """成交量加权移动平均 VWMA"""
        n = len(close)
        vwma = np.empty(n)
        
        for i in range(n):
            if i < window - 1:
                vwma[i] = np.nan
            else:
                vol_sum = np.sum(volume[i-window+1:i+1])
                if vol_sum > 0:
                    vwma[i] = np.sum(close[i-window+1:i+1] * volume[i-window+1:i+1]) / vol_sum
                else:
                    vwma[i] = close[i]
        
        return vwma
    
    @staticmethod
    @njit(cache=True)
    def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """成交量加权平均价 VWAP (日内)"""
        tp = (high + low + close) / 3
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        return cum_tp_vol / (cum_vol + 1e-10)
    
    @staticmethod
    @njit(cache=True)
    def mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
        """资金流量指标 MFI"""
        n = len(close)
        tp = (high + low + close) / 3
        raw_money_flow = tp * volume
        
        positive_flow = np.zeros(n)
        negative_flow = np.zeros(n)
        
        for i in range(1, n):
            if tp[i] > tp[i-1]:
                positive_flow[i] = raw_money_flow[i]
            else:
                negative_flow[i] = raw_money_flow[i]
        
        mfi = np.empty(n)
        for i in range(n):
            if i < period - 1:
                mfi[i] = np.nan
            else:
                pos_sum = np.sum(positive_flow[i-period+1:i+1])
                neg_sum = np.sum(negative_flow[i-period+1:i+1])
                if neg_sum > 0:
                    mfi[i] = 100 - (100 / (1 + pos_sum / neg_sum))
                else:
                    mfi[i] = 100
        
        return mfi
    
    @staticmethod
    @njit(cache=True)
    def ad_line(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """累积/派发线 A/D Line"""
        n = len(close)
        ad = np.empty(n)
        
        for i in range(n):
            if high[i] == low[i]:
                mf_multiplier = 0
            else:
                mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
            mf_volume = mf_multiplier * volume[i]
            
            if i == 0:
                ad[i] = mf_volume
            else:
                ad[i] = ad[i-1] + mf_volume
        
        return ad


# ============== 参数空间搜索 ==============

@dataclass
class IndicatorConfig:
    """指标配置"""
    name: str
    func: Callable
    param_ranges: Dict[str, Tuple[int, int, int]]  # {param: (min, max, step)}
    required_data: List[str]  # ['close'], ['high', 'low', 'close'], etc.


class ParameterSpaceSearch:
    """参数空间并行搜索"""
    
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)
    
    def generate_param_combinations(self, param_ranges: Dict[str, Tuple]) -> List[Dict]:
        """生成参数组合"""
        keys = list(param_ranges.keys())
        ranges = []
        
        for key in keys:
            min_val, max_val, step = param_ranges[key]
            ranges.append(range(min_val, max_val + 1, step))
        
        combinations = []
        for values in itertools.product(*ranges):
            combo = dict(zip(keys, values))
            combinations.append(combo)
        
        return combinations
    
    def search_single_indicator(self, 
                                data: Dict[str, np.ndarray],
                                config: IndicatorConfig,
                                score_fn: Callable[[np.ndarray], float]) -> List[Dict]:
        """
        对单个指标进行参数空间搜索
        
        Args:
            data: {'close': array, 'high': array, ...}
            config: 指标配置
            score_fn: 评分函数，接收指标值返回分数
        
        Returns:
            按分数排序的参数组合列表
        """
        param_combos = self.generate_param_combinations(config.param_ranges)
        
        def evaluate_params(params):
            try:
                # 提取需要的数据
                args = [data[field] for field in config.required_data]
                
                # 计算指标
                result = config.func(*args, **params)
                
                # 计算分数（跳过NaN）
                valid_result = result[~np.isnan(result)]
                if len(valid_result) > 0:
                    score = score_fn(valid_result)
                else:
                    score = -np.inf
                
                return {**params, 'score': score, 'indicator': config.name}
            except Exception as e:
                return {**params, 'score': -np.inf, 'error': str(e), 'indicator': config.name}
        
        # 并行评估
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(evaluate_params, combo): combo for combo in param_combos}
            for future in as_completed(futures):
                results.append(future.result())
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def search_multiple_indicators(self,
                                   data: Dict[str, np.ndarray],
                                   configs: List[IndicatorConfig],
                                   score_fn: Callable[[np.ndarray], float]) -> Dict[str, List[Dict]]:
        """
        并行搜索多个指标的参数空间
        
        每个指标视为独立实例，可以并行处理
        """
        all_results = {}
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            
            for config in configs:
                future = executor.submit(
                    self.search_single_indicator,
                    data,
                    config,
                    score_fn
                )
                futures[future] = config.name
            
            for future in as_completed(futures):
                indicator_name = futures[future]
                all_results[indicator_name] = future.result()
        
        return all_results


# ============== 使用示例 ==============

def example_parameter_search():
    """参数搜索示例"""
    # 生成测试数据
    np.random.seed(42)
    n = 1000
    close = np.cumsum(np.random.randn(n) * 0.02) + 100
    high = close * (1 + abs(np.random.randn(n) * 0.01))
    low = close * (1 - abs(np.random.randn(n) * 0.01))
    volume = np.random.randint(1000000, 10000000, n)
    
    data = {'close': close, 'high': high, 'low': low, 'volume': volume}
    
    # 定义指标配置
    configs = [
        IndicatorConfig(
            name='RSI',
            func=MomentumIndicators.rsi,
            param_ranges={'period': (5, 30, 5)},
            required_data=['close']
        ),
        IndicatorConfig(
            name='CCI',
            func=MomentumIndicators.cci,
            param_ranges={'period': (10, 40, 5)},
            required_data=['high', 'low', 'close']
        ),
        IndicatorConfig(
            name='BollingerBands',
            func=VolatilityIndicators.bollinger_bands,
            param_ranges={'window': (10, 50, 10), 'num_std': (1, 3, 1)},
            required_data=['close']
        ),
    ]
    
    # 评分函数：指标波动率
    def volatility_score(values):
        return np.std(values)
    
    # 并行搜索
    searcher = ParameterSpaceSearch(n_jobs=4)
    results = searcher.search_multiple_indicators(data, configs, volatility_score)
    
    # 输出结果
    for indicator_name, param_results in results.items():
        print(f"\n{indicator_name} 最优参数:")
        for i, result in enumerate(param_results[:3]):
            print(f"  {i+1}. {result}")


if __name__ == '__main__':
    import os
    example_parameter_search()