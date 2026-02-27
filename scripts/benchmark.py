#!/usr/bin/env python3
"""
性能基准测试脚本
对比不同实现方式的性能差异
"""

import numpy as np
import pandas as pd
import time
from numba import njit, prange

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CuPy not installed, GPU tests will be skipped")


# ============== 测试数据生成 ==============

def generate_test_data(n_stocks=3000, n_days=500, freq='1d'):
    """生成测试数据"""
    np.random.seed(42)
    
    if freq == '1m':
        n_periods = n_days * 240  # 每天240分钟
    else:
        n_periods = n_days
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq=freq.upper())
    stocks = [f'ST{i:04d}' for i in range(n_stocks)]
    
    data = {}
    for stock in stocks:
        # 随机游走生成价格
        returns = np.random.normal(0.0001, 0.02, n_periods)
        close = 100 * np.exp(np.cumsum(returns))
        
        data[stock] = pd.DataFrame({
            'open': close * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': close * (1 + abs(np.random.normal(0, 0.01, n_periods))),
            'low': close * (1 - abs(np.random.normal(0, 0.01, n_periods))),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, n_periods)
        }, index=dates)
    
    return data


# ============== 技术指标实现 ==============

def sma_python(prices, window):
    """纯Python实现SMA"""
    result = []
    for i in range(len(prices)):
        if i < window - 1:
            result.append(np.nan)
        else:
            result.append(sum(prices[i-window+1:i+1]) / window)
    return np.array(result)

def sma_pandas(prices, window):
    """Pandas实现SMA"""
    pd_series = pd.Series(prices)
    return pd_series.rolling(window=window).mean().values

@njit(cache=True, parallel=True)
def sma_numba(prices, window):
    """Numba加速SMA"""
    n = len(prices)
    result = np.empty(n)
    for i in prange(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(prices[i-window+1:i+1])
    return result

def sma_gpu(prices, window):
    """GPU加速SMA"""
    if not HAS_GPU:
        return None
    
    prices_gpu = cp.asarray(prices)
    n = len(prices)
    result = cp.empty(n)
    
    # 使用卷积计算移动平均
    kernel = cp.ones(window) / window
    result[window-1:] = cp.convolve(prices_gpu, kernel, mode='valid')
    result[:window-1] = cp.nan
    
    return cp.asnumpy(result)


def rsi_python(prices, period=14):
    """纯Python实现RSI"""
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = []
    avg_losses = []
    
    for i in range(len(prices)):
        if i < period:
            avg_gains.append(np.nan)
            avg_losses.append(np.nan)
        elif i == period:
            avg_gains.append(np.mean(gains[:period]))
            avg_losses.append(np.mean(losses[:period]))
        else:
            avg_gains.append((avg_gains[-1] * (period-1) + gains[i-1]) / period)
            avg_losses.append((avg_losses[-1] * (period-1) + losses[i-1]) / period)
    
    avg_gains = np.array(avg_gains)
    avg_losses = np.array(avg_losses)
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

@njit(cache=True)
def rsi_numba(prices, period=14):
    """Numba加速RSI"""
    n = len(prices)
    deltas = np.diff(prices)
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


# ============== 回测引擎 ==============

class VectorizedBacktest:
    """向量化回测引擎"""
    
    def __init__(self, prices, signals, initial_capital=1e6, commission=0.001):
        self.prices = prices
        self.signals = signals
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self):
        returns = self.prices.pct_change().fillna(0)
        position = self.signals.shift(1).fillna(0)
        strategy_returns = (position * returns).sum(axis=1)
        
        turnover = position.diff().abs().sum(axis=1)
        costs = turnover * self.commission
        
        net_returns = strategy_returns - costs
        cumulative = (1 + net_returns).cumprod()
        
        return {
            'cumulative_returns': cumulative,
            'returns': net_returns,
            'sharpe': self._calc_sharpe(net_returns),
            'max_drawdown': self._calc_drawdown(cumulative)
        }
    
    def _calc_sharpe(self, returns, risk_free=0.02):
        excess = returns - risk_free / 252
        return excess.mean() / (excess.std() + 1e-10) * np.sqrt(252)
    
    def _calc_drawdown(self, cumulative):
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()


# ============== 基准测试 ==============

def benchmark_indicator(indicator_fn, prices, name, iterations=10):
    """基准测试单个指标"""
    times = []
    
    # 预热（Numba编译）
    if 'numba' in name:
        indicator_fn(prices, 20)
    
    for _ in range(iterations):
        start = time.time()
        result = indicator_fn(prices, 20)
        elapsed = time.time() - start
        times.append(elapsed)
    
    avg_time = np.mean(times) * 1000  # 转换为毫秒
    std_time = np.std(times) * 1000
    
    return avg_time, std_time


def benchmark_backtest(data, name):
    """基准测试回测引擎"""
    prices = pd.DataFrame({k: v['close'] for k, v in data.items()})
    
    # 生成简单信号
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    for col in prices.columns:
        ma20 = prices[col].rolling(20).mean()
        signals[col] = (prices[col] > ma20).astype(int)
    
    start = time.time()
    engine = VectorizedBacktest(prices, signals)
    results = engine.run()
    elapsed = time.time() - start
    
    return elapsed * 1000, results


def run_benchmarks():
    """运行所有基准测试"""
    print("=" * 60)
    print("高性能回测系统 - 性能基准测试")
    print("=" * 60)
    
    # 生成测试数据
    print("\n[1] 生成测试数据...")
    data = generate_test_data(n_stocks=100, n_days=500)
    sample_prices = list(data.values())[0]['close'].values
    print(f"    单只股票数据点: {len(sample_prices)}")
    print(f"    股票数量: {len(data)}")
    
    # 指标计算基准测试
    print("\n[2] 技术指标计算性能对比")
    print("-" * 60)
    
    implementations = [
        ('Python Loop', sma_python),
        ('Pandas', sma_pandas),
        ('Numba JIT', sma_numba),
    ]
    
    if HAS_GPU:
        implementations.append(('CUDA GPU', sma_gpu))
    
    baseline_time = None
    for name, fn in implementations:
        try:
            avg_time, std_time = benchmark_indicator(fn, sample_prices, name)
            
            if baseline_time is None:
                baseline_time = avg_time
            
            speedup = baseline_time / avg_time if avg_time > 0 else float('inf')
            print(f"{name:15s}: {avg_time:8.2f} ± {std_time:6.2f} ms (加速: {speedup:6.1f}x)")
        except Exception as e:
            print(f"{name:15s}: 失败 - {e}")
    
    # RSI测试
    print("\n[3] RSI计算性能对比")
    print("-" * 60)
    
    rsi_impls = [
        ('Python', rsi_python),
        ('Numba', rsi_numba),
    ]
    
    baseline_time = None
    for name, fn in rsi_impls:
        try:
            avg_time, std_time = benchmark_indicator(fn, sample_prices, name)
            
            if baseline_time is None:
                baseline_time = avg_time
            
            speedup = baseline_time / avg_time if avg_time > 0 else float('inf')
            print(f"{name:15s}: {avg_time:8.2f} ± {std_time:6.2f} ms (加速: {speedup:6.1f}x)")
        except Exception as e:
            print(f"{name:15s}: 失败 - {e}")
    
    # 回测引擎基准测试
    print("\n[4] 向量化回测性能")
    print("-" * 60)
    
    elapsed, results = benchmark_backtest(data, "Vectorized")
    print(f"回测耗时: {elapsed:.2f} ms")
    print(f"夏普比率: {results['sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    
    # 大规模测试
    print("\n[5] 大规模数据测试 (3000只股票)")
    print("-" * 60)
    
    large_data = generate_test_data(n_stocks=3000, n_days=252)
    elapsed, _ = benchmark_backtest(large_data, "Large Scale")
    print(f"3000只股票年数据回测: {elapsed:.2f} ms ({elapsed/1000:.2f} 秒)")
    
    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == '__main__':
    run_benchmarks()