#!/usr/bin/env python3
"""
技术指标性能对比工具
对比NumPy/Pandas/Numba/CUDA实现
"""

import numpy as np
import pandas as pd
import time
from numba import njit, prange, cuda
from functools import wraps

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 预热
        if 'numba' in func.__name__ or 'cuda' in func.__name__:
            func(*args, **kwargs)
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times[1:]) * 1000  # 排除第一次，转换为ms
        return result, avg_time
    return wrapper


# ============== MACD实现 ==============

def macd_pandas(prices, fast=12, slow=26, signal=9):
    """Pandas实现MACD"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

@njit(cache=True)
def macd_numba(prices, fast=12, slow=26, signal=9):
    """Numba实现MACD"""
    n = len(prices)
    
    # 计算EMA
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    alpha_signal = 2.0 / (signal + 1)
    
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    macd_line = np.zeros(n)
    signal_line = np.zeros(n)
    histogram = np.zeros(n)
    
    ema_fast[0] = prices[0]
    ema_slow[0] = prices[0]
    
    for i in range(1, n):
        ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
        ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
        macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line[0] = macd_line[0]
    for i in range(1, n):
        signal_line[i] = alpha_signal * macd_line[i] + (1 - alpha_signal) * signal_line[i-1]
        histogram[i] = macd_line[i] - signal_line[i]
    
    return macd_line, signal_line, histogram

@timer
def macd_numba_timed(prices, fast=12, slow=26, signal=9):
    return macd_numba(prices, fast, slow, signal)


def macd_gpu(prices, fast=12, slow=26, signal=9):
    """GPU实现MACD"""
    if not HAS_GPU:
        return None
    
    prices_gpu = cp.asarray(prices)
    n = len(prices)
    
    # GPU EMA计算
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    alpha_signal = 2.0 / (signal + 1)
    
    ema_fast = cp.zeros(n)
    ema_slow = cp.zeros(n)
    macd_line = cp.zeros(n)
    signal_line = cp.zeros(n)
    histogram = cp.zeros(n)
    
    # 使用scan操作计算EMA
    ema_fast = cp.ewm(prices_gpu, alpha=alpha_fast)
    ema_slow = cp.ewm(prices_gpu, alpha=alpha_slow)
    macd_line = ema_fast - ema_slow
    signal_line = cp.ewm(macd_line, alpha=alpha_signal)
    histogram = macd_line - signal_line
    
    return cp.asnumpy(macd_line), cp.asnumpy(signal_line), cp.asnumpy(histogram)


# ============== Bollinger Bands ==============

def bollinger_pandas(prices, window=20, num_std=2):
    """Pandas实现布林带"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower

@njit(cache=True, parallel=True)
def bollinger_numba(prices, window=20, num_std=2):
    """Numba实现布林带"""
    n = len(prices)
    ma = np.zeros(n)
    std = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in prange(window - 1, n):
        window_data = prices[i-window+1:i+1]
        ma[i] = np.mean(window_data)
        std[i] = np.std(window_data)
        upper[i] = ma[i] + num_std * std[i]
        lower[i] = ma[i] - num_std * std[i]
    
    return upper, ma, lower


# ============== 性能测试 ==============

def compare_implementations():
    """对比不同实现"""
    print("=" * 70)
    print("技术指标性能对比")
    print("=" * 70)
    
    # 生成数据
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print(f"\n数据规模: {size:,} 数据点")
        print("-" * 70)
        
        prices = np.random.randn(size).cumsum() + 100
        
        # MACD测试
        print("\nMACD计算:")
        
        # Pandas
        pd_prices = pd.Series(prices)
        start = time.perf_counter()
        for _ in range(10):
            macd_pandas(pd_prices)
        pandas_time = (time.perf_counter() - start) / 10 * 1000
        print(f"  Pandas:  {pandas_time:8.3f} ms")
        
        # Numba
        _, numba_time = macd_numba_timed(prices)
        print(f"  Numba:   {numba_time:8.3f} ms (加速 {pandas_time/numba_time:.1f}x)")
        
        # GPU
        if HAS_GPU:
            start = time.perf_counter()
            for _ in range(10):
                macd_gpu(prices)
            gpu_time = (time.perf_counter() - start) / 10 * 1000
            print(f"  GPU:     {gpu_time:8.3f} ms (加速 {pandas_time/gpu_time:.1f}x)")
        
        # Bollinger Bands测试
        print("\n布林带计算:")
        
        # Pandas
        start = time.perf_counter()
        for _ in range(10):
            bollinger_pandas(pd_prices)
        pandas_time = (time.perf_counter() - start) / 10 * 1000
        print(f"  Pandas:  {pandas_time:8.3f} ms")
        
        # Numba
        start = time.perf_counter()
        for _ in range(10):
            bollinger_numba(prices)
        numba_time = (time.perf_counter() - start) / 10 * 1000
        print(f"  Numba:   {numba_time:8.3f} ms (加速 {pandas_time/numba_time:.1f}x)")


def batch_benchmark():
    """批量股票指标计算测试"""
    print("\n" + "=" * 70)
    print("批量股票指标计算 (1000只股票，各500日数据)")
    print("=" * 70)
    
    n_stocks = 1000
    n_days = 500
    
    # 生成数据
    all_prices = np.random.randn(n_stocks, n_days).cumsum(axis=1) + 100
    
    # 逐只计算（Python循环）
    print("\n逐只计算 (Python循环):")
    start = time.perf_counter()
    for i in range(n_stocks):
        macd_numba(all_prices[i])
    loop_time = (time.perf_counter() - start) * 1000
    print(f"  耗时: {loop_time:.2f} ms")
    
    # GPU批量计算
    if HAS_GPU:
        print("\nGPU批量计算:")
        prices_gpu = cp.asarray(all_prices)
        
        start = time.perf_counter()
        # 并行计算所有股票的MACD
        for i in range(n_stocks):
            macd_gpu(cp.asnumpy(prices_gpu[i]))
        gpu_time = (time.perf_counter() - start) * 1000
        print(f"  耗时: {gpu_time:.2f} ms (加速 {loop_time/gpu_time:.1f}x)")


if __name__ == '__main__':
    compare_implementations()
    batch_benchmark()