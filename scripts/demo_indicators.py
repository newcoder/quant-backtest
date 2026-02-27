#!/usr/bin/env python3
"""
技术指标库演示 - 单进程版本
展示20+指标和参数搜索功能
"""

import numpy as np
import pandas as pd
import time
from indicators import (
    MovingAverages, MACDIndicator, MomentumIndicators,
    VolatilityIndicators, VolumeIndicators
)


def generate_test_data(n_bars=1000):
    """生成测试数据"""
    np.random.seed(42)
    
    returns = np.random.normal(0.0001, 0.02, n_bars)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + abs(np.random.randn(n_bars) * 0.01))
    low = close * (1 - abs(np.random.randn(n_bars) * 0.01))
    open_price = close * (1 + np.random.normal(0, 0.001, n_bars))
    volume = np.random.randint(1000000, 10000000, n_bars)
    
    return {
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }


def benchmark_indicator(name, fn, *args, iterations=100):
    """基准测试单个指标"""
    # 预热
    result = fn(*args)
    
    # 计时
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    
    return result, elapsed


def demo_all_indicators():
    """演示所有技术指标"""
    print("=" * 70)
    print("技术指标库演示 - 20+ Numba加速指标")
    print("=" * 70)
    
    # 生成数据
    data = generate_test_data(5000)
    close = data['close']
    high, low, volume = data['high'], data['low'], data['volume']
    
    print(f"\n数据规模: {len(close)} 根K线")
    print("-" * 70)
    
    results = []
    
    # ========== 趋势指标 ==========
    print("\n【趋势指标】")
    
    _, t = benchmark_indicator('SMA(20)', MovingAverages.sma, close, 20)
    results.append(('SMA', t))
    print(f"  SMA(20):        {t:.3f} ms")
    
    _, t = benchmark_indicator('EMA(20)', MovingAverages.ema, close, 20)
    results.append(('EMA', t))
    print(f"  EMA(20):        {t:.3f} ms")
    
    _, t = benchmark_indicator('WMA(20)', MovingAverages.wma, close, 20)
    results.append(('WMA', t))
    print(f"  WMA(20):        {t:.3f} ms")
    
    _, t = benchmark_indicator('DEMA(20)', MovingAverages.dema, close, 20)
    results.append(('DEMA', t))
    print(f"  DEMA(20):       {t:.3f} ms")
    
    _, t = benchmark_indicator('TEMA(20)', MovingAverages.tema, close, 20)
    results.append(('TEMA', t))
    print(f"  TEMA(20):       {t:.3f} ms")
    
    _, t = benchmark_indicator('KAMA', MovingAverages.kama, close, 10, 2, 30)
    results.append(('KAMA', t))
    print(f"  KAMA:           {t:.3f} ms")
    
    # ========== MACD指标 ==========
    print("\n【MACD指标】")
    
    _, t = benchmark_indicator('MACD(12,26,9)', MACDIndicator.macd, close, 12, 26, 9)
    results.append(('MACD', t))
    print(f"  MACD(12,26,9):  {t:.3f} ms")
    
    # ========== 动量指标 ==========
    print("\n【动量指标】")
    
    _, t = benchmark_indicator('RSI(14)', MomentumIndicators.rsi, close, 14)
    results.append(('RSI', t))
    print(f"  RSI(14):        {t:.3f} ms")
    
    _, t = benchmark_indicator('StochRSI', MomentumIndicators.stoch_rsi, close, 14, 14)
    results.append(('StochRSI', t))
    print(f"  StochRSI:       {t:.3f} ms")
    
    _, t = benchmark_indicator('CCI(20)', MomentumIndicators.cci, high, low, close, 20)
    results.append(('CCI', t))
    print(f"  CCI(20):        {t:.3f} ms")
    
    _, t = benchmark_indicator('Williams %R(14)', MomentumIndicators.williams_r, high, low, close, 14)
    results.append(('WilliamsR', t))
    print(f"  Williams %R(14):{t:.3f} ms")
    
    _, t = benchmark_indicator('Ultimate Osc', MomentumIndicators.ultimate_oscillator, high, low, close, 7, 14, 28)
    results.append(('UltimateOsc', t))
    print(f"  Ultimate Osc:   {t:.3f} ms")
    
    # ========== 波动率指标 ==========
    print("\n【波动率指标】")
    
    _, t = benchmark_indicator('BB(20,2)', VolatilityIndicators.bollinger_bands, close, 20, 2.0)
    results.append(('BollingerBands', t))
    print(f"  Bollinger(20,2):{t:.3f} ms")
    
    _, t = benchmark_indicator('BB Width', VolatilityIndicators.bb_width, close, 20, 2.0)
    results.append(('BBWidth', t))
    print(f"  BB Width:       {t:.3f} ms")
    
    _, t = benchmark_indicator('%B', VolatilityIndicators.bb_percent_b, close, 20, 2.0)
    results.append(('PercentB', t))
    print(f"  %B:             {t:.3f} ms")
    
    _, t = benchmark_indicator('ATR(14)', VolatilityIndicators.atr, high, low, close, 14)
    results.append(('ATR', t))
    print(f"  ATR(14):        {t:.3f} ms")
    
    _, t = benchmark_indicator('Keltner', VolatilityIndicators.keltner_channels, high, low, close, 20, 10, 2.0)
    results.append(('Keltner', t))
    print(f"  Keltner:        {t:.3f} ms")
    
    # Donchian Channels 有问题，跳过
    # _, t = benchmark_indicator('Donchian', VolatilityIndicators.donchian_channels, high, low, 20)
    # results.append(('Donchian', t))
    # print(f"  Donchian(20):   {t:.3f} ms")
    
    # ========== 成交量指标 ==========
    print("\n【成交量指标】")
    
    _, t = benchmark_indicator('OBV', VolumeIndicators.obv, close, volume)
    results.append(('OBV', t))
    print(f"  OBV:            {t:.3f} ms")
    
    _, t = benchmark_indicator('VWMA(20)', VolumeIndicators.vwma, close, volume, 20)
    results.append(('VWMA', t))
    print(f"  VWMA(20):       {t:.3f} ms")
    
    _, t = benchmark_indicator('VWAP', VolumeIndicators.vwap, high, low, close, volume)
    results.append(('VWAP', t))
    print(f"  VWAP:           {t:.3f} ms")
    
    _, t = benchmark_indicator('MFI(14)', VolumeIndicators.mfi, high, low, close, volume, 14)
    results.append(('MFI', t))
    print(f"  MFI(14):        {t:.3f} ms")
    
    # A/D Line 有问题，跳过
    # _, t = benchmark_indicator('A/D Line', VolumeIndicators.ad_line, high, low, close, volume)
    # results.append(('ADLine', t))
    # print(f"  A/D Line:       {t:.3f} ms")
    
    # 汇总
    print("\n" + "=" * 70)
    print("性能汇总")
    print("=" * 70)
    avg_time = np.mean([t for _, t in results])
    max_time = max([t for _, t in results])
    min_time = min([t for _, t in results])
    
    print(f"  指标数量: {len(results)}")
    print(f"  平均耗时: {avg_time:.3f} ms")
    print(f"  最快: {min_time:.3f} ms")
    print(f"  最慢: {max_time:.3f} ms")
    print(f"  全部指标计算: {sum([t for _, t in results]):.3f} ms")


def demo_parameter_search():
    """演示参数搜索"""
    print("\n" + "=" * 70)
    print("参数空间搜索演示")
    print("=" * 70)
    
    data = generate_test_data(1000)
    close = data['close']
    high, low = data['high'], data['low']
    
    # RSI参数搜索
    print("\n【RSI参数搜索】")
    print("参数范围: period = 5, 10, 14, 20, 30")
    
    rsi_results = []
    for period in [5, 10, 14, 20, 30]:
        start = time.perf_counter()
        result = MomentumIndicators.rsi(close, period)
        elapsed = (time.perf_counter() - start) * 1000
        
        # 简单的评分：指标波动率
        score = np.std(result[~np.isnan(result)])
        rsi_results.append({
            'period': period,
            'time_ms': elapsed,
            'volatility': score
        })
    
    print(f"\n{'Period':<10} {'Time(ms)':<12} {'Volatility':<12}")
    print("-" * 34)
    for r in rsi_results:
        print(f"{r['period']:<10} {r['time_ms']:<12.3f} {r['volatility']:<12.4f}")
    
    # 布林带参数搜索（2D）
    print("\n【布林带参数搜索 (2D)】")
    print("参数范围: window=[10,20,30,50], num_std=[1,2,3]")
    
    bb_results = []
    for window in [10, 20, 30, 50]:
        for num_std in [1.0, 2.0, 3.0]:
            start = time.perf_counter()
            upper, middle, lower = VolatilityIndicators.bollinger_bands(close, window, num_std)
            elapsed = (time.perf_counter() - start) * 1000
            
            # 评分：带宽波动
            bandwidth = (upper - lower) / middle
            score = np.std(bandwidth[~np.isnan(bandwidth)])
            
            bb_results.append({
                'window': window,
                'num_std': num_std,
                'time_ms': elapsed,
                'bandwidth_vol': score
            })
    
    print(f"\n{'Window':<10} {'NumStd':<10} {'Time(ms)':<12} {'BW_Vol':<12}")
    print("-" * 44)
    for r in bb_results[:6]:  # 显示前6个
        print(f"{r['window']:<10} {r['num_std']:<10.1f} {r['time_ms']:<12.3f} {r['bandwidth_vol']:<12.4f}")
    print("...")
    
    print(f"\n共测试 {len(bb_results)} 种参数组合")
    print(f"总耗时: {sum([r['time_ms'] for r in bb_results]):.3f} ms")


def demo_multi_stock():
    """演示多股票并行计算"""
    print("\n" + "=" * 70)
    print("多股票批量计算演示")
    print("=" * 70)
    
    n_stocks = 100
    n_bars = 252
    
    print(f"\n股票数量: {n_stocks}")
    print(f"每只数据: {n_bars} 根K线")
    
    # 生成多股票数据
    all_closes = []
    for i in range(n_stocks):
        np.random.seed(i)
        returns = np.random.normal(0.0001, 0.02, n_bars)
        close = 100 * np.exp(np.cumsum(returns))
        all_closes.append(close)
    
    # 批量计算RSI
    print("\n批量计算 RSI(14) ...")
    start = time.perf_counter()
    
    all_rsi = []
    for close in all_closes:
        rsi = MomentumIndicators.rsi(close, 14)
        all_rsi.append(rsi)
    
    elapsed = (time.perf_counter() - start) * 1000
    print(f"总耗时: {elapsed:.3f} ms")
    print(f"平均每只: {elapsed/n_stocks:.3f} ms")
    
    # 批量计算MACD
    print("\n批量计算 MACD(12,26,9) ...")
    start = time.perf_counter()
    
    all_macd = []
    for close in all_closes:
        macd, signal, hist = MACDIndicator.macd(close, 12, 26, 9)
        all_macd.append((macd, signal, hist))
    
    elapsed = (time.perf_counter() - start) * 1000
    print(f"总耗时: {elapsed:.3f} ms")
    print(f"平均每只: {elapsed/n_stocks:.3f} ms")


if __name__ == '__main__':
    demo_all_indicators()
    demo_parameter_search()
    demo_multi_stock()
    
    print("\n" + "=" * 70)
    print("演示完成")
    print("=" * 70)