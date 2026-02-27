---
name: quant-backtest
description: High-performance stock strategy backtesting system with GPU acceleration and vectorized computation. Supports minute-level backtesting of thousands of stocks. Use when building quantitative trading backtest engines, implementing technical indicators, or optimizing backtest performance with CUDA/Numba vectorization.
---

# 高性能策略回测系统

## 核心架构

### 1. 数据层 (Data Layer)
- **存储**: Parquet/Arrow 列式存储，支持内存映射
- **缓存**: LRU缓存热点数据，预加载常用时间窗口
- **格式**: OHLCV + 复权因子，支持分钟/日/周级别

### 2. 计算层 (Compute Layer)
- **技术指标**: Numba/CUDA 加速的向量化计算
- **信号生成**: 批量矩阵运算，避免Python循环
- **向量化回测**: 一次性计算全时间序列，非事件驱动

### 3. 回测引擎 (Backtest Engine)
- **向量化回测**: 适用于规则策略，O(n)复杂度
- **事件驱动回测**: 适用于复杂逻辑，支持滑点/冲击成本
- **混合模式**: 向量化筛选 + 事件驱动精细化

## 快速开始

### 技术指标计算 (GPU加速)
```python
from backtest.indicators import TechnicalIndicators
import cupy as cp

# GPU加速的MACD计算
ti = TechnicalIndicators(use_gpu=True)
macd, signal, hist = ti.macd_gpu(prices, fast=12, slow=26, signal=9)
```

### 向量化回测
```python
from backtest.engine import VectorizedBacktest

engine = VectorizedBacktest(
    universe=stock_universe,  # 3000+ stocks
    start_date='2020-01-01',
    end_date='2024-01-01',
    frequency='1m'  # 分钟级别
)

# 定义策略信号
def strategy_fn(data):
    # 向量化计算，无Python循环
    ma20 = data['close'].rolling(20).mean()
    ma60 = data['close'].rolling(60).mean()
    return (ma20 > ma60) & (data['volume'] > data['volume'].rolling(20).mean())

results = engine.run(strategy_fn)
```

## 性能优化指南

### 1. 数据优化
- 使用 `pandas.to_parquet()` 存储，压缩率高
- 预计算并缓存技术指标
- 分块读取大文件，避免OOM

### 2. 计算优化
- 所有指标计算使用 Numba `@njit`
- GPU可用时使用 CuPy 替代 NumPy
- 避免DataFrame遍历，使用 `.apply()` 或矩阵运算

### 3. 回测优化
- 优先使用向量化回测（快100x+）
- 事件驱动仅用于复杂订单逻辑
- 并行处理多策略/多参数

## 脚本工具

- `scripts/benchmark.py` - 性能基准测试
- `scripts/indicator_perf.py` - 技术指标性能对比
- `scripts/data_converter.py` - 数据格式转换工具

## 参考文档

- `references/architecture.md` - 详细架构设计
- `references/optimization.md` - 性能优化技巧
- `references/examples.md` - 完整策略示例

## 依赖安装

```bash
pip install -r requirements.txt
# GPU支持（可选）
pip install cupy-cuda12x
```