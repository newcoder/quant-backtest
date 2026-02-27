# 高性能回测系统架构设计

## 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户接口层 (API/CLI)                      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      策略定义层 (Strategy)                     │
│  - 信号生成 (Signal Generation)                              │
│  - 仓位管理 (Position Sizing)                                │
│  - 风控规则 (Risk Management)                                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      回测引擎层 (Engine)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ 向量化回测   │  │ 事件驱动回测 │  │ 混合回测     │          │
│  │ Vectorized  │  │ Event-Driven│  │ Hybrid      │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      计算加速层 (Compute)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Numba JIT   │  │ CUDA GPU    │  │ 多进程并行   │          │
│  │ @njit       │  │ CuPy/Numba  │  │ multiprocess│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      数据管理层 (Data)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Parquet存储  │  │ 内存缓存     │  │ 实时数据     │          │
│  │ Arrow格式   │  │ LRU Cache   │  │ WebSocket   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. 数据管理层

#### 1.1 存储格式
```python
# 推荐：Parquet + PyArrow
import pyarrow.parquet as pq
import pyarrow as pa

# 列式存储优势：
# - 压缩率高（相比CSV节省70%+空间）
# - 读取速度快（只读需要的列）
# - 支持内存映射（零拷贝读取）

def save_data(df, path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression='zstd')

def load_data(path, columns=None):
    # 只加载需要的列
    return pq.read_table(path, columns=columns).to_pandas()
```

#### 1.2 数据缓存策略
```python
from functools import lru_cache
import pandas as pd

class DataCache:
    def __init__(self, max_size=100):
        self._cache = {}
        self._max_size = max_size
        self._access_count = {}
    
    def get(self, key, loader_fn):
        if key not in self._cache:
            if len(self._cache) >= self._max_size:
                # LRU淘汰
                lru_key = min(self._access_count, key=self._access_count.get)
                del self._cache[lru_key]
                del self._access_count[lru_key]
            self._cache[key] = loader_fn()
        self._access_count[key] = self._access_count.get(key, 0) + 1
        return self._cache[key]
```

### 2. 计算加速层

#### 2.1 Numba JIT编译
```python
from numba import njit, prange
import numpy as np

@njit(cache=True, parallel=True)
def compute_rsi_numba(prices, period=14):
    """Numba加速的RSI计算，比纯Python快100x"""
    n = len(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.zeros(n)
    avg_losses = np.zeros(n)
    
    # 初始值
    avg_gains[period] = np.mean(gains[:period])
    avg_losses[period] = np.mean(losses[:period])
    
    # 平滑计算
    for i in range(period + 1, n):
        avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
        avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
    
    rs = avg_gains / (avg_losses + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

#### 2.2 CUDA GPU加速
```python
import cupy as cp
from numba import cuda
import math

@cuda.jit
def ema_cuda_kernel(prices, out, alpha, n):
    """CUDA核函数：并行计算EMA"""
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if idx < n:
        ema = prices[idx]
        for i in range(idx+1, n):
            ema = alpha * prices[i] + (1 - alpha) * ema
        out[idx] = ema

def compute_ema_gpu(prices, span=20):
    """GPU加速的EMA计算"""
    prices_gpu = cp.asarray(prices)
    out_gpu = cp.empty_like(prices_gpu)
    alpha = 2.0 / (span + 1)
    
    threads_per_block = 256
    blocks = (len(prices) + threads_per_block - 1) // threads_per_block
    
    ema_cuda_kernel[blocks, threads_per_block](prices_gpu, out_gpu, alpha, len(prices))
    return cp.asnumpy(out_gpu)
```

### 3. 回测引擎层

#### 3.1 向量化回测（推荐）
```python
import numpy as np
import pandas as pd
from numba import njit

class VectorizedBacktest:
    """
    向量化回测引擎
    适用于：规则明确的策略，追求极致性能
    复杂度：O(n)，n为时间步数
    """
    
    def __init__(self, prices, signals, initial_capital=1e6, 
                 commission=0.001, slippage=0.001):
        self.prices = prices  # DataFrame: index=datetime, columns=stocks
        self.signals = signals  # DataFrame: 1=买入, -1=卖出, 0=持有
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
    
    def run(self):
        """执行向量化回测"""
        # 计算收益率
        returns = self.prices.pct_change().fillna(0)
        
        # 信号滞后一期（避免未来函数）
        position = self.signals.shift(1).fillna(0)
        
        # 向量化计算策略收益
        strategy_returns = (position * returns).sum(axis=1)
        
        # 考虑交易成本
        turnover = position.diff().abs().sum(axis=1)
        costs = turnover * (self.commission + self.slippage)
        
        # 净值曲线
        net_returns = strategy_returns - costs
        cumulative = (1 + net_returns).cumprod()
        
        return {
            'cumulative_returns': cumulative,
            'returns': net_returns,
            'sharpe': self._calc_sharpe(net_returns),
            'max_drawdown': self._calc_drawdown(cumulative),
            'positions': position
        }
    
    def _calc_sharpe(self, returns, risk_free=0.02):
        """计算夏普比率"""
        excess = returns - risk_free / 252
        return excess.mean() / (excess.std() + 1e-10) * np.sqrt(252)
    
    def _calc_drawdown(self, cumulative):
        """计算最大回撤"""
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
```

#### 3.2 事件驱动回测
```python
from dataclasses import dataclass
from typing import List, Dict
from queue import Queue
import heapq

@dataclass
class Event:
    timestamp: pd.Timestamp
    type: str  # 'MARKET', 'SIGNAL', 'ORDER', 'FILL'
    data: Dict

class EventDrivenBacktest:
    """
    事件驱动回测引擎
    适用于：复杂订单逻辑、滑点建模、冲击成本
    复杂度：O(n*m)，n为时间步，m为事件数
    """
    
    def __init__(self, initial_capital=1e6):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.events = []
        self.trades = []
    
    def add_event(self, event: Event):
        heapq.heappush(self.events, (event.timestamp, event))
    
    def run(self):
        """执行事件循环"""
        while self.events:
            _, event = heapq.heappop(self.events)
            self._process_event(event)
    
    def _process_event(self, event: Event):
        if event.type == 'MARKET':
            self._update_market(event.data)
        elif event.type == 'SIGNAL':
            self._generate_order(event.data)
        elif event.type == 'FILL':
            self._process_fill(event.data)
    
    def _generate_order(self, signal_data):
        """根据信号生成订单"""
        # 实现具体的订单逻辑
        pass
```

### 4. 并行计算架构

#### 4.1 多进程参数优化
```python
from multiprocessing import Pool, cpu_count
from itertools import product

def parallel_backtest(param_grid, data, n_jobs=-1):
    """
    并行回测多个参数组合
    """
    if n_jobs == -1:
        n_jobs = cpu_count()
    
    param_combinations = list(product(*param_grid.values()))
    
    def run_single(params):
        param_dict = dict(zip(param_grid.keys(), params))
        engine = VectorizedBacktest(data, **param_dict)
        return engine.run()
    
    with Pool(n_jobs) as pool:
        results = pool.map(run_single, param_combinations)
    
    return results
```

#### 4.2 多股票并行
```python
from concurrent.futures import ProcessPoolExecutor

def backtest_stock(stock_data, strategy_fn):
    """单只股票回测"""
    engine = VectorizedBacktest(stock_data)
    return engine.run(strategy_fn)

def backtest_universe(universe_data, strategy_fn, max_workers=8):
    """
    全市场股票并行回测
    universe_data: Dict[str, DataFrame] - 股票代码到数据的映射
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            symbol: executor.submit(backtest_stock, data, strategy_fn)
            for symbol, data in universe_data.items()
        }
        
        results = {}
        for symbol, future in futures.items():
            try:
                results[symbol] = future.result()
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
    
    return results
```

## 性能基准

| 组件 | 纯Python | Numba | CUDA | 加速比 |
|------|---------|-------|------|--------|
| RSI计算 | 1000ms | 10ms | 2ms | 500x |
| MACD计算 | 2000ms | 20ms | 5ms | 400x |
| 向量化回测 | - | 50ms | - | - |
| 事件驱动回测 | 5000ms | - | - | - |
| 3000只股票日级别 | 30min | 30s | 10s | 180x |
| 3000只股票分钟级 | 数小时 | 5min | 2min | 100x+ |

## 部署建议

### 硬件配置
- **CPU**: 16核+，高主频
- **内存**: 64GB+，支持大内存映射
- **GPU**: NVIDIA RTX 4090 / A100（可选但推荐）
- **存储**: NVMe SSD，支持高并发读取

### 软件栈
```
Python 3.10+
NumPy 1.24+ (BLAS优化)
Pandas 2.0+ (PyArrow后端)
Numba 0.58+ (LLVM编译)
CuPy 12+ (CUDA加速)
PyArrow 14+ (列式存储)
```