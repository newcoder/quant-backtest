# 性能优化技巧

## 1. 数据加载优化

### 1.1 内存映射
```python
import numpy as np

# 大文件内存映射，避免载入内存
mmapped = np.memmap('large_data.npy', dtype='float32', mode='r', shape=(1000000, 3000))

# 按需读取，零拷贝
subset = mmapped[0:1000]  # 不占用额外内存
```

### 1.2 分块处理
```python
def chunked_backtest(data, strategy, chunk_size=10000):
    """分块回测，控制内存使用"""
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        result = strategy(chunk)
        results.append(result)
    return pd.concat(results)
```

### 1.3 数据类型优化
```python
# 默认float64 -> float32，内存减半
df['price'] = df['price'].astype('float32')
df['volume'] = df['volume'].astype('int32')

# 类别型数据
df['stock_code'] = df['stock_code'].astype('category')
```

## 2. 计算优化

### 2.1 避免DataFrame遍历
```python
# ❌ 慢：逐行遍历
for i in range(len(df)):
    if df.iloc[i]['close'] > df.iloc[i]['open']:
        df.iloc[i, 'signal'] = 1

# ✅ 快：向量化操作
df['signal'] = np.where(df['close'] > df['open'], 1, 0)

# ✅ 更快：Numba编译
@njit
def compute_signal_numba(close, open_price):
    n = len(close)
    signals = np.zeros(n)
    for i in range(n):
        if close[i] > open_price[i]:
            signals[i] = 1
    return signals
```

### 2.2 预分配内存
```python
# ❌ 慢：动态扩容
results = []
for i in range(n):
    results.append(compute(i))

# ✅ 快：预分配
results = np.empty(n)
for i in range(n):
    results[i] = compute(i)
```

### 2.3 使用视图而非拷贝
```python
# ❌ 拷贝数据
subset = df[df['volume'] > 1000].copy()

# ✅ 使用视图（只读）
subset = df[df['volume'] > 1000]
```

## 3. GPU优化

### 3.1 数据传输最小化
```python
# ❌ 频繁传输
for i in range(1000):
    gpu_arr = cp.asarray(cpu_arr[i])
    result = cp.sum(gpu_arr)
    cpu_result = cp.asnumpy(result)

# ✅ 批量传输
gpu_arr = cp.asarray(cpu_arr)  # 一次性传输
results = cp.sum(gpu_arr, axis=1)  # GPU上批量计算
cpu_results = cp.asnumpy(results)  # 一次性传回
```

### 3.2 内存池管理
```python
# 使用内存池避免频繁分配
pool = cp.cuda.MemoryPool()
cp.cuda.set_allocator(pool.malloc)

# 手动释放
pool.free_all_blocks()
```

### 3.3 流并行
```python
# 多流并行执行
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

with stream1:
    result1 = compute_on_gpu(data1)
    
with stream2:
    result2 = compute_on_gpu(data2)

stream1.synchronize()
stream2.synchronize()
```

## 4. 缓存策略

### 4.1 技术指标缓存
```python
from functools import lru_cache
import hashlib

class IndicatorCache:
    def __init__(self, cache_dir='.cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_key(self, data, indicator_name, params):
        """生成缓存键"""
        data_hash = hashlib.md5(data.tobytes()).hexdigest()[:16]
        param_str = '_'.join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{indicator_name}_{param_str}_{data_hash}.npy"
    
    def get_or_compute(self, data, indicator_fn, indicator_name, **params):
        """获取缓存或重新计算"""
        key = self._get_key(data, indicator_name, params)
        cache_path = os.path.join(self.cache_dir, key)
        
        if os.path.exists(cache_path):
            return np.load(cache_path)
        
        result = indicator_fn(data, **params)
        np.save(cache_path, result)
        return result
```

### 4.2 热点数据常驻内存
```python
class HotDataCache:
    """热点数据缓存，LRU淘汰"""
    
    def __init__(self, max_size_gb=16):
        self.max_size = max_size_gb * 1024**3
        self.current_size = 0
        self.cache = {}
        self.access_times = {}
    
    def add(self, key, data):
        data_size = data.nbytes
        
        # 淘汰旧数据
        while self.current_size + data_size > self.max_size:
            lru_key = min(self.access_times, key=self.access_times.get)
            self.current_size -= self.cache[lru_key].nbytes
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = data
        self.access_times[key] = time.time()
        self.current_size += data_size
```

## 5. 并行优化

### 5.1 进程池vs线程池
```python
# CPU密集型：用进程池
from multiprocessing import Pool
with Pool(8) as p:
    results = p.map(cpu_intensive_task, data)

# IO密集型：用线程池
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(8) as e:
    results = e.map(io_intensive_task, urls)
```

### 5.2 避免GIL
```python
# ❌ 受GIL限制
import threading
threads = [threading.Thread(target=cpu_task) for _ in range(8)]

# ✅ 绕过GIL
from multiprocessing import Process
processes = [Process(target=cpu_task) for _ in range(8)]

# ✅ 使用Numba释放GIL
@njit(nogil=True, parallel=True)
def parallel_compute(arr):
    result = np.empty_like(arr)
    for i in prange(len(arr)):  # 并行循环
        result[i] = heavy_computation(arr[i])
    return result
```

## 6. 性能分析

### 6.1 代码分析
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# 运行回测
backtest.run()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 打印前20个耗时函数
```

### 6.2 内存分析
```python
from memory_profiler import profile

@profile
def backtest_strategy(data):
    # 分析内存使用
    indicators = compute_indicators(data)
    signals = generate_signals(indicators)
    return signals
```

### 6.3 GPU分析
```python
# CuPy性能分析
cp.cuda.Device().use()
with cp.cuda.profile():
    result = gpu_computation(data)
    
# 查看CUDA事件
cp.cuda.get_elapsed_time(start_event, end_event)
```

## 7. 常见陷阱

### 7.1 隐式拷贝
```python
# ❌ 隐式类型转换导致拷贝
df['col'] = df['col'].astype('float64')  # 如果原来是float32

# ✅ 保持类型一致
```

### 7.2 链式索引
```python
# ❌ 可能返回拷贝或视图，行为不确定
df[df['A'] > 0]['B'] = 1

# ✅ 使用.loc
df.loc[df['A'] > 0, 'B'] = 1
```

### 7.3 大数据拼接
```python
# ❌ 循环拼接，O(n²)
result = pd.DataFrame()
for df in dataframes:
    result = result.append(df)  # 每次创建新对象

# ✅ 预分配或使用concat
result = pd.concat(dataframes)  # 一次性分配
```