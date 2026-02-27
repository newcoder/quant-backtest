# 完整策略示例

## 示例1: 双均线策略（向量化）

```python
import numpy as np
import pandas as pd
from numba import njit

class DualMAStrategy:
    """
    双均线策略 - 完全向量化实现
    金叉买入，死叉卖出
    """
    
    def __init__(self, fast=20, slow=60):
        self.fast = fast
        self.slow = slow
    
    @njit(cache=True)
    def _compute_signals_numba(close, fast, slow):
        """Numba加速的信号计算"""
        n = len(close)
        ma_fast = np.zeros(n)
        ma_slow = np.zeros(n)
        
        # 计算均线
        for i in range(slow, n):
            ma_fast[i] = np.mean(close[i-fast:i])
            ma_slow[i] = np.mean(close[i-slow:i])
        
        # 生成信号
        signals = np.zeros(n)
        position = 0
        for i in range(slow, n):
            if ma_fast[i] > ma_slow[i] and position <= 0:
                signals[i] = 1  # 买入
                position = 1
            elif ma_fast[i] < ma_slow[i] and position >= 0:
                signals[i] = -1  # 卖出
                position = 0
        
        return signals
    
    def generate_signals(self, data):
        """
        生成交易信号
        data: DataFrame with 'close' column
        """
        close = data['close'].values
        signals = self._compute_signals_numba(close, self.fast, self.slow)
        return pd.Series(signals, index=data.index)


# 使用示例
def run_dual_ma_backtest():
    # 加载数据（假设已准备）
    data = load_stock_data('000001.SZ', '2020-01-01', '2024-01-01')
    
    # 创建策略
    strategy = DualMAStrategy(fast=20, slow=60)
    signals = strategy.generate_signals(data)
    
    # 向量化回测
    engine = VectorizedBacktest(
        prices=data['close'],
        signals=signals,
        initial_capital=1e6,
        commission=0.0003
    )
    
    results = engine.run()
    
    print(f"总收益率: {results['cumulative_returns'].iloc[-1]:.2%}")
    print(f"夏普比率: {results['sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    
    return results
```

## 示例2: 多因子选股策略

```python
class MultiFactorStrategy:
    """
    多因子选股策略
    因子：动量、价值、质量
    """
    
    def __init__(self, n_stocks=50, rebalance_freq='M'):
        self.n_stocks = n_stocks
        self.rebalance_freq = rebalance_freq
    
    def compute_factors(self, data):
        """
        计算多因子得分
        data: Panel (time x stocks x fields)
        """
        factors = {}
        
        # 动量因子：过去20日收益率
        returns = data['close'].pct_change(20)
        factors['momentum'] = returns.rank(axis=1, pct=True)
        
        # 价值因子：PE倒数
        pe = data['close'] / data['eps']
        factors['value'] = (1/pe).rank(axis=1, pct=True)
        
        # 质量因子：ROE
        roe = data['net_income'] / data['equity']
        factors['quality'] = roe.rank(axis=1, pct=True)
        
        # 合成因子得分
        score = (factors['momentum'] * 0.4 + 
                factors['value'] * 0.3 + 
                factors['quality'] * 0.3)
        
        return score
    
    def generate_portfolio(self, scores, date):
        """生成目标持仓"""
        # 选择得分最高的n只股票
        top_stocks = scores.loc[date].nlargest(self.n_stocks).index
        
        # 等权重配置
        weights = pd.Series(1/self.n_stocks, index=top_stocks)
        return weights
    
    def backtest(self, data):
        """执行回测"""
        scores = self.compute_factors(data)
        
        # 按月再平衡
        rebalance_dates = pd.date_range(
            start=data.index[0], 
            end=data.index[-1], 
            freq=self.rebalance_freq
        )
        
        portfolio_values = []
        current_weights = None
        
        for date in data.index:
            if date in rebalance_dates:
                current_weights = self.generate_portfolio(scores, date)
            
            # 计算组合收益
            if current_weights is not None:
                daily_returns = data['close'].loc[date].pct_change()
                port_return = (current_weights * daily_returns).sum()
                portfolio_values.append(port_return)
        
        return pd.Series(portfolio_values, index=data.index)


# 全市场并行回测
def run_universe_backtest():
    # 加载全市场数据
    universe = load_all_stocks()
    
    strategy = MultiFactorStrategy(n_stocks=50)
    
    # 并行回测
    results = backtest_universe(
        universe, 
        strategy.backtest,
        max_workers=16
    )
    
    # 汇总结果
    aggregate_returns = pd.DataFrame({
        symbol: res['returns'] 
        for symbol, res in results.items()
    }).mean(axis=1)
    
    return aggregate_returns
```

## 示例3: 分钟级高频策略

```python
class HighFrequencyStrategy:
    """
    分钟级高频策略
    基于订单流和微观结构
    """
    
    def __init__(self, lookback=30):
        self.lookback = lookback
    
    @njit(cache=True)
    def _compute_microstructure_numba(high, low, close, volume, lookback):
        """计算微观结构指标"""
        n = len(close)
        
        # 实现波动率
        rv = np.zeros(n)
        for i in range(lookback, n):
            rv[i] = np.sum((high[i-lookback:i] - low[i-lookback:i])**2)
        
        # 成交量加权价格
        vwap = np.zeros(n)
        for i in range(lookback, n):
            vol_sum = np.sum(volume[i-lookback:i])
            if vol_sum > 0:
                vwap[i] = np.sum(close[i-lookback:i] * volume[i-lookback:i]) / vol_sum
        
        # 价格冲击
        impact = np.zeros(n)
        for i in range(1, n):
            if volume[i] > 0:
                impact[i] = (close[i] - close[i-1]) / np.sqrt(volume[i])
        
        return rv, vwap, impact
    
    def generate_signals(self, data):
        """生成高频信号"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        
        rv, vwap, impact = self._compute_microstructure_numba(
            high, low, close, volume, self.lookback
        )
        
        # 信号逻辑
        signals = np.zeros(len(close))
        for i in range(self.lookback, len(close)):
            # 价格突破VWAP + 成交量确认 + 低冲击成本
            if (close[i] > vwap[i] and 
                volume[i] > np.mean(volume[i-5:i]) * 1.5 and
                abs(impact[i]) < np.percentile(impact[i-100:i], 90)):
                signals[i] = 1
            elif (close[i] < vwap[i] and 
                  volume[i] > np.mean(volume[i-5:i]) * 1.5):
                signals[i] = -1
        
        return pd.Series(signals, index=data.index)


# GPU加速的分钟级回测
def run_minute_backtest_gpu():
    import cupy as cp
    
    # 加载分钟数据（假设有3000只股票，每只2年分钟数据）
    data = load_minute_data(universe='all', years=2)
    
    # 转移到GPU
    close_gpu = cp.asarray(data['close'].values)
    volume_gpu = cp.asarray(data['volume'].values)
    
    # GPU并行计算指标
    ma_fast = cp.zeros_like(close_gpu)
    ma_slow = cp.zeros_like(close_gpu)
    
    # 批量计算所有股票的均线
    for i in range(20, len(close_gpu)):
        ma_fast[i] = cp.mean(close_gpu[i-20:i])
        ma_slow[i] = cp.mean(close_gpu[i-60:i])
    
    # 生成信号
    signals_gpu = (ma_fast > ma_slow).astype(cp.int8)
    
    # 传回CPU
    signals = cp.asnumpy(signals_gpu)
    
    return signals
```

## 示例4: 参数优化

```python
from skopt import gp_minimize
from skopt.space import Integer, Real

def optimize_strategy_params(data, strategy_class):
    """
    使用贝叶斯优化寻找最优参数
    """
    
    def objective(params):
        fast, slow, stop_loss = params
        
        if fast >= slow:
            return 0  # 无效参数
        
        strategy = strategy_class(
            fast=int(fast), 
            slow=int(slow),
            stop_loss=stop_loss
        )
        
        results = strategy.backtest(data)
        
        # 优化目标：夏普比率 - 回撤惩罚
        sharpe = results['sharpe']
        drawdown_penalty = abs(results['max_drawdown']) * 2
        
        return -(sharpe - drawdown_penalty)  # 最小化负数
    
    # 参数空间
    space = [
        Integer(5, 30, name='fast_ma'),      # 短期均线
        Integer(20, 120, name='slow_ma'),    # 长期均线
        Real(0.02, 0.1, name='stop_loss')    # 止损比例
    ]
    
    # 贝叶斯优化
    result = gp_minimize(
        objective,
        space,
        n_calls=100,
        n_random_starts=20,
        verbose=True
    )
    
    return result.x, -result.fun


# 使用示例
best_params, best_score = optimize_strategy_params(
    data, 
    DualMAStrategy
)
print(f"最优参数: {best_params}")
print(f"最佳得分: {best_score}")
```

## 示例5: 完整回测报告

```python
def generate_backtest_report(results):
    """生成详细回测报告"""
    
    report = {
        'summary': {
            'total_return': f"{results['cumulative_returns'].iloc[-1]:.2%}",
            'annualized_return': f"{results['returns'].mean() * 252:.2%}",
            'sharpe_ratio': f"{results['sharpe']:.2f}",
            'max_drawdown': f"{results['max_drawdown']:.2%}",
            'volatility': f"{results['returns'].std() * np.sqrt(252):.2%}",
            'win_rate': f"{(results['returns'] > 0).mean():.2%}",
        },
        'monthly_returns': results['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ),
        'drawdown_series': (
            results['cumulative_returns'] / 
            results['cumulative_returns'].cummax() - 1
        ),
        'trade_stats': compute_trade_stats(results['positions'], results['prices'])
    }
    
    return report


def plot_backtest_results(results):
    """绘制回测结果图表"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 净值曲线
    ax1 = axes[0]
    results['cumulative_returns'].plot(ax=ax1)
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Return')
    
    # 回撤
    ax2 = axes[1]
    drawdown = results['cumulative_returns'] / results['cumulative_returns'].cummax() - 1
    drawdown.plot(ax=ax2, color='red')
    ax2.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax2.set_title('Drawdown')
    ax2.set_ylabel('Drawdown')
    
    # 月度收益热力图
    ax3 = axes[2]
    monthly = results['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_matrix = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack()
    sns.heatmap(monthly_matrix, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax3)
    ax3.set_title('Monthly Returns Heatmap')
    
    plt.tight_layout()
    return fig
```