#!/usr/bin/env python3
"""
并行回测引擎 - 支持参数空间搜索
同一指标的不同参数实例并行处理
"""

import numpy as np
import pandas as pd
from numba import njit, prange
from typing import Dict, List, Callable, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import time
from multiprocessing import cpu_count

from indicators import (
    MovingAverages, MACDIndicator, MomentumIndicators,
    VolatilityIndicators, VolumeIndicators, ParameterSpaceSearch
)


@dataclass
class StrategyInstance:
    """策略实例 - 包含特定参数配置"""
    name: str
    indicator_fn: Callable
    params: Dict[str, Any]
    weight: float = 1.0


class ParallelBacktestEngine:
    """
    并行回测引擎
    
    核心特性：
    1. 同一指标的不同参数实例并行回测
    2. 多进程加速参数空间搜索
    3. 向量化计算 + 并行回测的混合架构
    """
    
    def __init__(self, n_jobs: int = -1, use_vectorized: bool = True):
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.use_vectorized = use_vectorized
        self.results_cache = {}
    
    def _compute_signals_vectorized(self, 
                                   prices: pd.DataFrame,
                                   strategy_instance: StrategyInstance) -> pd.DataFrame:
        """
        向量化计算信号
        
        对每只股票应用相同的指标参数
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
        
        for col in prices.columns:
            close = prices[col].values
            
            # 根据指标类型计算
            indicator_name = strategy_instance.name
            params = strategy_instance.params
            
            if 'sma' in indicator_name.lower():
                values = MovingAverages.sma(close, **params)
                signals[col] = (close > values).astype(int)
            elif 'rsi' in indicator_name.lower():
                values = MomentumIndicators.rsi(close, **params)
                signals[col] = ((values < 30).astype(int) - (values > 70).astype(int))
            elif 'macd' in indicator_name.lower():
                macd_line, signal_line, _ = MACDIndicator.macd(close, **params)
                signals[col] = ((macd_line > signal_line).astype(int) - 
                               (macd_line < signal_line).astype(int))
            elif 'bb' in indicator_name.lower():
                upper, middle, lower = VolatilityIndicators.bollinger_bands(close, **params)
                signals[col] = ((close < lower).astype(int) - (close > upper).astype(int))
            else:
                # 默认：价格上穿指标值买入
                values = strategy_instance.indicator_fn(close, **params)
                signals[col] = (close > values).astype(int)
        
        return signals
    
    def _run_single_backtest(self,
                            prices: pd.DataFrame,
                            strategy_instance: StrategyInstance,
                            commission: float = 0.001,
                            initial_capital: float = 1e6) -> Dict:
        """
        执行单个策略实例的回测
        """
        # 生成信号
        signals = self._compute_signals_vectorized(prices, strategy_instance)
        
        # 向量化回测计算
        returns = prices.pct_change().fillna(0)
        position = signals.shift(1).fillna(0)
        
        # 策略收益
        strategy_returns = (position * returns).sum(axis=1)
        
        # 交易成本
        turnover = position.diff().abs().sum(axis=1)
        costs = turnover * commission
        
        # 净收益
        net_returns = strategy_returns - costs
        cumulative = (1 + net_returns).cumprod()
        
        # 计算指标
        total_return = cumulative.iloc[-1] - 1
        sharpe = self._calc_sharpe(net_returns)
        max_dd = self._calc_drawdown(cumulative)
        
        return {
            'strategy_name': strategy_instance.name,
            'params': strategy_instance.params,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cumulative_returns': cumulative,
            'daily_returns': net_returns,
            'signals': signals
        }
    
    def _calc_sharpe(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        """计算夏普比率"""
        if returns.std() == 0:
            return 0
        excess = returns - risk_free / 252
        return excess.mean() / excess.std() * np.sqrt(252)
    
    def _calc_drawdown(self, cumulative: pd.Series) -> float:
        """计算最大回撤"""
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def parallel_backtest(self,
                         prices: pd.DataFrame,
                         strategy_instances: List[StrategyInstance],
                         commission: float = 0.001) -> List[Dict]:
        """
        并行回测多个策略实例
        
        每个实例可以视为"同一指标的不同参数配置"
        它们之间完全独立，可以并行处理
        """
        print(f"开始并行回测: {len(strategy_instances)} 个策略实例")
        print(f"使用 {self.n_jobs} 个进程")
        
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    self._run_single_backtest,
                    prices,
                    instance,
                    commission
                ): instance for instance in strategy_instances
            }
            
            # 收集结果
            for i, future in enumerate(as_completed(futures)):
                instance = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if (i + 1) % 10 == 0 or i == len(strategy_instances) - 1:
                        print(f"  已完成: {i+1}/{len(strategy_instances)}")
                        
                except Exception as e:
                    print(f"策略 {instance.name} {instance.params} 回测失败: {e}")
        
        # 按夏普比率排序
        results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return results
    
    def parameter_space_backtest(self,
                                prices: pd.DataFrame,
                                indicator_name: str,
                                indicator_fn: Callable,
                                param_ranges: Dict[str, Tuple],
                                required_data: List[str] = None,
                                commission: float = 0.001) -> List[Dict]:
        """
        参数空间回测 - 自动搜索最优参数
        
        Args:
            prices: 价格数据 DataFrame
            indicator_name: 指标名称
            indicator_fn: 指标计算函数
            param_ranges: 参数范围 {param: (min, max, step)}
            required_data: 需要的数据字段
            commission: 交易成本
        
        Returns:
            按夏普比率排序的回测结果列表
        """
        # 生成所有参数组合
        param_search = ParameterSpaceSearch(n_jobs=1)  # 外层已并行
        param_combos = param_search.generate_param_combinations(param_ranges)
        
        print(f"{indicator_name} 参数空间: {len(param_combos)} 种组合")
        
        # 创建策略实例列表
        strategy_instances = [
            StrategyInstance(
                name=indicator_name,
                indicator_fn=indicator_fn,
                params=combo
            )
            for combo in param_combos
        ]
        
        # 并行回测
        return self.parallel_backtest(prices, strategy_instances, commission)


class MultiIndicatorBacktest:
    """
    多指标组合回测
    
    支持同时优化多个指标的参数组合
    """
    
    def __init__(self, engine: ParallelBacktestEngine = None):
        self.engine = engine or ParallelBacktestEngine()
    
    def combine_signals(self, 
                       signals_list: List[pd.DataFrame],
                       weights: List[float] = None,
                       method: str = 'vote') -> pd.DataFrame:
        """
        组合多个信号
        
        Args:
            signals_list: 信号DataFrame列表
            weights: 权重列表
            method: 'vote'(投票), 'weighted'(加权), 'consensus'(共识)
        """
        if weights is None:
            weights = [1.0] * len(signals_list)
        
        if method == 'vote':
            # 多数投票
            combined = sum(s * w for s, w in zip(signals_list, weights))
            return np.sign(combined)
        
        elif method == 'weighted':
            # 加权平均
            combined = sum(s * w for s, w in zip(signals_list, weights))
            return np.sign(combined)
        
        elif method == 'consensus':
            # 所有信号一致才交易
            combined = sum(np.sign(s) for s in signals_list)
            return (combined == len(signals_list)).astype(int) - (combined == -len(signals_list)).astype(int)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def grid_search_combinations(self,
                                prices: pd.DataFrame,
                                indicator_configs: List[Dict],
                                commission: float = 0.001) -> List[Dict]:
        """
        网格搜索多指标组合
        
        Args:
            indicator_configs: [
                {
                    'name': 'RSI',
                    'fn': MomentumIndicators.rsi,
                    'param_ranges': {'period': (5, 30, 5)}
                },
                ...
            ]
        """
        # 先对每个指标进行参数搜索
        all_results = {}
        
        for config in indicator_configs:
            print(f"\n搜索 {config['name']} 最优参数...")
            results = self.engine.parameter_space_backtest(
                prices,
                config['name'],
                config['fn'],
                config['param_ranges'],
                commission=commission
            )
            all_results[config['name']] = results[:5]  # 保留前5名
        
        # 组合最优参数
        print("\n组合多指标策略...")
        combinations = list(itertools.product(*[
            [(name, r['params'], r['signals']) for r in results]
            for name, results in all_results.items()
        ]))
        
        print(f"共有 {len(combinations)} 种组合需要测试")
        
        # 评估每种组合
        combo_results = []
        for combo in combinations:
            # 组合信号
            signals_list = [item[2] for item in combo]
            combined_signal = self.combine_signals(signals_list, method='vote')
            
            # 回测
            result = self.engine._run_single_backtest(
                prices,
                StrategyInstance(
                    name='Combined',
                    indicator_fn=None,
                    params={'components': [item[:2] for item in combo]}
                ),
                commission
            )
            combo_results.append(result)
        
        combo_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        return combo_results


def demo_parameter_search():
    """
    参数搜索演示
    """
    print("=" * 60)
    print("并行参数空间回测演示")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    n_stocks = 50
    n_days = 252
    
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    stocks = [f'ST{i:04d}' for i in range(n_stocks)]
    
    prices = pd.DataFrame(index=dates, columns=stocks)
    for stock in stocks:
        returns = np.random.normal(0.0001, 0.02, n_days)
        prices[stock] = 100 * np.exp(np.cumsum(returns))
    
    print(f"\n数据: {n_stocks} 只股票, {n_days} 个交易日")
    
    # 创建引擎
    engine = ParallelBacktestEngine(n_jobs=4)
    
    # 1. 单指标参数搜索 - RSI
    print("\n" + "-" * 60)
    print("1. RSI 参数空间搜索")
    print("-" * 60)
    
    rsi_results = engine.parameter_space_backtest(
        prices,
        indicator_name='RSI',
        indicator_fn=MomentumIndicators.rsi,
        param_ranges={'period': (5, 30, 5)},
        commission=0.001
    )
    
    print(f"\nRSI 最优参数 (Top 3):")
    for i, r in enumerate(rsi_results[:3], 1):
        print(f"  {i}. 参数: {r['params']}")
        print(f"     夏普: {r['sharpe_ratio']:.2f}, 收益: {r['total_return']:.2%}, 回撤: {r['max_drawdown']:.2%}")
    
    # 2. 单指标参数搜索 - 布林带
    print("\n" + "-" * 60)
    print("2. Bollinger Bands 参数空间搜索")
    print("-" * 60)
    
    bb_results = engine.parameter_space_backtest(
        prices,
        indicator_name='BollingerBands',
        indicator_fn=VolatilityIndicators.bollinger_bands,
        param_ranges={'window': (10, 50, 10), 'num_std': (1, 3, 1)},
        commission=0.001
    )
    
    print(f"\n布林带 最优参数 (Top 3):")
    for i, r in enumerate(bb_results[:3], 1):
        print(f"  {i}. 参数: {r['params']}")
        print(f"     夏普: {r['sharpe_ratio']:.2f}, 收益: {r['total_return']:.2%}, 回撤: {r['max_drawdown']:.2%}")
    
    # 3. 多指标并行搜索
    print("\n" + "-" * 60)
    print("3. 多指标并行参数搜索")
    print("-" * 60)
    
    # 创建多个策略实例
    strategy_instances = []
    
    # RSI 不同参数
    for period in [7, 14, 21]:
        strategy_instances.append(StrategyInstance(
            name='RSI',
            indicator_fn=MomentumIndicators.rsi,
            params={'period': period}
        ))
    
    # MACD 不同参数
    for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 21, 5)]:
        strategy_instances.append(StrategyInstance(
            name='MACD',
            indicator_fn=MACDIndicator.macd,
            params={'fast': fast, 'slow': slow, 'signal': signal}
        ))
    
    # SMA 不同周期
    for window in [10, 20, 60]:
        strategy_instances.append(StrategyInstance(
            name='SMA',
            indicator_fn=MovingAverages.sma,
            params={'window': window}
        ))
    
    print(f"共 {len(strategy_instances)} 个策略实例")
    
    start_time = time.time()
    all_results = engine.parallel_backtest(prices, strategy_instances)
    elapsed = time.time() - start_time
    
    print(f"\n并行回测完成，耗时: {elapsed:.2f} 秒")
    print(f"\n综合排名 (Top 5):")
    for i, r in enumerate(all_results[:5], 1):
        print(f"  {i}. {r['strategy_name']} {r['params']}")
        print(f"     夏普: {r['sharpe_ratio']:.2f}, 收益: {r['total_return']:.2%}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    demo_parameter_search()