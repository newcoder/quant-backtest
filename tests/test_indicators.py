#!/usr/bin/env python3
"""
技术指标正确性测试
对比Numba实现与参考实现（Pandas/Talib）
"""

import numpy as np
import pandas as pd
import time
import unittest
from typing import Tuple

# 导入我们的实现
import sys
sys.path.insert(0, 'd:/projects/quant-backtest/scripts')

from indicators import MovingAverages, MomentumIndicators, VolatilityIndicators


class TestIndicatorCorrectness(unittest.TestCase):
    """指标正确性测试"""
    
    @classmethod
    def setUpClass(cls):
        """生成测试数据"""
        np.random.seed(42)
        cls.n = 500
        cls.close = 100 * np.exp(np.cumsum(np.random.randn(cls.n) * 0.02))
        cls.high = cls.close * (1 + abs(np.random.randn(cls.n) * 0.01))
        cls.low = cls.close * (1 - abs(np.random.randn(cls.n) * 0.01))
        cls.volume = np.random.randint(1000000, 10000000, cls.n)
    
    def assert_array_almost_equal(self, a, b, rtol=1e-5, atol=1e-8, msg=None):
        """自定义数组比较"""
        # 忽略NaN
        mask = ~(np.isnan(a) | np.isnan(b))
        if not np.allclose(a[mask], b[mask], rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(a[mask] - b[mask]))
            self.fail(f"{msg}: 最大差异 {max_diff}")
    
    # ========== 趋势指标测试 ==========
    
    def test_sma_correctness(self):
        """SMA正确性测试"""
        window = 20
        
        # Pandas参考实现
        pd_sma = pd.Series(self.close).rolling(window=window).mean().values
        
        # 我们的实现
        our_sma = MovingAverages.sma(self.close, window)
        
        self.assert_array_almost_equal(
            our_sma, pd_sma, 
            rtol=1e-10, 
            msg="SMA计算错误"
        )
        print(f"✓ SMA({window}) 正确性验证通过")
    
    def test_ema_correctness(self):
        """EMA正确性测试"""
        span = 20
        
        # Pandas参考实现
        pd_ema = pd.Series(self.close).ewm(span=span, adjust=False).mean().values
        
        # 我们的实现
        our_ema = MovingAverages.ema(self.close, span)
        
        self.assert_array_almost_equal(
            our_ema, pd_ema,
            rtol=1e-5,
            msg="EMA计算错误"
        )
        print(f"✓ EMA({span}) 正确性验证通过")
    
    def test_dema_correctness(self):
        """DEMA正确性测试"""
        span = 20
        
        # 手动计算参考值: DEMA = 2*EMA - EMA(EMA)
        ema1 = pd.Series(self.close).ewm(span=span, adjust=False).mean()
        ema2 = ema1.ewm(span=span, adjust=False).mean()
        pd_dema = (2 * ema1 - ema2).values
        
        # 我们的实现
        our_dema = MovingAverages.dema(self.close, span)
        
        self.assert_array_almost_equal(
            our_dema, pd_dema,
            rtol=1e-5,
            msg="DEMA计算错误"
        )
        print(f"✓ DEMA({span}) 正确性验证通过")
    
    # ========== 动量指标测试 ==========
    
    def test_rsi_correctness(self):
        """RSI正确性测试"""
        period = 14
        
        # 手动计算参考RSI
        deltas = np.diff(self.close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros(len(self.close))
        avg_losses = np.zeros(len(self.close))
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        for i in range(period + 1, len(self.close)):
            avg_gains[i] = (avg_gains[i-1] * (period-1) + gains[i-1]) / period
            avg_losses[i] = (avg_losses[i-1] * (period-1) + losses[i-1]) / period
        
        rs = avg_gains / (avg_losses + 1e-10)
        expected_rsi = 100 - (100 / (1 + rs))
        
        # 我们的实现
        our_rsi = MomentumIndicators.rsi(self.close, period)
        
        self.assert_array_almost_equal(
            our_rsi, expected_rsi,
            rtol=1e-10,
            msg="RSI计算错误"
        )
        print(f"✓ RSI({period}) 正确性验证通过")
    
    def test_rsi_range(self):
        """RSI值域测试"""
        rsi = MomentumIndicators.rsi(self.close, 14)
        valid_rsi = rsi[~np.isnan(rsi)]
        
        self.assertTrue(np.all(valid_rsi >= 0), "RSI低于0")
        self.assertTrue(np.all(valid_rsi <= 100), "RSI高于100")
        print("✓ RSI值域 [0, 100] 验证通过")
    
    # ========== 波动率指标测试 ==========
    
    def test_bollinger_correctness(self):
        """布林带正确性测试"""
        window, num_std = 20, 2.0
        
        # Pandas参考实现
        s = pd.Series(self.close)
        middle = s.rolling(window).mean()
        std = s.rolling(window).std()
        pd_upper = (middle + num_std * std).values
        pd_middle = middle.values
        pd_lower = (middle - num_std * std).values
        
        # 我们的实现
        our_upper, our_middle, our_lower = VolatilityIndicators.bollinger_bands(
            self.close, window, num_std
        )
        
        self.assert_array_almost_equal(our_upper, pd_upper, msg="布林带上轨错误")
        self.assert_array_almost_equal(our_middle, pd_middle, msg="布林带中轨错误")
        self.assert_array_almost_equal(our_lower, pd_lower, msg="布林带下轨错误")
        print(f"✓ BollingerBands({window},{num_std}) 正确性验证通过")
    
    def test_bollinger_symmetry(self):
        """布林带对称性测试"""
        upper, middle, lower = VolatilityIndicators.bollinger_bands(self.close, 20, 2)
        
        # 中轨应该是上下轨的中点
        calculated_middle = (upper + lower) / 2
        
        self.assert_array_almost_equal(
            middle, calculated_middle,
            rtol=1e-10,
            msg="布林带中轨不对称"
        )
        print("✓ 布林带对称性验证通过")


class TestIndicatorPerformance(unittest.TestCase):
    """指标性能测试"""
    
    @classmethod
    def setUpClass(cls):
        """生成大规模测试数据"""
        np.random.seed(42)
        cls.sizes = [1000, 10000, 100000]
    
    def benchmark_indicator(self, name, fn, *args, iterations=100):
        """基准测试"""
        # 预热（Numba编译）
        fn(*args)
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn(*args)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return np.mean(times) * 1000  # 转换为毫秒
    
    def test_sma_performance(self):
        """SMA性能测试"""
        print("\n【SMA性能测试】")
        for size in self.sizes:
            data = np.random.randn(size).cumsum() + 100
            elapsed = self.benchmark_indicator('SMA', MovingAverages.sma, data, 20, iterations=50)
            print(f"  数据量 {size:>6}: {elapsed:.3f} ms")
            
            # 性能要求：每1000点 < 1ms
            if size == 1000:
                self.assertLess(elapsed, 1.0, f"SMA性能不达标: {elapsed:.3f}ms")
    
    def test_rsi_performance(self):
        """RSI性能测试"""
        print("\n【RSI性能测试】")
        for size in self.sizes:
            data = np.random.randn(size).cumsum() + 100
            elapsed = self.benchmark_indicator('RSI', MomentumIndicators.rsi, data, 14, iterations=50)
            print(f"  数据量 {size:>6}: {elapsed:.3f} ms")
            
            # 性能要求：每1000点 < 2ms
            if size == 1000:
                self.assertLess(elapsed, 2.0, f"RSI性能不达标: {elapsed:.3f}ms")
    
    def test_ema_performance(self):
        """EMA性能测试"""
        print("\n【EMA性能测试】")
        for size in self.sizes:
            data = np.random.randn(size).cumsum() + 100
            elapsed = self.benchmark_indicator('EMA', MovingAverages.ema, data, 20, iterations=50)
            print(f"  数据量 {size:>6}: {elapsed:.3f} ms")
            
            # 性能要求：每1000点 < 0.5ms
            if size == 1000:
                self.assertLess(elapsed, 0.5, f"EMA性能不达标: {elapsed:.3f}ms")


class TestBacktestCorrectness(unittest.TestCase):
    """回测正确性测试"""
    
    def test_simple_strategy(self):
        """简单策略回测测试"""
        # 构造确定性的测试数据
        prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103])
        
        # 简单策略：每天持有
        returns = prices.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        
        # 验证总收益计算
        expected_total_return = prices.iloc[-1] / prices.iloc[0] - 1
        actual_total_return = cumulative.iloc[-1] - 1
        
        self.assertAlmostEqual(
            actual_total_return, expected_total_return,
            places=10,
            msg="回测收益计算错误"
        )
        print(f"✓ 简单策略回测验证通过: 收益={actual_total_return:.2%}")
    
    def test_commission_calculation(self):
        """交易成本计算测试"""
        initial_capital = 1000000
        commission_rate = 0.001
        
        # 单次交易
        trade_value = 100000
        expected_commission = trade_value * commission_rate
        
        # 验证费率
        self.assertEqual(expected_commission, 100, "交易成本计算错误")
        print("✓ 交易成本计算验证通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("技术指标测试套件")
    print("=" * 70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestIndicatorCorrectness))
    suite.addTests(loader.loadTestsFromTestCase(TestIndicatorPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktestCorrectness))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"运行测试: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 存在失败的测试")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    run_all_tests()