#!/usr/bin/env python3
"""
数据格式转换工具
支持CSV/Parquet/HDF5/Feather格式互转
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


def csv_to_parquet(input_path, output_path=None, compression='zstd', chunksize=None):
    """
    CSV转换为Parquet
    
    Args:
        input_path: 输入CSV文件路径
        output_path: 输出Parquet文件路径（默认同名）
        compression: 压缩算法 ('zstd', 'snappy', 'gzip')
        chunksize: 分块读取大小（大文件）
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.parquet'))
    
    print(f"转换: {input_path} -> {output_path}")
    
    if chunksize:
        # 分块处理大文件
        chunks = pd.read_csv(input_path, chunksize=chunksize)
        writer = None
        
        for i, chunk in enumerate(chunks):
            print(f"  处理块 {i+1}...")
            table = pa.Table.from_pandas(chunk)
            
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path, 
                    table.schema,
                    compression=compression
                )
            writer.write_table(table)
        
        if writer:
            writer.close()
    else:
        # 一次性读取
        df = pd.read_csv(input_path)
        df.to_parquet(output_path, compression=compression, engine='pyarrow')
    
    # 显示压缩效果
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    ratio = compressed_size / original_size
    
    print(f"  原始大小: {original_size / 1024**2:.2f} MB")
    print(f"  压缩后: {compressed_size / 1024**2:.2f} MB")
    print(f"  压缩率: {ratio:.1%}")
    
    return output_path


def optimize_dataframe(df):
    """
    优化DataFrame内存使用
    """
    print("优化数据类型...")
    
    # 数值列优化
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    
    for col in df.select_dtypes(include=['int64']).columns:
        col_min, col_max = df[col].min(), df[col].max()
        
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype('int8')
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype('int16')
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype('int32')
    
    # 字符串列优化为category
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        
        if num_unique / num_total < 0.5:  # 重复值较多
            df[col] = df[col].astype('category')
    
    return df


def batch_convert(input_dir, output_dir=None, format='parquet'):
    """
    批量转换目录中的所有CSV文件
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / 'converted'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    csv_files = list(input_path.glob('*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for csv_file in csv_files:
        output_file = output_dir / f"{csv_file.stem}.{format}"
        
        if format == 'parquet':
            csv_to_parquet(str(csv_file), str(output_file))
        elif format == 'feather':
            df = pd.read_csv(str(csv_file))
            df = optimize_dataframe(df)
            df.to_feather(str(output_file))
            print(f"转换: {csv_file.name} -> {output_file.name}")
        elif format == 'hdf5':
            df = pd.read_csv(str(csv_file))
            df = optimize_dataframe(df)
            df.to_hdf(str(output_file), key='data', mode='w')
            print(f"转换: {csv_file.name} -> {output_file.name}")


def create_sample_data(output_path, n_stocks=100, n_days=252, freq='1d'):
    """
    创建示例股票数据
    """
    print(f"生成示例数据: {n_stocks}只股票, {n_days}天")
    
    dates = pd.date_range('2020-01-01', periods=n_days, freq=freq)
    
    all_data = []
    for i in range(n_stocks):
        code = f"ST{i:04d}"
        
        # 生成随机价格
        returns = np.random.normal(0.0001, 0.02, n_days)
        close = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'code': code,
            'date': dates,
            'open': close * (1 + np.random.normal(0, 0.001, n_days)),
            'high': close * (1 + abs(np.random.normal(0, 0.01, n_days))),
            'low': close * (1 - abs(np.random.normal(0, 0.01, n_days))),
            'close': close,
            'volume': np.random.randint(1000000, 10000000, n_days)
        })
        
        all_data.append(df)
    
    result = pd.concat(all_data, ignore_index=True)
    
    # 优化并保存
    result = optimize_dataframe(result)
    result.to_parquet(output_path, compression='zstd')
    
    print(f"保存到: {output_path}")
    print(f"数据大小: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    
    return result


def verify_conversion(original, converted):
    """
    验证转换后的数据完整性
    """
    print("验证数据完整性...")
    
    orig_df = pd.read_csv(original) if original.endswith('.csv') else pd.read_parquet(original)
    conv_df = pd.read_parquet(converted)
    
    # 检查形状
    assert orig_df.shape == conv_df.shape, "形状不匹配"
    
    # 检查数值列
    numeric_cols = orig_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        orig_vals = orig_df[col].values
        conv_vals = conv_df[col].values
        
        # 考虑float32精度损失
        if orig_df[col].dtype == 'float64' and conv_df[col].dtype == 'float32':
            assert np.allclose(orig_vals, conv_vals, rtol=1e-5), f"{col} 数值不匹配"
        else:
            assert np.array_equal(orig_vals, conv_vals), f"{col} 数值不匹配"
    
    print("  ✓ 数据验证通过")


def main():
    parser = argparse.ArgumentParser(description='数据格式转换工具')
    parser.add_argument('command', choices=['convert', 'batch', 'sample', 'verify'])
    parser.add_argument('-i', '--input', help='输入文件/目录')
    parser.add_argument('-o', '--output', help='输出文件/目录')
    parser.add_argument('-f', '--format', default='parquet', 
                       choices=['parquet', 'feather', 'hdf5'])
    parser.add_argument('-c', '--compression', default='zstd',
                       choices=['zstd', 'snappy', 'gzip', 'none'])
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        csv_to_parquet(args.input, args.output, args.compression)
    
    elif args.command == 'batch':
        batch_convert(args.input, args.output, args.format)
    
    elif args.command == 'sample':
        output = args.output or 'sample_data.parquet'
        create_sample_data(output)
    
    elif args.command == 'verify':
        verify_conversion(args.input, args.output)


if __name__ == '__main__':
    main()