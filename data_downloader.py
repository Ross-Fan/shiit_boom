#!/usr/bin/env python3
"""
数据下载器 - 从币安下载历史交易数据

功能：
1. 获取所有USDT永续合约列表
2. 下载指定日期范围的aggTrades数据
3. 根据交易量筛选符合条件的合约
"""

import os
import sys
import json
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
import time

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from config import *


class BinanceDataDownloader:
    """币安数据下载器"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://data.binance.vision/data/futures/um/daily"
        self.session = requests.Session()

    def get_all_symbols(self) -> List[str]:
        """获取所有USDT永续合约符号"""
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['quoteAsset'] == 'USDT' and s['contractType'] == 'PERPETUAL'
            ]
            print(f"获取到 {len(symbols)} 个USDT永续合约")
            return symbols
        except Exception as e:
            print(f"获取合约列表失败: {e}")
            return []

    def download_file(self, symbol: str, date: str, data_type: str = 'aggTrades') -> Optional[Path]:
        """
        下载单个文件

        Args:
            symbol: 交易对符号
            date: 日期字符串 YYYY-MM-DD
            data_type: 数据类型 (aggTrades, metrics, bookDepth)

        Returns:
            下载成功返回文件路径，失败返回None
        """
        filename = f"{symbol}-{data_type}-{date}.zip"
        url = f"{self.base_url}/{data_type}/{symbol}/{filename}"
        local_path = self.data_dir / symbol / filename

        # 如果已存在则跳过
        csv_path = local_path.with_suffix('.csv')
        if csv_path.exists():
            return csv_path

        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = self.session.get(url, timeout=60)
            if resp.status_code == 404:
                return None  # 数据不存在
            resp.raise_for_status()

            # 保存zip文件
            with open(local_path, 'wb') as f:
                f.write(resp.content)

            # 解压
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(local_path.parent)

            # 删除zip文件
            local_path.unlink()

            return csv_path

        except Exception as e:
            if local_path.exists():
                local_path.unlink()
            return None

    def get_daily_volume(self, symbol: str, date: str) -> Optional[float]:
        """
        获取某个合约某天的交易量（USDT）

        通过metrics数据获取，如果没有则通过aggTrades估算
        """
        # 先尝试下载metrics
        metrics_path = self.download_file(symbol, date, 'metrics')
        if metrics_path and metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                if 'sum_open_interest_value' in df.columns:
                    # 使用持仓价值作为活跃度参考
                    avg_oi = df['sum_open_interest_value'].mean()
                    return avg_oi
            except:
                pass

        # 回退到aggTrades估算
        agg_path = self.download_file(symbol, date, 'aggTrades')
        if agg_path and agg_path.exists():
            try:
                df = pd.read_csv(agg_path)
                if len(df) > 0:
                    # 计算总交易额
                    total_volume = (df['price'] * df['quantity']).sum()
                    return total_volume
            except:
                pass

        return None

    def filter_symbols_by_volume(
        self,
        symbols: List[str],
        date: str,
        min_volume: float = MIN_DAILY_VOLUME_USDT,
        max_volume: float = MAX_DAILY_VOLUME_USDT,
        max_workers: int = 10
    ) -> List[Dict]:
        """
        按交易量筛选合约

        Returns:
            符合条件的合约列表，包含symbol和volume
        """
        print(f"\n正在筛选 {date} 的合约 (交易量 {min_volume/1e6:.0f}M - {max_volume/1e6:.0f}M USDT)...")

        results = []
        failed = []

        def check_volume(symbol):
            vol = self.get_daily_volume(symbol, date)
            if vol is not None and min_volume <= vol <= max_volume:
                return {'symbol': symbol, 'volume': vol}
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_volume, s): s for s in symbols}
            for i, future in enumerate(as_completed(futures)):
                symbol = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"  ✓ {symbol}: {result['volume']/1e6:.2f}M USDT")
                except Exception as e:
                    failed.append(symbol)

                # 进度显示
                if (i + 1) % 50 == 0:
                    print(f"  进度: {i+1}/{len(symbols)}")

        print(f"\n筛选完成: {len(results)} 个合约符合条件")
        return sorted(results, key=lambda x: x['volume'], reverse=True)

    def download_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        data_types: List[str] = ['aggTrades']
    ) -> Dict[str, List[Path]]:
        """
        下载指定合约的数据

        Returns:
            {data_type: [file_paths]}
        """
        result = {dt: [] for dt in data_types}
        current = start_date

        while current <= end_date:
            date_str = current.strftime('%Y-%m-%d')
            for data_type in data_types:
                path = self.download_file(symbol, date_str, data_type)
                if path:
                    result[data_type].append(path)
            current += timedelta(days=1)

        return result


def main():
    """主程序：下载回测所需数据"""
    downloader = BinanceDataDownloader()

    print("=" * 70)
    print("大规模回测数据下载器")
    print("=" * 70)
    print(f"回测时间范围: {START_DATE.date()} ~ {END_DATE.date()}")
    print(f"交易量筛选: {MIN_DAILY_VOLUME_USDT/1e6:.0f}M - {MAX_DAILY_VOLUME_USDT/1e6:.0f}M USDT")

    # 获取所有合约
    all_symbols = downloader.get_all_symbols()
    if not all_symbols:
        print("无法获取合约列表，退出")
        return

    # 按日期筛选合约并下载数据
    daily_symbols = {}  # {date: [symbols]}

    current_date = START_DATE
    while current_date <= END_DATE:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n{'='*50}")
        print(f"处理日期: {date_str}")
        print(f"{'='*50}")

        # 筛选当天符合条件的合约
        qualified = downloader.filter_symbols_by_volume(all_symbols, date_str)

        if qualified:
            daily_symbols[date_str] = [s['symbol'] for s in qualified]

            # 下载这些合约的aggTrades数据
            print(f"\n下载 {len(qualified)} 个合约的数据...")
            for item in qualified:
                symbol = item['symbol']
                path = downloader.download_file(symbol, date_str, 'aggTrades')
                if path:
                    print(f"  ✓ {symbol}")
                else:
                    print(f"  ✗ {symbol} (下载失败)")

        current_date += timedelta(days=1)
        time.sleep(0.5)  # 避免请求过快

    # 保存筛选结果
    result_file = downloader.data_dir / 'daily_symbols.json'
    with open(result_file, 'w') as f:
        json.dump(daily_symbols, f, indent=2)
    print(f"\n筛选结果已保存到: {result_file}")

    # 统计
    print(f"\n{'='*70}")
    print("下载统计")
    print(f"{'='*70}")
    total_symbols = set()
    for symbols in daily_symbols.values():
        total_symbols.update(symbols)
    print(f"总天数: {len(daily_symbols)}")
    print(f"涉及合约数: {len(total_symbols)}")
    print(f"平均每天合约数: {sum(len(s) for s in daily_symbols.values()) / len(daily_symbols):.1f}")


if __name__ == '__main__':
    main()
