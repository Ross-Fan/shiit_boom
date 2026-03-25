#!/usr/bin/env python3
"""
优化版数据下载器

优化点：
1. 通过K线API获取日交易量，快速筛选符合条件的合约
2. 只下载筛选后的合约数据，大幅减少下载量
3. 支持指定日期下载

使用方法:
    python data_downloader_v2.py 20260319
    python data_downloader_v2.py 20260319 20260320  # 多个日期
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
from typing import List, Dict, Optional, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent))
from config import MIN_DAILY_VOLUME_USDT, MAX_DAILY_VOLUME_USDT, DATA_DIR


class OptimizedDownloader:
    """优化版数据下载器"""

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })

        # API端点
        self.fapi_base = "https://fapi.binance.com"
        self.data_base = "https://data.binance.vision/data/futures/um/daily"

    def get_all_symbols(self) -> List[str]:
        """获取所有USDT永续合约"""
        url = f"{self.fapi_base}/fapi/v1/exchangeInfo"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            symbols = [
                s['symbol'] for s in data['symbols']
                if s['quoteAsset'] == 'USDT'
                and s['contractType'] == 'PERPETUAL'
                and s['status'] == 'TRADING'
            ]
            return symbols
        except Exception as e:
            print(f"获取合约列表失败: {e}")
            return []

    def get_daily_volume_by_kline(self, symbol: str, date: str) -> Optional[float]:
        """
        通过K线API获取指定日期的交易量（USDT）

        这是关键优化：用API获取交易量，避免下载完整数据

        Args:
            symbol: 交易对，如 'BTCUSDT'
            date: 日期，如 '2026-03-19'

        Returns:
            当日交易量（USDT），失败返回None
        """
        # 将日期转换为时间戳
        dt = datetime.strptime(date, '%Y-%m-%d')
        start_ts = int(dt.timestamp() * 1000)
        end_ts = int((dt + timedelta(days=1)).timestamp() * 1000) - 1

        url = f"{self.fapi_base}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': '1d',
            'startTime': start_ts,
            'endTime': end_ts,
            'limit': 1
        }

        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 400:
                # 合约可能在该日期不存在
                return None
            resp.raise_for_status()
            data = resp.json()

            if data and len(data) > 0:
                # K线数据格式: [开盘时间, 开, 高, 低, 收, 成交量, 收盘时间, 成交额, ...]
                # 索引7是成交额（quote asset volume，即USDT）
                volume_usdt = float(data[0][7])
                return volume_usdt
            return None

        except Exception as e:
            return None

    def filter_symbols_by_volume_api(
        self,
        symbols: List[str],
        date: str,
        min_volume: float = MIN_DAILY_VOLUME_USDT,
        max_volume: float = MAX_DAILY_VOLUME_USDT,
        max_workers: int = 20
    ) -> List[Dict]:
        """
        通过API快速筛选符合交易量条件的合约

        Args:
            symbols: 合约列表
            date: 日期
            min_volume: 最小交易量
            max_volume: 最大交易量
            max_workers: 并发数

        Returns:
            符合条件的合约列表 [{'symbol': 'XXX', 'volume': 123456}, ...]
        """
        print(f"\n通过API获取 {date} 的交易量数据...")
        print(f"筛选条件: {min_volume/1e6:.1f}M - {max_volume/1e6:.1f}M USDT")

        results = []
        checked = 0

        def check_volume(symbol):
            vol = self.get_daily_volume_by_kline(symbol, date)
            return symbol, vol

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_volume, s): s for s in symbols}

            for future in as_completed(futures):
                checked += 1
                symbol, volume = future.result()

                if volume is not None and min_volume <= volume <= max_volume:
                    results.append({'symbol': symbol, 'volume': volume})
                    print(f"  ✓ {symbol}: {volume/1e6:.2f}M USDT")

                # 进度显示
                if checked % 50 == 0:
                    print(f"  进度: {checked}/{len(symbols)} ({len(results)} 符合条件)")

        print(f"\n筛选完成: {len(results)}/{len(symbols)} 个合约符合条件")
        return sorted(results, key=lambda x: x['volume'], reverse=True)

    def download_aggTrades(self, symbol: str, date: str) -> Optional[Path]:
        """下载aggTrades数据"""
        filename = f"{symbol}-aggTrades-{date}.zip"
        url = f"{self.data_base}/aggTrades/{symbol}/{filename}"

        local_dir = self.data_dir / symbol
        local_dir.mkdir(parents=True, exist_ok=True)

        zip_path = local_dir / filename
        csv_path = local_dir / f"{symbol}-aggTrades-{date}.csv"

        # 如果CSV已存在则跳过
        if csv_path.exists():
            return csv_path

        try:
            resp = self.session.get(url, timeout=120)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            # 保存并解压
            with open(zip_path, 'wb') as f:
                f.write(resp.content)

            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(local_dir)

            zip_path.unlink()  # 删除zip
            return csv_path

        except Exception as e:
            if zip_path.exists():
                zip_path.unlink()
            return None

    def download_date(
        self,
        date_str: str,
        min_volume: float = MIN_DAILY_VOLUME_USDT,
        max_volume: float = MAX_DAILY_VOLUME_USDT
    ) -> Dict:
        """
        下载指定日期的数据

        Args:
            date_str: 日期，格式 'YYYYMMDD' 或 'YYYY-MM-DD'

        Returns:
            {'date': ..., 'symbols': [...], 'downloaded': [...], 'failed': [...]}
        """
        # 标准化日期格式
        if len(date_str) == 8:
            date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        else:
            date = date_str

        print(f"\n{'='*70}")
        print(f"处理日期: {date}")
        print(f"{'='*70}")

        # 获取所有合约
        all_symbols = self.get_all_symbols()
        if not all_symbols:
            return {'date': date, 'error': '无法获取合约列表'}

        print(f"获取到 {len(all_symbols)} 个USDT永续合约")

        # 通过API筛选符合条件的合约
        qualified = self.filter_symbols_by_volume_api(
            all_symbols, date, min_volume, max_volume
        )

        if not qualified:
            return {'date': date, 'symbols': [], 'downloaded': [], 'failed': []}

        # 下载符合条件的合约数据
        print(f"\n开始下载 {len(qualified)} 个合约的aggTrades数据...")

        downloaded = []
        failed = []

        for i, item in enumerate(qualified, 1):
            symbol = item['symbol']
            print(f"  [{i}/{len(qualified)}] {symbol}...", end=' ')

            path = self.download_aggTrades(symbol, date)
            if path:
                downloaded.append(symbol)
                print(f"✓ ({path.stat().st_size / 1e6:.1f}MB)")
            else:
                failed.append(symbol)
                print("✗ (下载失败)")

            time.sleep(0.1)  # 避免请求过快

        result = {
            'date': date,
            'symbols': [q['symbol'] for q in qualified],
            'volumes': {q['symbol']: q['volume'] for q in qualified},
            'downloaded': downloaded,
            'failed': failed
        }

        # 保存结果
        result_file = self.data_dir / f"download_result_{date.replace('-', '')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n下载完成: {len(downloaded)} 成功, {len(failed)} 失败")
        print(f"结果保存到: {result_file}")

        return result


def main():
    """主程序"""
    if len(sys.argv) < 2:
        print("用法: python data_downloader_v2.py <日期> [日期2] [日期3] ...")
        print("日期格式: YYYYMMDD 或 YYYY-MM-DD")
        print("示例: python data_downloader_v2.py 20260319")
        print("      python data_downloader_v2.py 20260319 20260320")
        sys.exit(1)

    dates = sys.argv[1:]

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         优化版数据下载器                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  优化点:                                                                      ║
║  1. 通过K线API获取日交易量，快速筛选（无需下载完整数据）                          ║
║  2. 只下载符合条件的合约，大幅减少下载量                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    downloader = OptimizedDownloader()

    all_results = {}

    for date_str in dates:
        result = downloader.download_date(date_str)
        all_results[result.get('date', date_str)] = result

    # 汇总
    print(f"\n{'='*70}")
    print("下载汇总")
    print(f"{'='*70}")

    total_symbols = set()
    total_downloaded = 0
    total_failed = 0

    for date, result in all_results.items():
        if 'error' in result:
            print(f"  {date}: 错误 - {result['error']}")
        else:
            total_symbols.update(result.get('symbols', []))
            total_downloaded += len(result.get('downloaded', []))
            total_failed += len(result.get('failed', []))
            print(f"  {date}: {len(result.get('downloaded', []))} 个合约")

    print(f"\n总计:")
    print(f"  - 处理天数: {len(dates)}")
    print(f"  - 涉及合约: {len(total_symbols)}")
    print(f"  - 下载成功: {total_downloaded}")
    print(f"  - 下载失败: {total_failed}")

    # 保存汇总结果
    summary_file = Path(DATA_DIR) / 'daily_symbols.json'

    # 合并已有结果
    if summary_file.exists():
        with open(summary_file) as f:
            existing = json.load(f)
    else:
        existing = {}

    for date, result in all_results.items():
        if 'symbols' in result:
            existing[date] = result['symbols']

    with open(summary_file, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"\n合约列表已更新: {summary_file}")


if __name__ == '__main__':
    main()
