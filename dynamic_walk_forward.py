#!/usr/bin/env python3
"""
动态滚动前向验证系统

完整流程：
┌─────────────────────────────────────────────────────────────────────────────┐
│                         动态滚动前向验证                                      │
│                                                                             │
│  每日流程:                                                                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  筛选    │───▶│  下载    │───▶│  学习    │───▶│  进化    │              │
│  │  合约    │    │  数据    │    │  模型    │    │  参数    │              │
│  │(API获取) │    │(aggTrades)│   │(累积)    │    │(自动)    │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│                                                                             │
│  滚动验证:                                                                   │
│                                                                             │
│  训练期 (Day 1-10):                                                          │
│    ├─ Day 1: 筛选100万-1000万合约 → 下载 → 学习                              │
│    ├─ Day 2: 筛选（可能不同）→ 下载 → 累积学习                               │
│    └─ Day 10: 模型就绪                                                      │
│                                                                             │
│  验证期 (Day 11-20):                                                         │
│    ├─ Day 11: 筛选当天合约（可能和训练期不同）→ 预测 → 记录                   │
│    └─ Day 20: 统计真实收益 → 分析 → 自我进化                                 │
│                                                                             │
│  关键：特征是相对值（百分位/Z-score），可跨合约迁移！                          │
└─────────────────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
import json
import zipfile
import requests
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ============================================================
# 配置
# ============================================================
DEFAULT_CONFIG = {
    # 合约筛选
    'min_volume_usdt': 1_000_000,   # 最小日交易量 100万U
    'max_volume_usdt': 10_000_000,  # 最大日交易量 1000万U

    # 特征窗口
    'windows': [15, 30, 60],

    # 交易参数
    'target_profit': 3.0,
    'stop_loss': -3.0,
    'hold_minutes': 30,

    # 模型参数
    'min_precision': 0.6,
    'evolution_rounds': 3,

    # 滚动验证
    'train_days': 10,
    'test_days': 10,
}


# ============================================================
# 数据结构
# ============================================================
@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    date: str
    entry_time: str
    entry_price: float
    exit_price: float
    return_pct: float
    exit_reason: str
    signal_proba: float
    hour: int = 0


@dataclass
class DailyContractInfo:
    """每日合约信息"""
    date: str
    symbol: str
    volume_usdt: float
    data_path: Optional[str] = None


@dataclass
class PeriodResult:
    """周期结果"""
    period: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_contracts: int = 0
    test_contracts: int = 0
    train_samples: int = 0
    test_trades: int = 0
    test_win_rate: float = 0.0
    test_return: float = 0.0
    threshold: float = 0.5
    blacklist_hours: List[int] = field(default_factory=list)


# ============================================================
# 合约筛选器
# ============================================================
class ContractFilter:
    """合约筛选器 - 通过API获取符合条件的合约"""

    def __init__(self, min_volume: float, max_volume: float):
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.fapi_base = "https://fapi.binance.com"

        # 缓存
        self._all_symbols_cache = None
        self._volume_cache: Dict[str, Dict[str, float]] = {}  # {date: {symbol: volume}}

    def get_all_symbols(self) -> List[str]:
        """获取所有USDT永续合约"""
        if self._all_symbols_cache:
            return self._all_symbols_cache

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
            self._all_symbols_cache = symbols
            return symbols
        except Exception as e:
            print(f"  获取合约列表失败: {e}")
            return []

    def get_daily_volume(self, symbol: str, date: str) -> Optional[float]:
        """通过K线API获取指定日期的交易量（USDT）"""
        # 检查缓存
        if date in self._volume_cache and symbol in self._volume_cache[date]:
            return self._volume_cache[date][symbol]

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
                return None
            resp.raise_for_status()
            data = resp.json()

            if data and len(data) > 0:
                volume_usdt = float(data[0][7])

                # 缓存
                if date not in self._volume_cache:
                    self._volume_cache[date] = {}
                self._volume_cache[date][symbol] = volume_usdt

                return volume_usdt
            return None
        except:
            return None

    def filter_contracts_for_date(
        self,
        date: str,
        max_workers: int = 20,
        progress_callback=None
    ) -> List[DailyContractInfo]:
        """筛选指定日期符合条件的合约"""
        all_symbols = self.get_all_symbols()
        if not all_symbols:
            return []

        results = []
        checked = 0

        def check_volume(symbol):
            vol = self.get_daily_volume(symbol, date)
            return symbol, vol

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_volume, s): s for s in all_symbols}

            for future in as_completed(futures):
                checked += 1
                symbol, volume = future.result()

                if volume is not None and self.min_volume <= volume <= self.max_volume:
                    results.append(DailyContractInfo(
                        date=date,
                        symbol=symbol,
                        volume_usdt=volume
                    ))

                if progress_callback and checked % 50 == 0:
                    progress_callback(checked, len(all_symbols), len(results))

        return sorted(results, key=lambda x: x.volume_usdt, reverse=True)


# ============================================================
# 数据下载器
# ============================================================
class DataDownloader:
    """数据下载器"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0'})
        self.data_base = "https://data.binance.vision/data/futures/um/daily"

    def download_aggTrades(self, symbol: str, date: str, timeout: int = 60) -> Optional[Path]:
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
            # 使用流式下载，设置超时
            resp = self.session.get(url, timeout=timeout, stream=True)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()

            # 写入文件
            with open(zip_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            # 解压
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(local_dir)

            zip_path.unlink()
            return csv_path

        except requests.exceptions.Timeout:
            if zip_path.exists():
                zip_path.unlink()
            return None
        except requests.exceptions.RequestException:
            if zip_path.exists():
                zip_path.unlink()
            return None
        except zipfile.BadZipFile:
            if zip_path.exists():
                zip_path.unlink()
            return None
        except Exception:
            if zip_path.exists():
                zip_path.unlink()
            return None

    def load_csv_to_minutes(self, csv_path: Path) -> pd.DataFrame:
        """将CSV转换为分钟K线"""
        try:
            df = pd.read_csv(csv_path)
            df['datetime'] = pd.to_datetime(df['transact_time'], unit='ms')
            df['minute'] = df['datetime'].dt.floor('1min')

            symbol = csv_path.stem.split('-')[0]

            agg = df.groupby('minute').agg({
                'price': ['min', 'max', 'first', 'last'],
                'quantity': 'sum',
                'is_buyer_maker': lambda x: (
                    df.loc[x.index, 'quantity'] *
                    (1 - df.loc[x.index, 'is_buyer_maker'].map(
                        {True: 1, False: 0, 'true': 1, 'false': 0}
                    ))
                ).sum()
            }).reset_index()

            agg.columns = ['minute', 'low', 'high', 'open', 'close', 'volume', 'buy_volume']
            agg['sell_volume'] = agg['volume'] - agg['buy_volume']
            agg['symbol'] = symbol
            agg['hour'] = agg['minute'].dt.hour
            agg['date'] = csv_path.stem.split('-')[-1]  # 从文件名提取日期

            return agg
        except Exception as e:
            print(f"    加载失败 {csv_path}: {e}")
            return pd.DataFrame()


# ============================================================
# 特征工程
# ============================================================
class FeatureEngine:
    """特征引擎 - 生成相对特征，可跨合约迁移"""

    def __init__(self, windows: List[int] = [15, 30, 60]):
        self.windows = windows
        self.feature_names = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征"""
        result = df.copy()

        # 基础指标
        result['net_buy'] = result['buy_volume'] - result['sell_volume']
        result['bsr'] = result['buy_volume'] / result['sell_volume'].replace(0, 0.001)
        result['volatility'] = (result['high'] - result['low']) / result['close']

        features = []

        for w in self.windows:
            # Z-score（相对于自身历史）
            for col in ['volume', 'bsr', 'net_buy', 'volatility']:
                fname = f'{col}_z_{w}'
                mean = result[col].rolling(w).mean()
                std = result[col].rolling(w).std().replace(0, 1)
                result[fname] = (result[col] - mean) / std
                features.append(fname)

            # 百分位（相对于自身历史）
            for col in ['volume', 'bsr', 'net_buy']:
                fname = f'{col}_pct_{w}'
                result[fname] = result[col].rolling(w+1).apply(
                    lambda x: (x[:-1] < x[-1]).sum() / (len(x)-1) if len(x) > 1 else 0.5,
                    raw=True
                )
                features.append(fname)

            # 价格位置（相对于自身区间）
            fname = f'price_pos_{w}'
            high_roll = result['high'].rolling(w).max()
            low_roll = result['low'].rolling(w).min()
            result[fname] = (result['close'] - low_roll) / (high_roll - low_roll).replace(0, 1)
            features.append(fname)

            # 动量
            fname = f'momentum_{w}'
            result[fname] = result['close'] / result['close'].shift(w) - 1
            features.append(fname)

            # 买入压力
            fname = f'buy_pressure_{w}'
            result[fname] = (result['net_buy'] > 0).rolling(w).mean()
            features.append(fname)

            # 成交量比率
            fname = f'vol_ratio_{w}'
            result[fname] = result['volume'] / result['volume'].rolling(w).mean().replace(0, 1)
            features.append(fname)

        self.feature_names = features
        return result


# ============================================================
# 动态滚动前向验证系统
# ============================================================
class DynamicWalkForwardSystem:
    """动态滚动前向验证系统"""

    def __init__(self, config: Dict = None, data_dir: str = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.data_dir = Path(data_dir) if data_dir else Path('data')

        # 组件
        self.contract_filter = ContractFilter(
            self.config['min_volume_usdt'],
            self.config['max_volume_usdt']
        )
        self.downloader = DataDownloader(str(self.data_dir))
        self.feature_engine = FeatureEngine(self.config['windows'])

        # 模型状态
        self.model = None
        self.threshold = 0.5
        self.blacklist_hours: List[int] = []
        self.blacklist_symbols: List[str] = []

        # 结果
        self.period_results: List[PeriodResult] = []
        self.all_trades: List[TradeRecord] = []

        # 缓存
        self._daily_contracts_cache: Dict[str, List[DailyContractInfo]] = {}
        self._minute_data_cache: Dict[str, pd.DataFrame] = {}

    def get_contracts_for_date(self, date: str) -> List[DailyContractInfo]:
        """获取指定日期符合条件的合约（带缓存）"""
        if date in self._daily_contracts_cache:
            return self._daily_contracts_cache[date]

        def progress(checked, total, qualified):
            print(f"\r    筛选进度: {checked}/{total} ({qualified} 符合条件)", end='')

        contracts = self.contract_filter.filter_contracts_for_date(
            date,
            max_workers=20,
            progress_callback=progress
        )
        print()  # 换行

        self._daily_contracts_cache[date] = contracts
        return contracts

    def download_and_load_data(
        self,
        contracts: List[DailyContractInfo],
        show_progress: bool = True,
        max_workers: int = 10
    ) -> pd.DataFrame:
        """下载并加载合约数据（并行下载）"""
        if not contracts:
            return pd.DataFrame()

        all_data = []
        failed_downloads = []
        success_count = 0
        total = len(contracts)

        def download_one(contract):
            """下载单个合约"""
            csv_path = self.downloader.download_aggTrades(contract.symbol, contract.date, timeout=60)
            return contract, csv_path

        # 并行下载
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_one, c): c for c in contracts}

            for future in as_completed(futures):
                contract = futures[future]
                try:
                    contract, csv_path = future.result(timeout=90)

                    if csv_path is not None:
                        contract.data_path = str(csv_path)

                        # 加载数据
                        cache_key = f"{contract.symbol}_{contract.date}"
                        if cache_key in self._minute_data_cache:
                            df = self._minute_data_cache[cache_key]
                        else:
                            df = self.downloader.load_csv_to_minutes(csv_path)
                            if not df.empty:
                                self._minute_data_cache[cache_key] = df

                        if not df.empty:
                            all_data.append(df)
                            success_count += 1
                    else:
                        failed_downloads.append(contract.symbol)

                except Exception as e:
                    failed_downloads.append(contract.symbol)

                # 更新进度
                if show_progress:
                    done = success_count + len(failed_downloads)
                    print(f"\r    下载进度: {done}/{total} (成功: {success_count}, 失败: {len(failed_downloads)})", end='')
                    sys.stdout.flush()

        if show_progress:
            print()  # 换行
            if failed_downloads and len(failed_downloads) <= 10:
                print(f"    下载失败的合约: {failed_downloads}")
            elif failed_downloads:
                print(f"    下载失败: {len(failed_downloads)} 个合约")

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """生成标签"""
        labels = []
        tp = self.config['target_profit']
        sl = self.config['stop_loss']
        hold = self.config['hold_minutes']

        for i in range(len(df)):
            if i + hold >= len(df):
                labels.append(np.nan)
                continue

            # 确保同一symbol
            if df.iloc[i]['symbol'] != df.iloc[i + hold]['symbol']:
                labels.append(np.nan)
                continue

            entry = df.iloc[i]['close']
            label = 0

            for j in range(1, hold + 1):
                if i + j >= len(df):
                    break
                bar = df.iloc[i + j]
                high_ret = (bar['high'] - entry) / entry * 100
                low_ret = (bar['low'] - entry) / entry * 100

                if high_ret >= tp:
                    label = 1
                    break
                if low_ret <= sl:
                    label = 0
                    break

            labels.append(label)

        return pd.Series(labels, index=df.index)

    def _compute_features_by_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """按symbol分组计算特征（避免跨symbol计算，提高效率）"""
        all_featured = []
        symbols = df['symbol'].unique()
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if (i + 1) % 50 == 0:
                print(f"\r    特征计算进度: {i+1}/{total} ({symbol})", end='')
                sys.stdout.flush()

            symbol_df = df[df['symbol'] == symbol].copy()
            if len(symbol_df) < self.config['windows'][-1] + 10:
                continue

            # 按时间排序
            symbol_df = symbol_df.sort_values('minute').reset_index(drop=True)

            # 计算特征
            featured = self.feature_engine.compute(symbol_df)
            all_featured.append(featured)

        print()  # 换行
        if not all_featured:
            return pd.DataFrame()

        return pd.concat(all_featured, ignore_index=True)

    def _generate_labels_fast(self, df: pd.DataFrame) -> pd.Series:
        """快速生成标签（按symbol分组处理）"""
        tp = self.config['target_profit']
        sl = self.config['stop_loss']
        hold = self.config['hold_minutes']

        labels = pd.Series(index=df.index, dtype=float)
        labels[:] = np.nan

        symbols = df['symbol'].unique()
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            if (i + 1) % 100 == 0:
                print(f"\r    标签生成进度: {i+1}/{total}", end='')
                sys.stdout.flush()

            mask = df['symbol'] == symbol
            symbol_df = df[mask]
            idx = symbol_df.index

            if len(symbol_df) <= hold:
                continue

            # 使用numpy加速
            closes = symbol_df['close'].values
            highs = symbol_df['high'].values
            lows = symbol_df['low'].values

            symbol_labels = []
            for j in range(len(symbol_df)):
                if j + hold >= len(symbol_df):
                    symbol_labels.append(np.nan)
                    continue

                entry = closes[j]
                label = 0

                for k in range(1, hold + 1):
                    high_ret = (highs[j + k] - entry) / entry * 100
                    low_ret = (lows[j + k] - entry) / entry * 100

                    if high_ret >= tp:
                        label = 1
                        break
                    if low_ret <= sl:
                        label = 0
                        break

                symbol_labels.append(label)

            labels.loc[idx] = symbol_labels

        print()  # 换行
        return labels

    def train_model(self, X: pd.DataFrame, y: pd.Series, warm_start: bool = False) -> Dict:
        """训练模型"""
        if not HAS_LGB:
            raise ImportError("需要安装lightgbm")

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        # 时间序列分割
        split = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # 累积学习
        if warm_start and self.model is not None:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=100,
                valid_sets=[val_data],
                init_model=self.model,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        else:
            self.model = lgb.train(
                params, train_data,
                num_boost_round=300,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

        # 找最优阈值
        val_pred = self.model.predict(X_val)
        y_val_np = y_val.values

        best_threshold = 0.5
        best_score = -float('inf')

        for th in np.arange(0.3, 0.8, 0.02):
            pred = (val_pred >= th).astype(int)
            tp = ((pred == 1) & (y_val_np == 1)).sum()
            fp = ((pred == 1) & (y_val_np == 0)).sum()
            precision = tp / max(tp + fp, 1)

            score = precision * np.sqrt(tp + fp) if precision >= self.config['min_precision'] else 0

            if score > best_score:
                best_score = score
                best_threshold = th

        self.threshold = best_threshold

        # 计算指标
        val_pred_binary = (val_pred >= best_threshold).astype(int)
        tp = ((val_pred_binary == 1) & (y_val_np == 1)).sum()
        fp = ((val_pred_binary == 1) & (y_val_np == 0)).sum()
        fn = ((val_pred_binary == 0) & (y_val_np == 1)).sum()

        return {
            'threshold': best_threshold,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1),
            'n_samples': len(X),
        }

    def backtest(self, df: pd.DataFrame) -> List[TradeRecord]:
        """执行回测"""
        if self.model is None or df.empty:
            return []

        features = self.feature_engine.feature_names
        X = df[features]

        valid_mask = ~X.isna().any(axis=1)
        predictions = np.zeros(len(df))
        predictions[valid_mask] = self.model.predict(X[valid_mask])

        signals = predictions >= self.threshold

        # 应用过滤
        if self.blacklist_symbols:
            symbol_mask = ~df['symbol'].isin(self.blacklist_symbols)
            signals = signals & symbol_mask.values

        if self.blacklist_hours:
            hour_mask = ~df['hour'].isin(self.blacklist_hours)
            signals = signals & hour_mask.values

        # 模拟交易
        trades = []
        symbol_cooldown = defaultdict(int)
        cooldown = self.config['hold_minutes']

        for i in np.where(signals)[0]:
            symbol = df.iloc[i]['symbol']

            if i < symbol_cooldown[symbol]:
                continue

            if i + self.config['hold_minutes'] >= len(df):
                continue

            if df.iloc[i + self.config['hold_minutes']]['symbol'] != symbol:
                continue

            entry_price = df.iloc[i]['close'] * 1.001  # 滑点

            exit_price = None
            exit_reason = 'timeout'

            for h in range(1, self.config['hold_minutes'] + 1):
                bar = df.iloc[i + h]
                high_ret = (bar['high'] - entry_price) / entry_price * 100
                low_ret = (bar['low'] - entry_price) / entry_price * 100

                if high_ret >= self.config['target_profit']:
                    exit_price = entry_price * (1 + self.config['target_profit'] / 100)
                    exit_reason = 'take_profit'
                    break
                if low_ret <= self.config['stop_loss']:
                    exit_price = entry_price * (1 + self.config['stop_loss'] / 100)
                    exit_reason = 'stop_loss'
                    break

            if exit_price is None:
                exit_price = df.iloc[i + self.config['hold_minutes']]['close']

            ret = (exit_price - entry_price) / entry_price * 100 - 0.08  # 手续费

            trades.append(TradeRecord(
                symbol=symbol,
                date=df.iloc[i]['date'],
                entry_time=str(df.iloc[i]['minute']),
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=ret,
                exit_reason=exit_reason,
                signal_proba=predictions[i],
                hour=df.iloc[i]['hour']
            ))

            symbol_cooldown[symbol] = i + cooldown

        return trades

    def evolve(self, trades: List[TradeRecord]):
        """根据交易结果进化"""
        if not trades:
            return

        trades_df = pd.DataFrame([asdict(t) for t in trades])

        # 分析时段
        hour_stats = trades_df.groupby('hour')['return_pct'].agg(['count', 'mean'])
        bad_hours = hour_stats[hour_stats['mean'] < -0.5].index.tolist()

        for h in bad_hours:
            if h not in self.blacklist_hours:
                self.blacklist_hours.append(h)
                print(f"    → 将时段 {h} 加入黑名单")

        # 分析币种
        symbol_stats = trades_df.groupby('symbol')['return_pct'].agg(['count', 'mean'])
        bad_symbols = symbol_stats[(symbol_stats['count'] >= 3) & (symbol_stats['mean'] < -0.5)].index.tolist()

        for s in bad_symbols:
            if s not in self.blacklist_symbols:
                self.blacklist_symbols.append(s)
                print(f"    → 将币种 {s} 加入黑名单")

        # 调整阈值
        precision = (trades_df['return_pct'] > 0).mean()
        if precision < self.config['min_precision']:
            old_threshold = self.threshold
            self.threshold = min(self.threshold + 0.03, 0.8)
            print(f"    → 提高阈值: {old_threshold:.2f} → {self.threshold:.2f}")

    def run(self, start_date: str, end_date: str) -> Dict:
        """
        运行完整的动态滚动前向验证

        Args:
            start_date: 起始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        train_days = self.config['train_days']
        test_days = self.config['test_days']

        print("=" * 70)
        print("动态滚动前向验证系统")
        print("=" * 70)
        print(f"  时间范围: {start_date} 至 {end_date}")
        print(f"  交易量筛选: {self.config['min_volume_usdt']/1e6:.0f}M - {self.config['max_volume_usdt']/1e6:.0f}M USDT")
        print(f"  训练周期: {train_days} 天")
        print(f"  验证周期: {test_days} 天")
        print("=" * 70)

        period = 0
        cumulative_train_data = None
        all_train_contracts: Set[str] = set()

        while True:
            period += 1
            train_end = start + timedelta(days=train_days * period - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days - 1)

            if test_end > end:
                print(f"\n验证期超出范围，停止")
                break

            print(f"\n{'='*70}")
            print(f"第 {period} 轮")
            print(f"{'='*70}")
            print(f"  训练期: {start.strftime('%Y-%m-%d')} 至 {train_end.strftime('%Y-%m-%d')}")
            print(f"  验证期: {test_start.strftime('%Y-%m-%d')} 至 {test_end.strftime('%Y-%m-%d')}")

            # ========== 训练阶段 ==========
            print(f"\n[训练阶段]")

            # 获取训练期新增数据
            if period == 1:
                # 第一轮：加载全部训练期数据
                new_train_start = start
            else:
                # 后续轮：只加载新增部分
                prev_train_end = start + timedelta(days=train_days * (period - 1) - 1)
                new_train_start = prev_train_end + timedelta(days=1)

            new_train_data_list = []
            new_contracts_count = 0

            current = new_train_start
            while current <= train_end:
                date_str = current.strftime('%Y-%m-%d')
                print(f"\n  [{date_str}] 筛选合约...")

                contracts = self.get_contracts_for_date(date_str)
                print(f"    找到 {len(contracts)} 个符合条件的合约")

                if contracts:
                    print(f"    下载数据...")
                    day_data = self.download_and_load_data(contracts, show_progress=True)

                    if not day_data.empty:
                        new_train_data_list.append(day_data)
                        new_contracts_count += len(contracts)
                        for c in contracts:
                            all_train_contracts.add(c.symbol)

                current += timedelta(days=1)

            if not new_train_data_list:
                print(f"  无新训练数据，跳过此轮")
                continue

            # 累积数据
            print(f"\n  合并数据中...")
            sys.stdout.flush()
            new_train_data = pd.concat(new_train_data_list, ignore_index=True)

            if cumulative_train_data is None:
                cumulative_train_data = new_train_data
            else:
                cumulative_train_data = pd.concat(
                    [cumulative_train_data, new_train_data],
                    ignore_index=True
                )

            print(f"  累积训练数据: {len(cumulative_train_data)} 条")
            print(f"  累积合约数: {len(all_train_contracts)}")
            sys.stdout.flush()

            # 计算特征（按symbol分组处理，避免跨symbol计算）
            print(f"\n  计算特征中（这可能需要几分钟）...")
            sys.stdout.flush()
            train_featured = self._compute_features_by_symbol(cumulative_train_data)

            print(f"  生成标签中...")
            sys.stdout.flush()
            train_featured['label'] = self._generate_labels_fast(train_featured)
            train_valid = train_featured.dropna(subset=['label'] + self.feature_engine.feature_names)

            if len(train_valid) < 100:
                print(f"  有效样本不足 ({len(train_valid)}), 跳过")
                continue

            X_train = train_valid[self.feature_engine.feature_names]
            y_train = train_valid['label'].astype(int)

            # 训练模型
            print(f"\n[模型训练]")
            warm_start = period > 1
            train_result = self.train_model(X_train, y_train, warm_start=warm_start)
            print(f"  阈值: {train_result['threshold']:.2f}")
            print(f"  精确率: {train_result['precision']:.2%}")
            print(f"  召回率: {train_result['recall']:.2%}")

            # 内部进化
            print(f"\n[内部进化]")
            for _ in range(self.config['evolution_rounds']):
                internal_trades = self.backtest(train_valid)
                if internal_trades:
                    self.evolve(internal_trades)

            # ========== 验证阶段 ==========
            print(f"\n[验证阶段] - 真实样本外测试")

            test_data_list = []
            test_contracts_set: Set[str] = set()

            current = test_start
            while current <= test_end:
                date_str = current.strftime('%Y-%m-%d')
                print(f"\n  [{date_str}] 筛选验证期合约...")

                contracts = self.get_contracts_for_date(date_str)
                print(f"    找到 {len(contracts)} 个符合条件的合约")

                # 统计和训练期的重叠
                overlap = sum(1 for c in contracts if c.symbol in all_train_contracts)
                new_contracts = len(contracts) - overlap
                print(f"    其中: {overlap} 个在训练期出现过, {new_contracts} 个新合约")

                if contracts:
                    print(f"    下载数据...")
                    day_data = self.download_and_load_data(contracts, show_progress=True)

                    if not day_data.empty:
                        test_data_list.append(day_data)
                        for c in contracts:
                            test_contracts_set.add(c.symbol)

                current += timedelta(days=1)

            if not test_data_list:
                print(f"\n  无验证数据")
                result = PeriodResult(
                    period=period,
                    train_start=start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    test_start=test_start.strftime('%Y-%m-%d'),
                    test_end=test_end.strftime('%Y-%m-%d'),
                    train_contracts=len(all_train_contracts),
                    train_samples=len(train_valid),
                    threshold=self.threshold,
                    blacklist_hours=self.blacklist_hours.copy(),
                )
                self.period_results.append(result)
                continue

            test_data = pd.concat(test_data_list, ignore_index=True)

            # 计算特征
            print(f"  计算验证数据特征...")
            sys.stdout.flush()
            test_featured = self._compute_features_by_symbol(test_data)
            test_valid = test_featured.dropna(subset=self.feature_engine.feature_names)

            # 执行验证
            test_trades = self.backtest(test_valid)
            self.all_trades.extend(test_trades)

            if test_trades:
                wins = sum(1 for t in test_trades if t.return_pct > 0)
                total_ret = sum(t.return_pct for t in test_trades)
                win_rate = wins / len(test_trades)

                print(f"\n  验证结果:")
                print(f"    合约数: {len(test_contracts_set)}")
                print(f"    交易数: {len(test_trades)}")
                print(f"    胜率: {win_rate*100:.1f}%")
                print(f"    总收益: {total_ret:.2f}%")

                result = PeriodResult(
                    period=period,
                    train_start=start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    test_start=test_start.strftime('%Y-%m-%d'),
                    test_end=test_end.strftime('%Y-%m-%d'),
                    train_contracts=len(all_train_contracts),
                    test_contracts=len(test_contracts_set),
                    train_samples=len(train_valid),
                    test_trades=len(test_trades),
                    test_win_rate=win_rate,
                    test_return=total_ret,
                    threshold=self.threshold,
                    blacklist_hours=self.blacklist_hours.copy(),
                )

                # 用验证期结果继续进化
                print(f"\n[验证后进化]")
                self.evolve(test_trades)
            else:
                print(f"\n  无交易信号")
                result = PeriodResult(
                    period=period,
                    train_start=start.strftime('%Y-%m-%d'),
                    train_end=train_end.strftime('%Y-%m-%d'),
                    test_start=test_start.strftime('%Y-%m-%d'),
                    test_end=test_end.strftime('%Y-%m-%d'),
                    train_contracts=len(all_train_contracts),
                    test_contracts=len(test_contracts_set),
                    train_samples=len(train_valid),
                    threshold=self.threshold,
                    blacklist_hours=self.blacklist_hours.copy(),
                )

            self.period_results.append(result)

        # 汇总
        self._print_summary()

        return {
            'period_results': [asdict(r) for r in self.period_results],
            'all_trades': [asdict(t) for t in self.all_trades],
            'final_threshold': self.threshold,
            'blacklist_hours': self.blacklist_hours,
            'blacklist_symbols': self.blacklist_symbols,
        }

    def _print_summary(self):
        """打印汇总"""
        print(f"\n{'='*70}")
        print("动态滚动前向验证汇总")
        print(f"{'='*70}")

        if not self.period_results:
            print("无验证结果")
            return

        print(f"\n{'轮次':<4}{'训练期':<24}{'验证期':<24}{'合约':<6}{'交易':<6}{'胜率':<8}{'收益':<10}")
        print("-" * 82)

        total_trades = 0
        total_wins = 0
        total_return = 0

        for r in self.period_results:
            train_period = f"{r.train_start} ~ {r.train_end}"
            test_period = f"{r.test_start} ~ {r.test_end}"

            if r.test_trades > 0:
                print(f"{r.period:<4}{train_period:<24}{test_period:<24}"
                      f"{r.test_contracts:<6}{r.test_trades:<6}{r.test_win_rate*100:>5.1f}%  {r.test_return:>+7.2f}%")
                total_trades += r.test_trades
                total_wins += int(r.test_trades * r.test_win_rate)
                total_return += r.test_return
            else:
                print(f"{r.period:<4}{train_period:<24}{test_period:<24}"
                      f"{r.test_contracts:<6}{'0':<6}{'N/A':<8}{'N/A':<10}")

        print("-" * 82)

        if total_trades > 0:
            overall_win_rate = total_wins / total_trades
            print(f"{'总计':<4}{'':<24}{'':<24}"
                  f"{'':<6}{total_trades:<6}{overall_win_rate*100:>5.1f}%  {total_return:>+7.2f}%")

            print(f"\n关键指标:")
            print(f"  - 总交易次数: {total_trades}")
            print(f"  - 整体胜率: {overall_win_rate*100:.1f}%")
            print(f"  - 累计收益: {total_return:.2f}%")
            print(f"  - 平均每轮收益: {total_return/len(self.period_results):.2f}%")

        print(f"\n学习到的知识:")
        if self.blacklist_hours:
            print(f"  - 黑名单时段: {self.blacklist_hours}")
        if self.blacklist_symbols:
            print(f"  - 黑名单币种: {self.blacklist_symbols[:10]}{'...' if len(self.blacklist_symbols) > 10 else ''}")
        print(f"  - 最终阈值: {self.threshold:.2f}")

    def save(self, path: str):
        """保存系统状态"""
        Path(path).mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_model(f"{path}/model.lgb")

        state = {
            'config': self.config,
            'threshold': self.threshold,
            'blacklist_hours': self.blacklist_hours,
            'blacklist_symbols': self.blacklist_symbols,
            'feature_names': self.feature_engine.feature_names,
            'period_results': [asdict(r) for r in self.period_results],
            'all_trades': [asdict(t) for t in self.all_trades],
        }

        with open(f"{path}/state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)

        print(f"\n系统已保存到: {path}")


def main():
    """主程序"""
    import sys

    # 默认参数
    start_date = '2026-03-19'
    end_date = '2026-03-20'
    data_dir = '/Users/fanwei/study/quants/bian_data/uai/backtest/data'

    # 命令行参数
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]

    if len(sys.argv) >= 4:
        data_dir = sys.argv[3]

    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    动态滚动前向验证系统                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  完整流程:                                                                    ║
║  1. 每天通过API筛选交易量100万-1000万的合约                                    ║
║  2. 下载符合条件的合约数据                                                    ║
║  3. 累积学习，训练ML模型                                                      ║
║  4. 在未见过的验证期数据上测试（验证期合约可能不同！）                           ║
║  5. 根据结果自我进化                                                          ║
║                                                                              ║
║  关键: 特征是相对值（百分位/Z-score），可跨合约迁移                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    if not HAS_LGB:
        print("请安装lightgbm: pip install lightgbm")
        return

    system = DynamicWalkForwardSystem(
        config={
            'min_volume_usdt': 1_000_000,
            'max_volume_usdt': 10_000_000,
            'train_days': 10,
            'test_days': 10,
        },
        data_dir=data_dir
    )

    result = system.run(start_date, end_date)

    # 保存
    save_dir = Path(data_dir).parent / 'dynamic_model'
    system.save(str(save_dir))


if __name__ == '__main__':
    main()
