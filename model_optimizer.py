#!/usr/bin/env python3
"""
策略优化框架

解决的问题：
1. 手动调参容易混乱 → 使用自动化优化算法
2. 多因子判断 → 使用机器学习模型自动学习最优组合
3. 预测要快 → 使用LightGBM（预测<1ms）+ 预计算特征

架构：
┌─────────────────────────────────────────────────────────────────┐
│                    特征工程层                                    │
│  原始数据 → 滚动统计 → 百分位 → 技术指标 → 特征向量               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    模型层 (LightGBM)                             │
│  特征向量 → 预测概率 → 信号                                      │
│  训练时自动学习最优阈值和特征权重                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    优化层 (Optuna)                               │
│  自动搜索最优超参数：模型参数、特征窗口、止盈止损等                 │
└─────────────────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 尝试导入可选依赖
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("提示: 安装lightgbm可获得更好的模型性能: pip install lightgbm")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("提示: 安装optuna可使用自动参数优化: pip install optuna")


# ============================================================
# 特征工程
# ============================================================
class FeatureEngineer:
    """
    特征工程类

    将原始交易数据转换为模型可用的特征向量
    所有特征都是相对值（百分位/Z-score），确保跨币种通用
    """

    def __init__(self, windows: List[int] = [15, 30, 60]):
        """
        Args:
            windows: 滚动窗口列表（分钟）
        """
        self.windows = windows
        self.feature_names = []

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征

        Args:
            df: 分钟K线数据，需包含 volume, buy_volume, sell_volume, close, high, low

        Returns:
            添加了特征列的DataFrame
        """
        result = df.copy()

        # 基础衍生指标
        result['net_buy'] = result['buy_volume'] - result['sell_volume']
        result['buy_sell_ratio'] = result['buy_volume'] / result['sell_volume'].replace(0, 0.001)
        result['volatility'] = (result['high'] - result['low']) / result['close']
        result['price_change'] = result['close'].pct_change()

        features = []

        for w in self.windows:
            suffix = f'_{w}'

            # 成交量相关
            result[f'vol_zscore{suffix}'] = self._rolling_zscore(result['volume'], w)
            result[f'vol_pct{suffix}'] = self._rolling_percentile(result['volume'], w)
            result[f'vol_ratio{suffix}'] = result['volume'] / result['volume'].rolling(w).mean()

            # 买卖比相关
            result[f'bsr_zscore{suffix}'] = self._rolling_zscore(result['buy_sell_ratio'], w)
            result[f'bsr_pct{suffix}'] = self._rolling_percentile(result['buy_sell_ratio'], w)

            # 净买入相关
            result[f'netbuy_zscore{suffix}'] = self._rolling_zscore(result['net_buy'], w)
            result[f'netbuy_pct{suffix}'] = self._rolling_percentile(result['net_buy'], w)

            # 价格位置
            high_roll = result['high'].rolling(w).max()
            low_roll = result['low'].rolling(w).min()
            result[f'price_position{suffix}'] = (result['close'] - low_roll) / (high_roll - low_roll).replace(0, 1)

            # 波动率
            result[f'volatility_zscore{suffix}'] = self._rolling_zscore(result['volatility'], w)

            # 价格动量
            result[f'momentum{suffix}'] = result['close'] / result['close'].shift(w) - 1

            # 连续买入压力（过去N分钟净买入为正的比例）
            result[f'buy_pressure{suffix}'] = (result['net_buy'] > 0).rolling(w).mean()

            features.extend([
                f'vol_zscore{suffix}', f'vol_pct{suffix}', f'vol_ratio{suffix}',
                f'bsr_zscore{suffix}', f'bsr_pct{suffix}',
                f'netbuy_zscore{suffix}', f'netbuy_pct{suffix}',
                f'price_position{suffix}', f'volatility_zscore{suffix}',
                f'momentum{suffix}', f'buy_pressure{suffix}'
            ])

        self.feature_names = features
        return result

    def _rolling_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """滚动Z-score"""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std().replace(0, 1)
        return (series - mean) / std

    def _rolling_percentile(self, series: pd.Series, window: int) -> pd.Series:
        """滚动百分位"""
        def pct_rank(x):
            if len(x) < 2:
                return 0.5
            return (x[:-1] < x[-1]).sum() / (len(x) - 1)
        return series.rolling(window + 1).apply(pct_rank, raw=True)

    def get_feature_names(self) -> List[str]:
        return self.feature_names


# ============================================================
# 标签生成
# ============================================================
class LabelGenerator:
    """
    生成训练标签

    正样本：未来N分钟内涨幅超过X%
    负样本：未来N分钟内未达到涨幅目标或触发止损
    """

    def __init__(
        self,
        target_profit: float = 3.0,
        max_loss: float = -3.0,
        hold_minutes: int = 30
    ):
        self.target_profit = target_profit
        self.max_loss = max_loss
        self.hold_minutes = hold_minutes

    def generate_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        生成标签

        Returns:
            Series: 1=正样本（达到止盈），0=负样本
        """
        labels = []

        for i in range(len(df)):
            if i + self.hold_minutes >= len(df):
                labels.append(np.nan)
                continue

            entry_price = df.iloc[i]['close']
            future_data = df.iloc[i+1:i+1+self.hold_minutes]

            # 检查是否达到止盈
            max_return = (future_data['high'].max() - entry_price) / entry_price * 100
            min_return = (future_data['low'].min() - entry_price) / entry_price * 100

            # 先触发止损则为负样本
            if min_return <= self.max_loss:
                # 检查是否止损前先止盈
                for _, bar in future_data.iterrows():
                    high_ret = (bar['high'] - entry_price) / entry_price * 100
                    low_ret = (bar['low'] - entry_price) / entry_price * 100
                    if high_ret >= self.target_profit:
                        labels.append(1)
                        break
                    if low_ret <= self.max_loss:
                        labels.append(0)
                        break
                else:
                    labels.append(0)
            elif max_return >= self.target_profit:
                labels.append(1)
            else:
                labels.append(0)

        return pd.Series(labels, index=df.index)


# ============================================================
# 模型训练与预测
# ============================================================
class SignalModel:
    """
    信号预测模型

    使用LightGBM，特点：
    - 训练快（秒级）
    - 预测快（<1ms）
    - 自动学习特征重要性
    - 自动处理特征交互
    """

    def __init__(self, params: Dict = None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
        }
        self.model = None
        self.feature_names = []
        self.threshold = 0.5

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict:
        """
        训练模型

        Returns:
            训练结果统计
        """
        if not HAS_LIGHTGBM:
            raise ImportError("需要安装lightgbm: pip install lightgbm")

        self.feature_names = list(X.columns)

        # 分割数据（时间序列方式）
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # 训练
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # 找最优阈值
        val_pred = self.model.predict(X_val)
        self.threshold = self._find_best_threshold(y_val, val_pred)

        # 计算验证集指标
        val_pred_binary = (val_pred >= self.threshold).astype(int)

        results = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'positive_rate': y.mean(),
            'best_threshold': self.threshold,
            'val_accuracy': (val_pred_binary == y_val).mean(),
            'val_precision': (val_pred_binary & y_val).sum() / max(val_pred_binary.sum(), 1),
            'val_recall': (val_pred_binary & y_val).sum() / max(y_val.sum(), 1),
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importance(importance_type='gain')
            ))
        }

        return results

    def _find_best_threshold(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """找收益最优的阈值"""
        best_threshold = 0.5
        best_score = -float('inf')

        for threshold in np.arange(0.3, 0.8, 0.05):
            pred = (y_pred >= threshold).astype(int)

            # 简单的收益评估：正确预测+3，错误预测-3
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()

            score = tp * 3 - fp * 3  # 模拟止盈止损

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测

        Returns:
            (概率, 信号)
        """
        if self.model is None:
            raise ValueError("模型未训练")

        proba = self.model.predict(X[self.feature_names])
        signal = (proba >= self.threshold).astype(int)

        return proba, signal

    def predict_single(self, features: Dict) -> Tuple[float, int]:
        """
        单条预测（实盘用）

        Args:
            features: 特征字典

        Returns:
            (概率, 信号)
        """
        X = pd.DataFrame([features])[self.feature_names]
        proba, signal = self.predict(X)
        return float(proba[0]), int(signal[0])

    def save(self, path: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        self.model.save_model(f"{path}.lgb")

        meta = {
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'params': self.params
        }
        with open(f"{path}.meta.json", 'w') as f:
            json.dump(meta, f)

    def load(self, path: str):
        """加载模型"""
        if not HAS_LIGHTGBM:
            raise ImportError("需要安装lightgbm: pip install lightgbm")

        self.model = lgb.Booster(model_file=f"{path}.lgb")

        with open(f"{path}.meta.json") as f:
            meta = json.load(f)

        self.feature_names = meta['feature_names']
        self.threshold = meta['threshold']
        self.params = meta['params']


# ============================================================
# 超参数优化
# ============================================================
class HyperparameterOptimizer:
    """
    超参数优化器

    使用Optuna自动搜索最优参数
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def optimize(
        self,
        n_trials: int = 50,
        timeout: int = 600
    ) -> Dict:
        """
        运行优化

        Args:
            n_trials: 尝试次数
            timeout: 超时时间（秒）

        Returns:
            最优参数
        """
        if not HAS_OPTUNA:
            raise ImportError("需要安装optuna: pip install optuna")

        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'verbose': -1,
            }

            # 交叉验证
            split_idx = int(len(self.X) * 0.8)
            X_train, X_val = self.X.iloc[:split_idx], self.X.iloc[split_idx:]
            y_train, y_val = self.y.iloc[:split_idx], self.y.iloc[split_idx:]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )

            # 评估：模拟交易收益
            val_pred = model.predict(X_val)

            best_score = -float('inf')
            for threshold in np.arange(0.3, 0.8, 0.1):
                pred = (val_pred >= threshold).astype(int)
                tp = ((pred == 1) & (y_val == 1)).sum()
                fp = ((pred == 1) & (y_val == 0)).sum()
                score = tp * 3 - fp * 3
                best_score = max(best_score, score)

            return best_score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }


# ============================================================
# 快速预测器（实盘用）
# ============================================================
class FastPredictor:
    """
    快速预测器

    特点：
    - 维护滚动窗口数据
    - 增量计算特征
    - 预测延迟 < 5ms
    """

    def __init__(self, model_path: str, windows: List[int] = [15, 30, 60]):
        self.windows = windows
        self.max_window = max(windows)
        self.feature_engineer = FeatureEngineer(windows)

        # 加载模型
        self.model = SignalModel()
        self.model.load(model_path)

        # 滚动数据缓存
        self.data_buffer: Dict[str, pd.DataFrame] = {}

    def update_and_predict(
        self,
        symbol: str,
        minute_bar: Dict
    ) -> Tuple[float, int, float]:
        """
        更新数据并预测

        Args:
            symbol: 交易对
            minute_bar: 分钟K线 {volume, buy_volume, sell_volume, close, high, low}

        Returns:
            (概率, 信号, 耗时ms)
        """
        start = time.perf_counter()

        # 初始化或更新缓存
        if symbol not in self.data_buffer:
            self.data_buffer[symbol] = pd.DataFrame()

        # 添加新数据
        new_row = pd.DataFrame([minute_bar])
        self.data_buffer[symbol] = pd.concat([
            self.data_buffer[symbol], new_row
        ]).tail(self.max_window * 2)

        # 数据不足
        if len(self.data_buffer[symbol]) < self.max_window + 1:
            return 0.0, 0, (time.perf_counter() - start) * 1000

        # 计算特征
        df_with_features = self.feature_engineer.compute_features(
            self.data_buffer[symbol]
        )

        # 获取最新特征
        latest_features = df_with_features[self.feature_engineer.feature_names].iloc[-1]

        # 检查是否有nan
        if latest_features.isna().any():
            return 0.0, 0, (time.perf_counter() - start) * 1000

        # 预测
        proba, signal = self.model.predict_single(latest_features.to_dict())

        elapsed = (time.perf_counter() - start) * 1000

        return proba, signal, elapsed


# ============================================================
# 便捷函数
# ============================================================
def train_model_from_data(
    data_files: List[str],
    target_profit: float = 3.0,
    stop_loss: float = -3.0,
    hold_minutes: int = 30,
    windows: List[int] = [15, 30, 60],
    optimize_params: bool = False
) -> Tuple[SignalModel, Dict]:
    """
    从数据文件训练模型

    Args:
        data_files: aggTrades CSV文件列表
        target_profit: 目标收益率 %
        stop_loss: 止损 %
        hold_minutes: 持仓时间
        windows: 特征窗口
        optimize_params: 是否优化超参数

    Returns:
        (模型, 训练结果)
    """
    print("=" * 60)
    print("模型训练流程")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    all_data = []
    for f in data_files:
        df = pd.read_csv(f)
        df['datetime'] = pd.to_datetime(df['transact_time'], unit='ms')
        df['minute'] = df['datetime'].dt.floor('1min')

        # 聚合为分钟数据
        agg = df.groupby('minute').agg({
            'price': ['min', 'max', 'first', 'last'],
            'quantity': 'sum',
            'is_buyer_maker': lambda x: (
                df.loc[x.index, 'quantity'] *
                (1 - df.loc[x.index, 'is_buyer_maker'].map({True: 1, False: 0, 'true': 1, 'false': 0}))
            ).sum()
        }).reset_index()

        agg.columns = ['minute', 'low', 'high', 'open', 'close', 'volume', 'buy_volume']
        agg['sell_volume'] = agg['volume'] - agg['buy_volume']
        all_data.append(agg)

    data = pd.concat(all_data, ignore_index=True)
    print(f"  加载 {len(data_files)} 个文件, {len(data)} 条分钟数据")

    # 2. 特征工程
    print("\n[2/5] 计算特征...")
    fe = FeatureEngineer(windows)
    data = fe.compute_features(data)
    print(f"  生成 {len(fe.feature_names)} 个特征")

    # 3. 生成标签
    print("\n[3/5] 生成标签...")
    lg = LabelGenerator(target_profit, stop_loss, hold_minutes)
    data['label'] = lg.generate_labels(data)

    # 移除nan
    valid_data = data.dropna(subset=['label'] + fe.feature_names)
    print(f"  有效样本: {len(valid_data)}, 正样本率: {valid_data['label'].mean():.2%}")

    X = valid_data[fe.feature_names]
    y = valid_data['label']

    # 4. 超参数优化（可选）
    if optimize_params and HAS_OPTUNA and HAS_LIGHTGBM:
        print("\n[4/5] 超参数优化...")
        optimizer = HyperparameterOptimizer(X, y)
        opt_result = optimizer.optimize(n_trials=30, timeout=300)
        print(f"  最优参数: {opt_result['best_params']}")
        model_params = {**SignalModel().params, **opt_result['best_params']}
    else:
        print("\n[4/5] 使用默认参数...")
        model_params = None

    # 5. 训练模型
    print("\n[5/5] 训练模型...")
    model = SignalModel(model_params)
    results = model.train(X, y)

    print(f"\n训练完成!")
    print(f"  验证集准确率: {results['val_accuracy']:.2%}")
    print(f"  验证集精确率: {results['val_precision']:.2%}")
    print(f"  验证集召回率: {results['val_recall']:.2%}")
    print(f"  最优阈值: {results['best_threshold']:.2f}")

    # 打印特征重要性
    print(f"\n特征重要性 Top 10:")
    importance = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for name, score in importance[:10]:
        print(f"  {name}: {score:.1f}")

    return model, results


if __name__ == '__main__':
    # 示例：使用现有数据训练
    data_dir = Path('/Users/fanwei/study/quants/bian_data/uai')
    data_files = list(data_dir.glob('*-aggTrades-*.csv'))

    if data_files:
        print(f"找到 {len(data_files)} 个数据文件")

        if HAS_LIGHTGBM:
            model, results = train_model_from_data(
                [str(f) for f in data_files[:2]],  # 使用前2个文件测试
                target_profit=3.0,
                stop_loss=-3.0,
                optimize_params=False  # 设为True开启自动优化
            )

            # 保存模型
            model.save(str(data_dir / 'backtest/signal_model'))
            print(f"\n模型已保存")
        else:
            print("请安装lightgbm: pip install lightgbm")
    else:
        print("未找到数据文件")
