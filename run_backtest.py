#!/usr/bin/env python3
"""
大规模回测主程序

使用方法:
    python run_backtest.py [--download] [--backtest] [--analyze] [--evolve] [--all]

参数:
    --download  下载数据
    --backtest  运行回测
    --analyze   分析结果
    --evolve    运行自我进化回测（ML模型 + 自动优化）
    --all       执行全部步骤（默认，不含evolve）

示例:
    python run_backtest.py --download 20260319  # 下载指定日期
    python run_backtest.py --evolve             # 运行自我进化回测
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_download():
    """运行数据下载"""
    print("\n" + "=" * 70)
    print("步骤 1/3: 下载数据")
    print("=" * 70)
    from data_downloader import main as download_main
    download_main()


def run_backtest():
    """运行回测"""
    print("\n" + "=" * 70)
    print("步骤 2/3: 执行回测")
    print("=" * 70)
    from backtest_engine import main as backtest_main
    backtest_main()


def run_analyze():
    """运行分析"""
    print("\n" + "=" * 70)
    print("步骤 3/3: 生成分析报告")
    print("=" * 70)
    from analysis_report import main as analyze_main
    analyze_main()


def run_evolve(data_files=None, max_generations=5):
    """
    运行自我进化回测

    特点：
    1. 使用ML模型（LightGBM）学习最优信号
    2. 自动分析回测结果，识别问题
    3. 自动优化策略参数
    4. 迭代进化，持续改进
    """
    print("\n" + "=" * 70)
    print("自我进化回测")
    print("=" * 70)

    try:
        from self_evolving import SelfEvolvingSystem
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保 self_evolving.py 存在")
        return

    # 查找数据文件
    if data_files is None:
        data_dir = Path(__file__).parent.parent
        data_files = list(data_dir.glob('*-aggTrades-*.csv'))
        # 也检查backtest/data目录
        data_files.extend(list((data_dir / 'backtest/data').glob('**/*-aggTrades-*.csv')))

    if not data_files:
        print("未找到数据文件")
        print("请先运行: python data_downloader_v2.py <日期>")
        return

    print(f"找到 {len(data_files)} 个数据文件")

    # 创建并运行进化系统
    system = SelfEvolvingSystem({
        'windows': [15, 30, 60],
        'target_profit': 3.0,
        'stop_loss': -3.0,
        'hold_minutes': 30,
        'min_precision': 0.6,
        'max_generations': max_generations,
    })

    result = system.evolve([str(f) for f in data_files])

    # 保存模型
    save_dir = Path(__file__).parent / 'evolved_model'
    system.save(str(save_dir))

    return result


def run_walk_forward(data_dir=None, start_date=None, end_date=None, train_days=10, test_days=10):
    """
    运行滚动前向验证（静态合约版本）

    特点：
    1. 无穿越 - 永远用历史数据训练，预测未来
    2. 累积学习 - 训练数据持续增长，知识不断积累
    3. 自动进化 - 根据验证结果自动优化参数
    """
    print("\n" + "=" * 70)
    print("滚动前向验证（静态版）")
    print("=" * 70)

    try:
        from walk_forward import WalkForwardSystem
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保 walk_forward.py 存在")
        return

    if data_dir is None:
        data_dir = str(Path(__file__).parent.parent)

    system = WalkForwardSystem({
        'windows': [15, 30, 60],
        'target_profit': 3.0,
        'stop_loss': -3.0,
        'hold_minutes': 30,
        'min_precision': 0.6,
        'train_days': train_days,
        'test_days': test_days,
        'evolution_rounds': 3,
    })

    result = system.run_walk_forward(data_dir, start_date, end_date)

    # 保存
    save_dir = Path(__file__).parent / 'walk_forward_model'
    system.save(str(save_dir))

    return result


def run_dynamic_walk_forward(start_date=None, end_date=None, train_days=10, test_days=10):
    """
    运行动态滚动前向验证

    特点：
    1. 每天动态筛选符合条件的合约（100万-1000万交易量）
    2. 自动下载数据
    3. 训练期和验证期合约可以不同
    4. 累积学习 + 自我进化
    """
    print("\n" + "=" * 70)
    print("动态滚动前向验证")
    print("=" * 70)

    try:
        from dynamic_walk_forward import DynamicWalkForwardSystem
    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保 dynamic_walk_forward.py 存在")
        return

    data_dir = str(Path(__file__).parent / 'data')

    system = DynamicWalkForwardSystem(
        config={
            'min_volume_usdt': 1_000_000,
            'max_volume_usdt': 10_000_000,
            'train_days': train_days,
            'test_days': test_days,
        },
        data_dir=data_dir
    )

    result = system.run(start_date, end_date)

    # 保存
    save_dir = Path(__file__).parent / 'dynamic_model'
    system.save(str(save_dir))

    return result


def main():
    parser = argparse.ArgumentParser(description='大规模回测系统')
    parser.add_argument('--download', action='store_true', help='下载数据')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--analyze', action='store_true', help='分析结果')
    parser.add_argument('--evolve', action='store_true', help='运行自我进化回测')
    parser.add_argument('--walk-forward', action='store_true', help='运行滚动前向验证（静态版）')
    parser.add_argument('--dynamic', action='store_true', help='运行动态滚动前向验证（推荐）')
    parser.add_argument('--generations', type=int, default=5, help='进化代数（默认5）')
    parser.add_argument('--start', type=str, help='起始日期 YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='结束日期 YYYY-MM-DD')
    parser.add_argument('--train-days', type=int, default=10, help='训练周期天数（默认10）')
    parser.add_argument('--test-days', type=int, default=10, help='验证周期天数（默认10）')
    parser.add_argument('--all', action='store_true', help='执行全部步骤')

    args = parser.parse_args()

    # 如果没有指定任何参数，默认执行全部
    if not any([args.download, args.backtest, args.analyze, args.evolve,
                args.walk_forward, args.dynamic, args.all]):
        args.all = True

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          大规模回测系统                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  本系统支持以下模式:                                                          ║
║                                                                              ║
║  传统回测流程 (--all):                                                        ║
║  1. 数据下载 - 从币安下载历史交易数据                                          ║
║  2. 执行回测 - 运行自适应策略                                                  ║
║  3. 生成报告 - 分析回测结果                                                    ║
║                                                                              ║
║  自我进化回测 (--evolve):                                                      ║
║  - 使用LightGBM机器学习模型                                                   ║
║  - 自动分析问题（差时段/差币种）                                               ║
║  - 自动优化（黑名单/参数调整）                                                 ║
║                                                                              ║
║  动态滚动前向验证 (--dynamic) [推荐]:                                          ║
║  - 每天动态筛选100万-1000万交易量的合约                                        ║
║  - 自动下载数据                                                               ║
║  - 训练期和验证期合约可以不同                                                  ║
║  - 无穿越 + 累积学习 + 自我进化                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    if args.all or args.download:
        run_download()

    if args.all or args.backtest:
        run_backtest()

    if args.all or args.analyze:
        run_analyze()

    if args.evolve:
        run_evolve(max_generations=args.generations)

    if args.walk_forward:
        if not args.start or not args.end:
            print("错误: --walk-forward 需要指定 --start 和 --end 日期")
            print("示例: python run_backtest.py --walk-forward --start 2026-01-01 --end 2026-01-30")
            return
        run_walk_forward(
            start_date=args.start,
            end_date=args.end,
            train_days=args.train_days,
            test_days=args.test_days
        )

    if args.dynamic:
        if not args.start or not args.end:
            print("错误: --dynamic 需要指定 --start 和 --end 日期")
            print("示例: python run_backtest.py --dynamic --start 2026-01-01 --end 2026-01-30")
            return
        run_dynamic_walk_forward(
            start_date=args.start,
            end_date=args.end,
            train_days=args.train_days,
            test_days=args.test_days
        )

    print("\n" + "=" * 70)
    print("回测完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
