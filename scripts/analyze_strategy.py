import pandas as pd
import numpy as np
import os
import datetime

# File paths
RECORD_DIR = 'record'
POSITION_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26持仓记录.csv')
TRADE_FILE = os.path.join(RECORD_DIR, '2017-08-10 至 2026-02-26调仓记录.csv')

def calculate_metrics():
    # Load Position Data
    df_pos = pd.read_csv(POSITION_FILE)
    df_pos['日期'] = pd.to_datetime(df_pos['日期'])
    df_pos = df_pos.sort_values('日期')
    
    # Calculate Daily Returns
    df_pos['daily_ret'] = df_pos['净值'].pct_change().fillna(0)
    
    # Total Return
    total_return = df_pos['净值'].iloc[-1] / df_pos['净值'].iloc[0] - 1
    
    # Annualized Return (CAGR)
    days = (df_pos['日期'].iloc[-1] - df_pos['日期'].iloc[0]).days
    cagr = (1 + total_return) ** (365 / days) - 1
    
    # Volatility (Annualized)
    volatility = df_pos['daily_ret'].std() * np.sqrt(252)
    
    # Sharpe Ratio (Assuming Risk-Free Rate = 0 for simplicity or 2% as typical)
    rf = 0.02
    sharpe = (cagr - rf) / volatility if volatility != 0 else 0
    
    # Max Drawdown
    cumulative_returns = df_pos['净值']
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # --- Trading Analysis ---
    df_trade = pd.read_csv(TRADE_FILE)
    df_trade['调仓时间'] = pd.to_datetime(df_trade['调仓时间'].str.replace(' 开盘', '')) # "2026-02-06 开盘" -> "2026-02-06"
    df_trade = df_trade.sort_values('调仓时间')
    
    # Calculate Holding Periods
    # For each trade, calculate the time until the next trade
    df_trade['next_trade_date'] = df_trade['调仓时间'].shift(-1)
    df_trade['holding_days'] = (df_trade['next_trade_date'] - df_trade['调仓时间']).dt.days
    
    # Fill last trade holding days with days until end of record
    df_trade.loc[df_trade.index[-1], 'holding_days'] = (df_pos['日期'].iloc[-1] - df_trade['调仓时间'].iloc[-1]).days
    
    # Win Rate: Was the return positive during the holding period?
    # We can check the net value change between trades
    df_trade['net_val_change'] = df_trade['调仓后净值'].shift(-1) - df_trade['调仓后净值']
    # For the last trade, check current net value
    df_trade.loc[df_trade.index[-1], 'net_val_change'] = df_pos['净值'].iloc[-1] - df_trade['调仓后净值'].iloc[-1]
    
    win_trades = (df_trade['net_val_change'] > 0).sum()
    total_trades = len(df_trade)
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    # Asset Preference
    # Sum holding days by asset
    # Note: '买入' column indicates what we bought and held
    asset_holding_days = df_trade.groupby('买入')['holding_days'].sum().sort_values(ascending=False)
    
    # Frequency
    avg_holding_period = df_trade['holding_days'].mean()
    
    # Print Report
    print("--- 策略量化分析报告 ---")
    print(f"回测区间: {df_pos['日期'].iloc[0].date()} 至 {df_pos['日期'].iloc[-1].date()} ({days} 天)")
    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率 (CAGR): {cagr:.2%}")
    print(f"年化波动率: {volatility:.2%}")
    print(f"夏普比率 (Sharpe, Rf=2%): {sharpe:.2f}")
    print(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")
    print("-" * 30)
    print(f"总交易次数: {total_trades}")
    print(f"平均持仓周期: {avg_holding_period:.1f} 天")
    print(f"胜率 (Win Rate): {win_rate:.2%}")
    print("-" * 30)
    print("资产持有偏好 (天数):")
    print(asset_holding_days)
    print("-" * 30)
    
    # Calculate monthly returns for consistency/seasonality check? Maybe overkill.
    # Just basic stats are enough for "Expert Analysis".

if __name__ == "__main__":
    calculate_metrics()
