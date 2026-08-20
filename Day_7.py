import pandas as pd
import numpy as np

# 读取你已有的数据
df_all = pd.read_csv('df_all_with_indicators.csv')
df_all['date'] = pd.to_datetime(df_all['date'])

# 按股票和日期排序
df_all = df_all.sort_values(['code', 'date']).reset_index(drop=True)

# 1. 计算过去20天的累计收益率（反转因子）
df_all['ret_20d'] = df_all.groupby('code')['ret'].rolling(20).sum().reset_index(level=0, drop=True)

# 2. 确定调仓日期（每月最后一个交易日）
df_all['year_month'] = df_all['date'].dt.to_period('M')
rebalance_dates = df_all.groupby('year_month')['date'].max().sort_values().tolist()

# 3. 在每个调仓日，选出 ret_20d 最小的2只股票（跌最多的）
signals = []
for dt in rebalance_dates:
    day_data = df_all[df_all['date'] == dt].copy()
    day_data = day_data.dropna(subset=['ret_20d'])
    if len(day_data) < 2:
        continue
    selected = day_data.nsmallest(2, 'ret_20d')['code'].tolist()
    for code in selected:
        signals.append({'date': dt, 'code': code, 'signal': 1})

signals_df = pd.DataFrame(signals)

# 4. 合并回主数据
df_all = df_all.merge(signals_df, on=['date', 'code'], how='left')
df_all['signal'] = df_all['signal'].fillna(0).astype(int)

# 5. 检查结果
print("信号数量：", len(signals_df))
print("\n每只股票被选中的次数：")
print(df_all[df_all['signal'] == 1]['code'].value_counts())

# 6. 保存
df_all.to_csv('df_all_with_signals.csv', index=False)
print("\n✅ 已保存：df_all_with_signals.csv")