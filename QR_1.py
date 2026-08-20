import yfinance as yf
import pandas as pd
import numpy as np

#df.shape          # 查看表格有几行几列 (比如 (252, 5) 表示252天、5列)
#df.columns        # 查看所有列名
#df['Close']       # 单独取出"收盘价"这一列
#df[['Close', 'Volume']]  # 同时取出多列
#df.iloc[0]        # 查看第一行数据
#df.describe()     # 看每一列的统计信息（均值、最大、最小等）
# 下载数据
df = yf.download('AAPL', start='2024-01-01', end='2024-12-31')
print(df.columns)
df = df[['Close', 'High', 'Low', 'Open', 'Volume']] #第一个中括号是“取数据”，第二个中括号是“列表”。
print(df.head())
df['avg_volume_5'] = df['Volume'].rolling(5).mean() # 计算过去5天平均成交量
df['Surge'] = (df['Volume'].squeeze() > df['avg_volume_5'].squeeze() * 2).astype(int) # 判断是否突增（今天的量 > 过去5天均量的2倍）
df['future_return'] = df['Close'].shift(-1) / df['Close'] - 1 # 计算次日涨跌幅（预测目标）
df_clean = df.dropna().copy()
grouped = df_clean.groupby('Surge')['future_return'].agg(['mean', 'count', 'std'])
print(grouped)
df_clean['is_up']= (df_clean['future_return']> 0).astype(int)
# 胜率
win_rate = df_clean.groupby('Surge')['is_up'].mean()
print("\n=== 胜率 ===")
print(win_rate)