import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取你保存的数据
data = pd.read_csv('pingan_bank_2020_2024.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')
data['close'] = data['close'].astype(float)

# 2. 计算日收益率
data['ret'] = data['close'].pct_change()      # 普通收益率
data['log_ret'] = np.log(data['close'] / data['close'].shift(1))  # 对数收益率

# 3. 计算移动平均线
data['ma5'] = data['close'].rolling(5).mean()
data['ma20'] = data['close'].rolling(20).mean()
data['ma60'] = data['close'].rolling(60).mean()

# 4. 计算波动率（20日，年化）
data['vol20'] = data['ret'].rolling(20).std() * np.sqrt(252)

# 5. 了解 shift / diff / pct_change 的区别
data['close_shift1'] = data['close'].shift(1)
data['close_diff'] = data['close'].diff()
data['close_pct'] = data['close'].pct_change()

# 6. 打印前10行看看结果
print(data[['date', 'close', 'ma5', 'ma20', 'vol20']].head(10))

# 7. 画个图
plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['close'], label='Close')
plt.plot(data['date'], data['ma5'], label='MA5')
plt.plot(data['date'], data['ma20'], label='MA20')
plt.legend()
plt.title('平安银行 收盘价与均线')
plt.show()