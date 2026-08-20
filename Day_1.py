# 一次性安装
# pip install baostock pandas matplotlib

import baostock as bs
import pandas as pd

# 登录
lg = bs.login()
# 下载平安银行(000001) 2020-2024日线数据
rs = bs.query_history_k_data_plus("sz.000001",
    "date,open,high,low,close,volume",
    start_date='2020-01-01', end_date='2024-12-31',
    frequency="d", adjustflag="2")
data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())
data = pd.DataFrame(data_list, columns=rs.fields)
bs.logout()

# 看一眼数据长什么样
print(data.head())
print(data.shape)

# 1. 看一眼数据类型（检查哪些列是字符串，哪些是数字）
print("\n=== 数据类型 ===")
print(data.dtypes)

# 2. 把 date 列转为真正的日期类型（方便后续筛选）
data['date'] = pd.to_datetime(data['date'])

# 3. 按日期排序（重要！数据下载时可能不是按时间顺序）
data = data.sort_values('date')

# 4. 把 close 列转为浮点数（baostock 返回的是字符串）
data['close'] = data['close'].astype(float)
data['open'] = data['open'].astype(float)
data['high'] = data['high'].astype(float)
data['low'] = data['low'].astype(float)
data['volume'] = data['volume'].astype(float)

# 5. 验证一下转换是否成功
print("\n=== 转换后的数据 ===")
print(data[['date', 'close', 'volume']].head())

# 6. 保存成 CSV 文件（以后可以直接读，不用每次重新下载）
data.to_csv('pingan_bank_2020_2024.csv', index=False)
print("\n✅ 数据已保存到 pingan_bank_2020_2024.csv")