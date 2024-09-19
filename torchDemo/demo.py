import pandas as pd

data = pd.read_csv('urls.csv',usecols=['urls'])

# 对data 去重
data = data.drop_duplicates(subset=['urls'], keep='first', inplace=False)
print(data)

# 保存去重后 删除不是以 https://webofscience.clarivate.cn/wos/alldb/full-record/WOS 开头的url
data = data[data['urls'].str.contains('https://webofscience.clarivate.cn/wos/alldb/full-record/WOS')]
data.to_csv('reslove_urls.csv',index=False)