import pandas as pd

def set_console():
    # 设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
    # pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)  # 加了这一行那表格的一行就不会分段出现了
    # pd.set_option('display.max_colwidth', 1000)
    # pd.set_option('display.height', 1000)
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # 显示所有行
    pd.set_option('display.max_rows', None)
    return pd