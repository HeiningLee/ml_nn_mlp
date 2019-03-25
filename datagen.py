"""这是一个专门用于生成训练数据和测试数据的辅助函数。
这里用的目标函数是 y = 2*x1**2 + x2**2

"""

import random


def target_func(x1, x2):
    output = 2 * x1**2 + x2**2
    return output


# 训练数据存放路径
file_path = 'data_testing.csv'
lines = 500

with open(file_path, 'w') as fp:
    for i in range(lines):
        a = 1000 * random.random() - 500  # 这里的变换是因为random使用的是(0,1)均匀分布
        b = 1000 * random.random() - 500
        y = target_func(a, b)
        fp.write(str(y))
        fp.write(',')
        fp.write(str(a))
        fp.write(',')
        fp.write(str(b))
        fp.write('\n')
        pass
    pass


