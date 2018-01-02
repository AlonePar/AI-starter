"""
numpy数组
"""
import numpy as np
import matplotlib.pyplot as plot

# 定义输入数据数组x,输出数据数组y
x, y = [], []

for sample in open('./180102-prices.txt', 'r'):
    xx, yy = sample.split(',')
    x.append(float(xx))
    y.append(float(yy))

x, y = np.array(x), np.array(y)
# 标准化
x = (x - x.mean()) / x.std()

# 原始数据散点图
plot.figure()
plot.scatter(x, y, c='g', s=6)
plot.show()

# 在-2至4之间取100个点
x0 = np.linspace(-2, 4, 100)

def get_model(deg):
    """
    利用Numpy的函数定义训练并返回多项式回归模型的函数
    deg 模型中多项式的系数
    返回的模型能够根据输入的x()
    """
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)

def get_cost(deg, input_x, input_y):
    """
    根据参数n、输入数据x, y返回相应的损失
    """
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()

test_n = (1, 4, 10)
for d in test_n:
    print(get_cost(d, x, y))

plot.scatter(x, y, c='g', s=20)
for d in test_n:
    plot.plot(x0, get_model(d)(), label='degree = {}'.format(d))

# 将x轴，y轴的范围分别限制在(-2, 4)、(10^5, 8*10^5)
plot.xlim(-2, 4)
plot.ylim(1e5, 8e5)
# 显示label
plot.legend()
plot.show()
