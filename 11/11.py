# HW11 给定函数的梯度下降优化
# By ZincCat
import numpy as np
from matplotlib import pyplot as plt

# 设置随机种子
np.random.seed(19890817)

# 初始化问题
m = 10
n = 15

a = np.random.normal(5, 5, [n, m])
x = np.ones(n)


def f(x):
    # 计算函数值
    return np.sum(np.exp(a.T@x)) + np.sum(np.exp(-a.T@x))


def gradient_f(x):
    # 计算函数梯度
    return a@(np.exp(a.T@x) - np.exp(-a.T@x))


def descent(x, grad, grad_norm, mode='2'):
    # 梯度下降函数
    # 输入目前x取值, 梯度, 梯度的范数, 下降模式
    # 输出下降后x取值, 步长t
    # 下降模式为'2'时采用2范数, 为'inf'时采用无穷范数
    normalized_grad = grad/grad_norm
    t = 1.0
    if mode == '2':
        # l_2 norm
        while f(x - t*normalized_grad) > value - alpha*t*np.dot(grad, normalized_grad):
            t *= beta
        x -= t*normalized_grad
    elif mode == 'inf':
        # l_infty norm
        while f(x - t*np.sign(normalized_grad)) > value - alpha*t*np.dot(grad, np.sign(normalized_grad)):
            t *= beta
        x -= t*np.sign(normalized_grad)
    return x, t


minValue = f(np.zeros(n))  # 函数最小值

alpha_list = [0.22]
beta_list = [0.62]
maxIter = 1000  # 最大迭代次数
eta = 0.01  # 停止条件

result = []  # 记录 参数-结果 对
time = []  # 记录时间步, 用于绘图
values = []  # 记录某一时间步下函数值, 用于绘图
stepsize = []  # 记录某一时间步下步长, 用于绘图
Plot = True  # 是否绘图, 请保证此时alpha, beta均为单一取值

t = 0  # 用于绘图
# 实验
for alpha in alpha_list:
    for beta in beta_list:
        timestep = 0
        x = np.ones(n)
        while True:
            value = f(x)
            # print("Iteration:", timestep, "Error", value - minValue)
            if Plot:
                time.append(timestep)
                stepsize.append(t)
                values.append(value)
            grad = gradient_f(x)
            grad_norm = np.linalg.norm(grad)
            if grad_norm <= eta or timestep > maxIter:
                break
            x, t = descent(x, grad, grad_norm, mode='inf')  # 此时使用无穷范数
            timestep += 1
        result.append((alpha, beta, f(x)-minValue, timestep))
for i in result:
    print(i)

# 绘图
if Plot:
    # f − p^* versus iteration
    plt.plot(time, values)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.savefig('alpha'+str(alpha)+'beta'+str(beta)+'value.pdf')
    plt.show()

    # step length versus iteration number
    del(time[0])
    del(stepsize[0])
    plt.plot(time, stepsize)
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Step length', fontsize=14)
    plt.savefig('alpha'+str(alpha)+'beta'+str(beta)+'step.pdf')
    plt.show()
