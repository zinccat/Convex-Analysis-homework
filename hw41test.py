# HW4.1 Test
# 此处用于定义求导所用参数

from hw41 import *

# 先定义输入, 用holder构造
x = holder(np.random.rand(5, 5))

# 再定义网络元素
k = Conv((3, 3), padding=1)  # 卷积, 需提供大小, 可以提供默认权重, 用weight制定
l = Linear((5, 5))  # 线性, 需提供大小, 不指定weight时根据大小调用numpy.random.rand()生成随机矩阵
y = np.random.rand(5, 5)  # 正确输出值
loss = MSE((5, 5), y)  # 损失函数, 程序中提供了MSE

# 构建网络, 此处定义了题目要求的resnet
f = ExecutionGraph(loss(x+k(x)))

# 进行计算
print('------------------------------------------------------------------------')
print("Forward:")
res1 = f.cal()  # 输出值
print("Result =", res1)

print('------------------------------------------------------------------------')
print("Gradient:")
f.grad()  # 计算梯度

print("x's grad", x.input_grad)  # 每层网络输入的权重用input_grad表示
print(f.topolist[1].name + "'s grad:", f.topolist[1].grad)  # 每层网络自身的权重用grad表示

# 数值测试
print('------------------------------------------------------------------------')
print("Numeric test on variables")
x_grad = x.input_grad
conv_grad = f.topolist[1].grad

x.inputs[0][4, 2] += 10**(-8)
res2 = f.cal()
print("x[4,2]'s real grad:", x_grad[4, 2])
print("x[4,2]'s numeric grad:", (res2-res1)*(10**8))
x.inputs[0][4, 2] -= 10**(-8)

f.topolist[1].weight[0, 1] += 10**(-8)
res2 = f.cal()
f.grad()
print(f.topolist[1].name + "[0,1]'s real grad:", conv_grad[0, 1])
print(f.topolist[1].name + "[0,1]'s numeric grad:", (res2-res1)*(10**8))
f.topolist[1].weight[0, 1] -= 10**(-8)
