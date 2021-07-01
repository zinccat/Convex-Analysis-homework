# HW 4.2 Autodiff for multivariate formula

import math
from math import e, pi


class Op(object):
    def name(self):
        pass

    def __call__(self, inputs):
        pass

    def cal(self, inputs):
        pass

    def grad(self, inputs, gradback):
        pass


class Node(object):
    # 初始化
    def __init__(self, op: Op, inputs):
        self.op = op
        self.grad = 0
        self.inputs = inputs
        self.value = None
        self.name = op.name()

    # 计算
    def cal(self):
        self.value = self.op.cal(self.inputsToValue())

    def inputsToValue(self):
        lst = []
        for i in self.inputs:
            if isinstance(i, Node):
                lst.append(i.value)
            else:
                lst.append(i)
        return lst
    # 重载运算符

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

# 加法


class Add(Op):

    def name(self):
        return 'Add'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs):
        assert(len(inputs) == 2)
        return inputs[0] + inputs[1]

    def grad(self, inputs, gradback):
        return [gradback, gradback]

# 减法


class Sub(Op):

    def name(self):
        return 'Sub'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs):
        assert(len(inputs) == 2)
        return inputs[0] - inputs[1]

    def grad(self, inputs, gradback):
        return [gradback, -gradback]

# 乘法


class Mul(Op):
    def name(self):
        return 'Mul'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs):
        assert(len(inputs) == 2)
        return inputs[0] * inputs[1]

    def grad(self, inputs, gradback):
        return [gradback * inputs[1], gradback * inputs[0]]


# 除法
class Div(Op):
    def name(self):
        return 'Div'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs):
        assert(len(inputs) == 2)
        try:
            assert(inputs[1] != 0)
        except:
            print("Divided by 0!")
        return inputs[0] / inputs[1]

    def grad(self, inputs, gradback):
        return [gradback / inputs[1], -gradback * inputs[0]/inputs[1]**2]

# 对数


class Log(Op):
    def name(self):
        return 'Log'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs):
        assert(len(inputs) == 1)
        assert(inputs[0] > 0)
        return math.log(inputs[0])

    def grad(self, inputs, gradback):
        return [1.0/inputs[0]*gradback]


class Sin(Op):
    def name(self):
        return 'Sin'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs):
        assert(len(inputs) == 1)
        return math.sin(inputs[0])

    def grad(self, inputs, gradback):
        return [math.cos(inputs[0])*gradback]


class Cos(Op):

    def name(self):
        return 'Cos'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs):
        assert(len(inputs) == 1)
        return math.cos(inputs[0])

    def grad(self, inputs, gradback):
        return [-math.sin(inputs[0])*gradback]


class Tan(Op):
    def name(self):
        return 'Tan'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs):
        assert(len(inputs) == 1)
        return math.tan(inputs[0])

    def grad(self, inputs, gradback):
        return [gradback/((math.cos(inputs[0]))**2)]


class Exp(Op):

    def name(self):
        return 'Exp'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs):
        assert(len(inputs) == 1)
        return math.exp(inputs[0])

    def grad(self, inputs, gradback):
        return [gradback*math.exp(inputs[0])]


class Pow(Op):
    def name(self):
        return 'Pow'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs):
        assert(len(inputs) == 2)
        return inputs[0]**inputs[1]

    def grad(self, inputs, gradback):
        return [gradback * inputs[1]*(inputs[0]**(inputs[1]-1)), gradback * math.log(inputs[0])*(inputs[0]**inputs[1])]


class Holder(Op):
    def name(self):
        return 'Holder'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs):
        assert(len(inputs) == 1)
        return inputs[0]

    def grad(self, inputs, gradback):
        return [gradback]


class ExecutionGraph(object):

    def __init__(self, root=None) -> None:
        if root == None:
            return
        self.root = root
        self.list = []
        self.topoSort(root)
        # 去重
        self.topolist = []
        for node in self.list:
            if not node in self.topolist:
                self.topolist.append(node)

    def cal(self, Print=False) -> float:
        for index, node in enumerate(self.topolist):
            node.cal()
            if Print:
                print(index, node.name, node.value)
        return self.root.value

    # 拓扑排序
    def topoSort(self, root):
        if root == None or not isinstance(root, Node):
            return
        for n in root.inputs:
            self.topoSort(n)
        self.list.append(root)

    def grad(self):
        reversed_lst = list(reversed(self.topolist))
        reversed_lst[0].grad = 1.0
        for node in reversed_lst:
            grad = node.op.grad(node.inputsToValue(), node.grad)
            for inputNode, inputGrad in zip(node.inputs, grad):
                if isinstance(inputNode, Node):
                    inputNode.grad += inputGrad


# 定义运算符
add, sub, mul, div, log, sin, cos, tan, exp, pow, holder = Add(), Sub(
), Mul(), Div(), Log(), Sin(), Cos(), Tan(), Exp(), Pow(), Holder()

# 输入值
# feedDict = {'x': 2.0, 'y': 5.0}
# feedDict = {'x1': 2.0, 'x2': 5.0, 'x3': 7.0}
feedDict = {}

# 要计算的式子
# formula = '3/x'
# formula = 'log(x)+x**y-sin(y)'
# formula = '(sin(x1+1)+cos(2*x2))*tan(log(x3))+(sin(x2+1)+cos(2*x1))*exp(1+sin(x3))'
formula = ''

# 从文件读入要积分的变量和表达式
with open('hw42.in', 'r') as f:
    flag = False
    for line in f:
        line = line.strip()
        if not len(line):
            continue
        if line[0] == '#':
            flag = True
            continue
        if line[0] == '%':
            continue
        if not flag:
            tmp = line.strip().split()
            feedDict[tmp[0]] = float(tmp[1])
        else:
            formula = line.strip()
            break

for ele in feedDict:
    exec(ele + ' = holder(' + str(feedDict[ele]) + ')')

graph = ExecutionGraph()

try:
    exec('graph = ExecutionGraph(' + formula+')')
except:
    print('Illegal Formula!')
    exit(0)


# 打印计算图
# for n in z.topolist:
#    print(n.name)

print('------------------------------')
print(formula)
for ele in feedDict:
    print(ele, '=', feedDict[ele])

# 前向计算
print('------------------------------')
print("Forward:")
result = graph.cal(Print=False)
print('Result =', result)

# 计算梯度
graph.grad()
print('------------------------------')
print("Gradient:")
for ele in feedDict:
    print(ele + "'s grad =", eval(ele).grad)
print('------------------------------')

# 数值测试
print("Numeric test on variables")
graph2 = ExecutionGraph()
for ele in feedDict:
    exec(ele + '+= 10**(-7)')
    exec('graph2 = ExecutionGraph(' + formula+')')
    result2 = graph2.cal(Print=False)
    print(ele + "'s grad =", (result2-result)/10**(-7))
    exec(ele + '-= 10**(-7)')
print('------------------------------')
