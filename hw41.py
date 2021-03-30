# HW4.1
# Gradient calculator on matrices
import numpy as np
np.random.seed(19890817)

# 卷积运算


def conv(inputs: np.array, filter: np.array, padding=0):
    H, W = inputs.shape
    padded_input = np.zeros((H+2*padding, W+2*padding))
    padded_input[padding:H+padding, padding:W+padding] = inputs
    fH, fW = filter.shape
    result = np.zeros((H-fH+1+2*padding, W-fW+1+2*padding))
    for i in range(0, H-fH+1+2*padding):
        for j in range(0, W-fW+1+2*padding):
            input_now = padded_input[i:i + fH, j:j + fW]
            result[i, j] = np.sum(input_now * filter)
    return result


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
    def __init__(self, op: Op, inputs, shape=None, weight=None, has_grad=False):
        self.op = op
        if not isinstance(inputs, Node):
            self.inputs = np.array(inputs)
        else:
            self.inputs = inputs
        self.name = op.name()
        if self.name in ['Add', 'Sub']:
            self.inputs_shape = self.inputs[0].value.squeeze().shape
        else:
            self.inputs_shape = self.inputsToValue().squeeze().shape
        self.input_grad = np.zeros(self.inputs_shape)
        self.value = None
        self.shape = shape
        self.has_grad = has_grad
        if self.has_grad:
            self.grad = np.zeros(self.shape)
        self.weight = weight
        self.gradback = None
        # 预计算给后续节点提供输出大小信息
        self.cal()

    # 前向计算
    def cal(self):
        self.value = self.op.cal(self.inputsToValue(), self.weight)
        # 此时已知输入大小, 初始化后向积累梯度
        self.gradback = np.zeros(self.value.shape)

    # 梯度计算
    def calgrad(self):
        if not self.has_grad:
            self.input_grad = self.op.grad(self.inputsToValue(), self.gradback)
            return self.input_grad
        else:
            tmp, self.input_grad = self.op.grad(
                self.inputsToValue(), self.weight, self.gradback)
            self.grad = tmp.squeeze()
            return self.input_grad

    # 把Node对象转成其数值用于计算
    def inputsToValue(self):
        lst = []
        for i in self.inputs:
            if isinstance(i, Node):
                lst.append(i.value)
            else:
                lst.append(i)
        return np.array(lst)

    # 重载运算符
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

# 加法


class Add(Op):

    def name(self):
        return 'Add'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs, weight=None):
        assert(len(inputs) == 2)
        return inputs[0] + inputs[1]

    def grad(self, inputs, gradback):
        return np.array([gradback, gradback])

# 减法


class Sub(Op):

    def name(self):
        return 'Sub'

    def __call__(self, x, y):
        return Node(self, [x, y])

    def cal(self, inputs, weight=None):
        assert(len(inputs) == 2)
        return inputs[0] - inputs[1]

    def grad(self, inputs, gradback):
        return np.array([gradback, -gradback])

# 线性层


class Linear(Op):
    def name(self):
        return 'Linear'

    def __init__(self, shape, weight=None):
        self.shape = shape
        if not type(weight) == np.ndarray:
            self.weight = np.random.rand(*shape)
        else:
            self.weight = weight

    def __call__(self, inputs: np.array):
        return Node(self, [inputs], self.shape, self.weight, True)

    def cal(self, inputs, weight):
        return weight@inputs

    def grad(self, inputs, weight, gradback):
        return gradback@inputs.T, weight.T@gradback

# 卷积层


class Conv(Op):
    def name(self):
        return 'Conv'

    def __init__(self, shape, padding=0, weight=None):
        self.shape = shape
        self.padding = padding
        if not type(weight) == np.ndarray:
            self.weight = np.random.rand(*shape)
        else:
            self.weight = weight

    def __call__(self, inputs: np.array):
        return Node(self, [inputs], self.shape, self.weight, True)

    def cal(self, inputs, weight):
        return conv(inputs.squeeze(), weight, padding=self.padding)

    def grad(self, inputs, weight, gradback):
        inputs = inputs.squeeze()
        gradback = gradback.squeeze()
        input_shape = inputs.shape
        input_grad = np.zeros(input_shape)
        shape = self.shape
        weight_grad = np.zeros(shape)
        # 计算输入梯度
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                for a in range(gradback.shape[0]):
                    for b in range(gradback.shape[1]):
                        if i-a+self.padding < 0 or j-b+self.padding < 0 or i-a+self.padding >= shape[0] or j-b+self.padding >= shape[1]:
                            continue
                        input_grad[i, j] += gradback[a,
                                                     b]*weight[i-a+self.padding, j-b+self.padding]
        # 计算卷积核梯度
        for i in range(shape[0]):
            for j in range(shape[1]):
                for a in range(gradback.shape[0]):
                    for b in range(gradback.shape[1]):
                        if i+a-self.padding < 0 or j+b-self.padding < 0 or i+a-self.padding >= input_shape[0] or j+b-self.padding >= input_shape[1]:
                            continue
                        weight_grad[i, j] += gradback[a,
                                                      b]*inputs[i+a-self.padding, j+b-self.padding]
        return weight_grad, [input_grad]

# MSE损失函数


class MSE(Op):
    def name(self):
        return 'MSE'

    def __init__(self, shape, weight=None):
        self.shape = shape
        if not type(weight) == np.ndarray:
            self.weight = np.random.rand(*shape)
        else:
            self.weight = weight

    def __call__(self, inputs: np.array):
        return Node(self, [inputs], self.shape, self.weight, False)

    def cal(self, inputs: np.array, weight=None):
        return np.sum((inputs-self.weight) ** 2)

    def grad(self, inputs, gradback):
        return 2*(inputs-self.weight)*gradback

# 输入变量


class Holder(Op):
    def name(self):
        return 'Holder'

    def __call__(self, x):
        return Node(self, [x])

    def cal(self, inputs, weight=None):
        return inputs[0]

    def grad(self, inputs, gradback):
        return gradback


# 定义运算符
add, sub, holder = Add(), Sub(), Holder()


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
        if type(root) is np.ndarray or root == None or not isinstance(root, Node):
            return
        for n in root.inputs:
            self.topoSort(n)
        self.list.append(root)

    def grad(self):
        reversed_lst = list(reversed(self.topolist))
        reversed_lst[0].gradback = np.array([1.0])
        for node in reversed_lst:
            node.calgrad()
            for inputNode, backGrad in zip(node.inputs, node.input_grad):
                if isinstance(inputNode, Node):
                    if inputNode.name == 'Holder':
                        backGrad = backGrad.squeeze()
                    inputNode.gradback += backGrad
