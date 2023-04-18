# 固定训练集和验证集和测试集
# 全局搜索
import random
import numpy as np
import pandas as pd #读取csv文件的库
from matplotlib import pyplot as plt
import torch
from model.model import Network
from itertools import product
from sklearn.model_selection import train_test_split
# from model2 import Network

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
# setup_seed(30)


# 预处理数据以及训练模型
# ...
# ...


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

data_path = 'Data.csv'#数据位置
orign_data = pd.read_csv(data_path)#读取数据文件
fields_to_drop = ['0', '1', '2', '3','4'] #要去掉的列名字
data = orign_data.drop(fields_to_drop, axis=1) #从文件中去掉

fields_to_drop2 = ['5','6','7','8']
data2 = data.drop(fields_to_drop2, axis=1)

quant_features = list(data2.columns.values)#要归一化的列

# 我们将每一个变量的均值和方差都存储到scaled_features变量中。
scaled_features = {}#空集合
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()#求均值和方差
    scaled_features[each] = [mean, std]#将均值和方差都装进集合
    data.loc[:, each] = (data[each] - mean)/std#求归一化结果并装载进data
# print(data)

train_data = data[:635]
# val_data = data[535:635]
test_data = data[635:]

# 训练数据和测试数据划分
target_fields = ['5', '6', '7','8'] #目标列索引
train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
# val_features, val_targets = val_data.drop(target_fields, axis=1), val_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# 将数据从pandas dataframe转换为numpy
# 将数据转换为tensor
X = train_features.values
Y = train_targets['7'].values
Y = np.reshape(Y, [len(Y), 1])
X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
"""
val_x = val_features.values
val_y = val_targets['7'].values
val_y = np.reshape(val_y, [len(val_y), 1])
val_y = val_y.astype(float)
val_x = torch.tensor(val_x, dtype=torch.float32,
                 requires_grad=True)
val_y = torch.tensor(val_y, dtype=torch.float32, requires_grad=True)
val_x = val_x.reshape(-1,1,75)
val_x = val_x.cuda()
val_y = val_y.cuda()"""


parameters = dict(
    lr = [0.001, 0.0001, 0.0005]
    ,batch_size = [32, 64]
    ,wd = [0.0001, 0.0005, 0.0009]
)
param_values = [v for v in parameters.values()]

for lr, batch_size, wd in product(*param_values):

    losses = []
    setup_seed(30)  # 随机种子放在循环中，每次循环重置。
    # maml_state_dict = torch.load('maml_model_save/maml_model.pth')
    model = Network()
    # model.load_state_dict(maml_state_dict)
    model = model.cuda()

    batch_size = batch_size
    cost = torch.nn.MSELoss()
    cost = cost.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
    losses = []
    val_losses = []
    min_loss = 100
    min_epoch = 0

    for i in range(800):
        batch_loss = []
        model.train()
        # start和end分别是提取一个batch数据的起始和终止下标
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size if start + batch_size < X.shape[0] else len(X)
            xx = torch.tensor(X[start:end], dtype=torch.float, requires_grad=True)
            yy = torch.tensor(Y[start:end], dtype = torch.float, requires_grad = True)
            xx = xx.reshape(-1,1,75)
            xx = xx.cuda()
            yy = yy.cuda()
            predict = model(xx)
            loss = cost(predict, yy)
            loss = torch.sqrt(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.data.cpu().numpy())

        losses.append(np.mean(batch_loss))
        train_loss = losses[i]
        """#验证
        with torch.no_grad():
            model.eval()
            val_predict = model(val_x)
            val_mse = cost(val_predict,val_y)
            val_rmse = torch.sqrt(val_mse)
        val_losses.append(val_rmse.data.cpu().numpy())
        val_loss = val_losses[i]
        # print(f'epoch:{i},train_RMSE:{losses[i]},val_RMSE:{val_losses[i]}')

        if val_loss < min_loss:
            min_loss = val_loss
            min_epoch = i
            torch.save(model.state_dict(), 'model_weight/min_val_loss_model.pth')
            # torch.save(model, 'model/model.pkl')

    # 将损失函数作图
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.plot(losses, label='loss', linestyle='-')
    ax.plot(val_losses, label='val_loss', linestyle='--')
    ax.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('')
    plt.show()"""

    # 测试数据集
    y = test_targets['7'].values
    y = y.reshape([len(y), 1])  # 将数据转换成合适的tensor形式
    y = y.astype(float)  # 保证数据为实数

    x = torch.tensor(test_features.values, dtype=torch.float32,
                     requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True)

    x = x.reshape(-1, 1, 75)
    x = x.cuda()
    y = y.cuda()

    # 用神经网络进行预测

    # 加载保存的模型
    # m_state_dict = torch.load('model_weight/min_val_loss_model.pth')
    # model2 = Network()
    # model2 = torch.load('model/model.pkl')
    # model2.load_state_dict(m_state_dict)
    model.eval()
    model = model.cuda()

    predict2 = model(x)
    test_mse = cost(predict2,y)
    test_rmse = torch.sqrt(test_mse)
    predict2 = predict2.data.cpu().numpy()
    y = y.data.cpu().numpy()
    # print(test_rmse)
    print(f'learn_rate:{lr},batch_size:{batch_size},weight_dency:{wd},test_rmse:{test_rmse}')





"""fig, ax = plt.subplots(figsize=(150, 7))
ax.plot(predict2, label='Prediction', linestyle='--')
ax.plot(y, label='Data', linestyle='-')
ax.legend()
ax.set_xlabel(' ')
ax.set_ylabel('Counts')
plt.show()"""
