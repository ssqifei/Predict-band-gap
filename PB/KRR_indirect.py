# 核岭回归KRR
from __future__ import print_function
import random
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared
from scipy.stats import pearsonr

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# 设置随机数种子
setup_seed(30)

# 读取数据

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
test_data = data[635:]

#划分训练集
target_fields = ['5', '6', '7','8'] #目标列索引
train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# 将数据从pandas dataframe转换为numpy
# 将数据转换为tensor
X = train_features.values
Y = train_targets['7'].values
Y = np.reshape(Y, [len(Y), 1])

x = test_features.values
y = test_targets['7'].values
y2 = test_targets['7'].values
y = np.reshape(y, [len(y), 1])

 ##  Define Random Forest Hyperparameter Space  ##
param_grid = {"alpha": [1e1, 1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}

krr_opt = GridSearchCV(KernelRidge(), param_grid=param_grid, cv=5)
krr_opt.fit(X, Y)
Pred_test = krr_opt.predict(x)
Pred_test2 = Pred_test.reshape(100,)

mse_test_prop = sklearn.metrics.mean_squared_error(Pred_test,y)
r = pearsonr(Pred_test2, y2)
print('KRR_rmse_test=', np.sqrt(mse_test_prop))
print('KRR_r_test=', r)


#画图
fig=plt.figure(figsize=(6,6),dpi=450)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.96, top=0.97)

a = [-175,0,125]
b = [-175,0,125]

plt.rc('font', family='Arial narrow')

plt.plot(b, a, c='k', ls='-')

plt.ylabel('KRR Prediction', fontname='Arial Narrow', size=20)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=20)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)


## Direct_gap
plt.ylim([-0.5,8.5])
plt.xlim([-0.5,8.5])
plt.xticks([0.0, 2.0, 4.0, 6.0, 8.0])
plt.yticks([0.0, 2.0, 4.0, 6.0, 8.0])

plt.title('Indirect band gap', fontsize = 20, fontweight='heavy')
plt.text(6, 0, 'RMSE = 0.704\nr = 0.863')
plt.scatter(y[:], Pred_test[:], c='b', marker='o')

# plt.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':24})
plt.tight_layout()
plt.savefig('image/KRR_indirect.png', dpi=450)
plt.show()
