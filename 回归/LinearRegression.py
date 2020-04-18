from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def linear_regression():
    """
    线性回归直接预测房子价格-正规方程
    :return: None
    """

    # 获取数据
    lb = load_boston()  # sklearn自带的数据集

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)

    # 进行标准化处理  ! 对于线性回归目标值也要分开标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 将目标值数据改为二维
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测（正规方程求解方式预测结果）
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)

    # 预测测试集的房子价格
    y_predict = std_y.inverse_transform(lr.predict(x_test))
    print("正规方程测试集里面每个房子的预测价格：", y_predict)

    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))


def sgd_regressor():
    """
    线性回归直接预测房子价格-梯度下降
    :return:
    """

    # 获取数据
    lb = load_boston()  # sklearn自带的数据集

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(y_train, y_test)

    # 进行标准化处理  ! 对于线性回归目标值也要分开标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 将目标值数据改为二维
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimator预测（梯度下降求解方式预测结果）
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)

    # 预测测试集的房子价格
    y_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("梯度下降测试集里面每个房子的预测价格：", y_predict)

    print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))


def ridge():
    """
    岭回归方式进行预测房价
    :return: None
    """

    # 获取数据
    lb = load_boston()  # sklearn自带的数据集

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(y_train, y_test)

    # 进行标准化处理  ! 对于线性回归目标值也要分开标准化处理
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 将目标值数据改为二维
    y_test = std_y.transform(y_test.reshape(-1, 1))
    
     # 使用保存的模型预测结果
    # model = joblib.load("./model/test.pkl")
    # y_predict = std_y.inverse_transform(model.predict(x_test))
    # print("保存的模型预测的结果：", y_predict)

    # 岭回归方式预测房价
    rd = Ridge(alpha=1.0)  # alpha一般（0~1）（1~10）
    rd.fit(x_train, y_train)
    print(rd.coef_)
    
     # 保存训练好的模型
    # joblib.dump(rd, "./model/test.pkl")

    # 预测测试集的房子价格
    y_predict = std_y.inverse_transform(rd.predict(x_test))
    print("梯度下降测试集里面每个房子的预测价格：", y_predict)

    print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))


if __name__ == '__main__':
    # linear_regression()
    # sgd_regressor()
    ridge()
