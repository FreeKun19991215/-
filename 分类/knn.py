from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd


def knncls():
    """
    K-近邻预测用户入住位置
    :return: None
    """

    # 获取数据
    data = pd.read_csv("./data/facebook-v-predicting-check-ins/train.csv")
    # print(data.head(10))

    # 处理数据
        # 1、缩小数据（节省时间），查询数据筛选
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

        # 2、处理时间，将时间戳转化成可视化时间
    time_value = pd.to_datetime(data["time"], unit="s")
    time_value = pd.DatetimeIndex(time_value)  # 把日期格式转换成字典格式

        # 3、构造一些特征
    data["day"] = time_value.day
    data["hour"] = time_value.hour
    # data["weekday"] = time_value.weekday  # 将此特征无视，准确率从47%提升到48%

        # 4、把时间戳特征删除
    data = data.drop(["time"], axis=1)  # 列
    # print(data)

        # 5、把签到数量少于n个的目标位置删除
    place_count = data.groupby("place_id").count()  # 安照“place_id”分组
    tf = place_count[place_count.row_id > 3].reset_index()  # 保留“row_id”大于3的数据
    data = data[data["place_id"].isin(tf.place_id)]  # 保留签到次数大于3的数据

    data = data.drop(["row_id"], axis=1)  # 删除无关特征

        # 6、去除数据当中的特征值和目标值
    y = data["place_id"]
    x = data.drop(["place_id"], axis=1)

        # 7、进行数据的分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()
    x_train = std.fit_transform(x_train)  # 对测试集和训练集的特征值进行标准化
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=8)  # 50%左右最优
    knn.fit(x_train, y_train)

    # 得到预测结果
    y_predict = knn.predict(x_test)
    print("预测的目标签到位置为：", y_predict)

    # 得出准确率
    print("预测的准确率：", knn.score(x_test, y_test))


def knnGCV():
    """
    K-近邻预测用户入住位置-网格搜索方式
    :return: None
    """

    # 获取数据
    data = pd.read_csv("./data/facebook-v-predicting-check-ins/train.csv")
    # print(data.head(10))

    # 处理数据
        # 1、缩小数据（节省时间），查询数据筛选
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

        # 2、处理时间，将时间戳转化成可视化时间
    time_value = pd.to_datetime(data["time"], unit="s")
    time_value = pd.DatetimeIndex(time_value)  # 把日期格式转换成字典格式

        # 3、构造一些特征
    data["day"] = time_value.day
    data["hour"] = time_value.hour
    # data["weekday"] = time_value.weekday  # 将此特征无视，准确率从47%提升到48%

        # 4、把时间戳特征删除
    data = data.drop(["time"], axis=1)  # 列
    # print(data)

        # 5、把签到数量少于n个的目标位置删除
    place_count = data.groupby("place_id").count()  # 安照“place_id”分组
    tf = place_count[place_count.row_id > 5].reset_index()  # 在网络搜索中，当去除的数据小于5时，报警告有分数内，数据类别小于1，保留“row_id”大于5的数据
    data = data[data["place_id"].isin(tf.place_id)]  # 保留签到次数大于3的数据

    data = data.drop(["row_id"], axis=1)  # 删除无关特征

        # 6、去除数据当中的特征值和目标值
    y = data["place_id"]
    x = data.drop(["place_id"], axis=1)

        # 7、进行数据的分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()
    x_train = std.fit_transform(x_train)  # 对测试集和训练集的特征值进行标准化
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier()

    # 构造一些参数的值进行搜索
    param = {"n_neighbors": [3, 5, 10]}

    # 进行网络搜索
    gc = GridSearchCV(knn, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    print("在测试集上的准确率：", gc.score(x_test, y_test))
    print("在交叉验证当中最好的结果：", gc.best_score_)
    print("选择的最好的模型是：", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果：", gc.cv_results_)


if __name__ == '__main__':
    # knncls()
    knnGCV()
