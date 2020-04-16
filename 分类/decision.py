import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return: None
    """

    # 获取数据
    titan = pd.read_csv("./data/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[["pclass", "age", "sex"]]
    y = titan["survived"]
    print(x)

    # 缺失值处理
    x["age"].fillna(x["age"].mean(), inplace=True)

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理（特征工程） 特征-》类别-》one-hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    print(x_train)
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 用决策树继续预测
    dec = DecisionTreeClassifier(max_depth=8)  # 树的最大深度，可防止模型过拟合
    dec.fit(x_train, y_train)

    # 预测准确率
    print("预测的准确率：", dec.score(x_test, y_test))

    # 导出决策树的结构
    export_graphviz(dec, out_file="./outfile/tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', 'sex=male'])


def random_forest():
    """
    随机森林对泰坦尼克号进行预测生死
    :return: None
    """

    # 获取数据
    titan = pd.read_csv("./data/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[["pclass", "age", "sex"]]
    y = titan["survived"]
    # print(x)

    # 缺失值处理
    x["age"].fillna(x["age"].mean(), inplace=True)

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理（特征工程） 特征-》类别-》one-hot编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    # print(dict.get_feature_names())
    # print(x_train)
    x_test = dict.transform(x_test.to_dict(orient="records"))

    # 随机森林进行预测（超参数调优）
    rf = RandomForestClassifier()

    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}  # 根据经验，一般是这些

    # 网格搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    print("准确率：", gc.score(x_test, y_test))
    print("选择的参数模型：", gc.best_params_)


if __name__ == '__main__':
    # decision()
    random_forest()
