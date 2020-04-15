from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import jieba  # 中文分词
import numpy as np


def dictvec():
    """
    字典数据抽取
    :return: None
    """

    dict = DictVectorizer(sparse=False)  # 实例化

    data = dict.fit_transform([{'city': '北京', 'temperature': 100},
                               {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}])

    print(dict.get_feature_names())
    # print(dict.inverse_transform(data))
    print(data)


def countvec():
    """
    对文本进行特征值化
    :return: None
    """

    cv = CountVectorizer()

    # 英文
    englishData = cv.fit_transform(["life is short,i like python", "life is too long,i dislike python"])
    print(cv.get_feature_names())
    print(englishData.toarray())

    # 中文
    content1 = jieba.cut("人生苦短，我喜欢python,人生漫长，不用python")  # 分词
    result1 = " ".join(list(content1))  # 每个词后面加一个空格
    content2 = jieba.cut("成分：水，均匀涂抹到身体上")  # 分词
    result2 = " ".join(list(content2))  # 每个词后面加一个空格
    chineseData = cv.fit_transform([result1, result2])
    print(cv.get_feature_names())
    print(chineseData.toarray())


def tfidfvec():
    """
    特征值重要性提取，TF-IDF方法
    :return: None
    """

    content1 = jieba.cut("人生苦短，我喜欢python,人生漫长，不用python")  # 分词
    result1 = " ".join(list(content1))  # 每个词后面加一个空格

    content2 = jieba.cut("成分：水，喜欢均匀涂抹到身体上")  # 分词
    result2 = " ".join(list(content2))  # 每个词后面加一个空格

    tf = TfidfVectorizer()
    chineseData = tf.fit_transform([result1, result2])
    print(tf.get_feature_names())
    print(chineseData.toarray())


def normalization():
    """
    归一化处理
    :return: None
    """

    mm = MinMaxScaler(feature_range=(0, 1))  # feature_range: 归一化后的数值范围
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)


def stand():
    """
    标准化缩放
    :return: None
    """

    std = StandardScaler()
    data = std.fit_transform([[1., -1., 3.], [2., 4., 2.], [4., 6., -1.]])
    print(data)


def sim():
    """
    缺失值处理
    :return: None
    """

    im = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)


def var():
    """
    特征选择-删除低方差的特征
    :return: None
    """

    var = VarianceThreshold(threshold=0.0)  # 要降维的最小方差阈值
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)


def pca():
    """
    主成份分析进行特征降维
    :return: None
    """

    pca = PCA(n_components=0.9)
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # tfidfvec()
    # normalization()
    # stand()
    # sim()
    # var()
    pca()
