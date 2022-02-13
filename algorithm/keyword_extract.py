from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import jieba
import matplotlib.pyplot as plt

def tf_idf(text, Top_K=10, return_v2i=False, return_idf_=True, return_tfidf=False, use_visual=False):
    '''
    文本关键词提取
    Args:
        text: 待提取的文本
        Top_K: 返回几个TF/IDF权重最大的关键词，默认值为10
        return_v2i: 是否返回{v:i}字典，默认为False
        return_idf_：是否返回idf
        return_tfidf：是否返回tfidf
        use_visual: 是否可视化TF-IDF，默认为False

    Returns:
        word：关键词
        score：关键词的tf-idf值
        vectorizer.vocabulary_：{word_name:tfidf}
        idf_：idf

    Examples:
        // 1、return_idf_=True和use_visual=True时数据太大时不建议开启可视化
        >>> import TianNLP as TN
        >>> x_train = [
            "it is a good day, I like to stay here",
            "I am happy to be here",
            "I am bob",
            "it is sunny today",
            "I have a party today",
        ]
        >>> name, score = TN.tf_idf(x_train, return_tfidf= True ,use_visual=True)
        >>> print(name, score)
        // 2、return_tfidf=True，只建议单个句子单个句子的输入，或者投入整个文章，不建议开启可视化
        >>> import TianNLP as TN
        >>> x_train = [
            "it is a good day, I like to stay here",
        ]
        >>> name, score = TN.tf_idf(x_train, return_tfidf= True ,use_visual=True)
        >>> print(name, score)
    '''
    text = [" ".join(jieba.cut(c)) for c in text]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(text)

    # 获得每个词的tfidf值，因为tfidf是一个稀疏矩阵
    feature_names = vectorizer.get_feature_names()
    name, score = [], []
    for i in range(tfidf.shape[0]):
        tfidf_index = tfidf[i,:].nonzero()[1]
        for j in tfidf_index:
            name.append(feature_names[j])
            score.append(tfidf[i,j])
    name, score = np.array(name)[np.argsort(score)][::-1], np.array(score)[np.argsort(score)][::-1]

    if use_visual:
        show_tf_idf(tfidf.todense(), [v for v in vectorizer.vocabulary_.keys()])
    if return_v2i:
        return vectorizer.vocabulary_
    if return_tfidf:
        return [(n, s) for n, s in zip(name[:Top_K], score[:Top_K])]
    if return_idf_:
        return {n: v for n, v in zip(vectorizer.get_feature_names(), vectorizer.idf_)}

def show_tf_idf(tfidf_matrix, vocab):
    plt.imshow(tfidf_matrix, cmap="YlGn")
    plt.xticks(np.arange(0,tfidf_matrix.shape[1]), vocab, fontsize=7, rotation=90)
    plt.yticks(np.arange(0,tfidf_matrix.shape[0]), np.arange(1, tfidf_matrix.shape[0]+1), fontsize=7)
    plt.title("tfidf可视化")
    plt.show()

if __name__ == "__main__":
    x_train = [
        "it is a good day, I like to stay here",
        "I am happy to be here",
        "I am bob",
        "it is sunny today",
        "I have a party today",
    ]
    name, score = tf_idf(x_train, return_tfidf=True)
    print(name, score)