from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import jieba
import matplotlib.pyplot as plt

def tf_idf(text, Top_K=10, return_v2i=False, use_visual=False):
    text = [" ".join(jieba.lcut(c)) for c in text]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(text)

    # 获得每个词的tfidf值，因为tfidf是一个稀疏矩阵
    feature_names = vectorizer.get_feature_names()
    name, score = [], []
    for i in range(tfidf.shape[0]):
        tfidf_index = tfidf[i, :].nonzero()[1]
        for j in tfidf_index:
            name.append(feature_names[j])
            score.append(tfidf[i,j])
    name, score = np.array(name)[np.argsort(score)][::-1], np.array(score)[np.argsort(score)][::-1]

    show_tf_idf(tfidf.todense(), [v for v in vectorizer.vocabulary_.keys()])

    if return_v2i:
        return name[:Top_K], score[:Top_K], vectorizer.vocabulary_

    return name[:Top_K], score[:Top_K]

def show_tf_idf(tfidf_matrix, vocab):
    plt.imshow(tfidf_matrix, cmap="YlGn")
    plt.xticks(np.arange(0,tfidf_matrix.shape[1]), vocab, fontsize=7, rotation=90)
    plt.yticks(np.arange(0,tfidf_matrix.shape[0]), np.arange(1, tfidf_matrix.shape[0]+1), fontsize=7)
    plt.title("tfidf可视化")
    plt.show()

if __name__ == "__main__":
    x_train = docs = [
        "it is a good day, I like to stay here",
        "I am happy to be here",
        "I am bob",
        "it is sunny today",
        "I have a party today",
        "it is a dog and that is a cat",
        "there are dog and cat on the tree",
        "I study hard this morning",
        "today is a good day",
        "tomorrow will be a good day",
        "I like coffee, I like book and I like apple",
        "I do not like it",
        "I am kitty, I like bob",
        "I do not care who like bob, but I like kitty",
        "It is coffee time, bring your cup",
    ]
    name, score = tf_idf(x_train)
    print(name, score)