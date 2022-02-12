# EDA
import jieba
import numpy as np
import synonyms

class EDA():

    def __init__(self):
        self.stop_words = []

    def get_stop_words(self, path):
        '''
        # 导入停用词
        :param path: 停用词表的路径
        :return: 停用词列表
        '''
        fp = open(path, "r", encoding="utf-8")
        for stop_word in fp.readlines():
            self.stop_words.append(stop_word.replace("\n", ""))

    def synonym_replace(self, sentence, alpha=0.1, n_aug=4):
        '''
        # 同义词替换
        :param sentence: 需要增强的句子
        :param alpha: 增强参数(0-1),通常选择0.05,0.1
        :param n_aug: 增广句的数量
        :return: 返回生成后的文本(numpy格式数据),第一句为原句子，后面的为增广句
        '''
        # 计算需要改变的单词的数量n
        n = round(alpha*len(sentence))
        # 防止因为句子较短生成的n=0
        if n == 0:
            n = 1
        sentence = jieba.lcut(sentence)
        # 构建返回句子，但是第一句为原句
        new_sentence = np.array([sentence]).repeat(n_aug+1, axis=0)
        for i in range(1, len(new_sentence)):
            # 初始化需要同义词替换的点位
            replace_points = np.random.choice(np.arange(0, len(sentence)), size=n, replace=False)
            while 1:
                # 判断是否有停用词
                for p in replace_points:
                    if sentence[p] in self.stop_words:
                        replace_points = np.random.choice(np.arange(0, len(sentence)), size=n, replace=False)
                        break
                else:
                    break
            for p in replace_points:
                new_word = synonyms.nearby(new_sentence[i, p])[0]
                if len(new_word) == 0:
                    # 判断是否找到了同义词，如果没找到则不改变
                    continue
                new_sentence[i, p] = np.random.choice(new_word)
        return new_sentence

    def random_insertion(self, sentence, alpha=0.1, n_aug=4):
        '''
        # 随机插入
        :param sentence: 需要增强的句子
        :param alpha: 增强参数(0-1),通常选择0.05,0.1
        :param n_aug: 增广句的数量
        :return: 返回生成后的文本(list格式数据),第一句为原句子，后面的为增广句
        '''
        # 计算需要插入的index的数量
        n = round(alpha*len(sentence))
        # 防止因为句子较短生成的n=0
        if n == 0:
            n = 1
        sentence = jieba.lcut(sentence)
        new_sentence = []
        new_sentence.append(sentence)
        for i in range(n_aug):
            # 初始化需要插入的index
            replace_points = np.random.choice(np.arange(0, len(sentence)), size=n, replace=False)
            while 1:
                # 判断选到的index位置的词是否为停用词
                for p in replace_points:
                    if sentence[p] in self.stop_words:
                        replace_points = np.random.choice(np.arange(0, len(sentence)), size=n, replace=False)
                        break
                else:
                    break
            content = sentence.copy()
            for p in replace_points:
                # 插入过程
                nearby_word = synonyms.nearby(sentence[p])[0]
                if len(nearby_word) == 0:
                    # 判断是否找到了同义词，如果没找到则不改变
                    continue
                # 随机选择插入的index
                insertion_index = np.random.randint(0, len(content), size=1)
                # 在随机选择的位置插入一个随机选择的同义词
                content.insert(int(insertion_index), np.random.choice(nearby_word))
            new_sentence.append(content)
        new_sentence = new_sentence
        return new_sentence

    def random_swap(self, sentence, alpha=0.05, n_aug=4):
        '''
        # 随机交换
        :param sentence: 需要增强的句子
        :param alpha: 增强参数(0-1),通常选择0.05,0.1
        :param n_aug: 增广句的数量
        :return: 返回生成后的文本(numpy格式数据),第一句为原句子，后面的为增广句
        '''
        # 计算需要交换的次数
        n = round(alpha*len(sentence))
        # 防止因为句子较短生成的n=0
        if n == 0:
            n = 1
        sentence = jieba.lcut(sentence)
        # 构建返回句子，但是第一句为原句
        new_sentence = np.array([sentence]).repeat(n_aug+1, axis=0)
        for i in range(1, len(new_sentence)):
            for j in range(n):
                # 生成交换点
                swap_points = np.random.choice(np.arange(0, len(sentence)), size=2, replace=False)
                new_sentence[i, swap_points[0]], new_sentence[i, swap_points[1]] = new_sentence[i, swap_points[1]], new_sentence[i, swap_points[0]]
        new_sentence = np.array(new_sentence)
        return new_sentence

    def random_delete(self, sentence, alpha=0.05, n_aug=4):
        '''
        # 随机删除
        :param sentence: 需要增强的句子
        :param alpha: 增强参数(0-1),通常选择0.05,删除的概率
        :param n_aug: 增广句的数量
        :return: 返回生成后的文本(list格式数据),第一句为原句子，后面的为增广句
        '''
        sentence = jieba.lcut(sentence)
        new_sentence = []
        new_sentence.append(sentence)
        for i in range(n_aug):
            content = sentence.copy()
            if np.random.rand() < alpha:
                # 随机选择一个删除点
                delete_point = np.random.randint(0, len(sentence), size=1)
                content.pop(int(delete_point))
            new_sentence.append(content)
        return new_sentence
