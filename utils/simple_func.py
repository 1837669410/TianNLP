import numpy as np

def list_to_str(data):
    '''读取每行数据，用join生成器返回字符串并
    添加到数组中返回
    Args:
        data：需要转换的数据

    Return:
        list：返回转换后的数据

    Examples:
    ------------------------------------------------
        >>> data = np.array([["人们", "在", "过马路"],
                         ["上班", "的", "人"]])
        >>> new_data = list_to_str(data)
        # new_data = ['人们在过马路' '上班的人']
    ------------------------------------------------
    '''
    result = []
    for c in data:
        result.append("".join(c))
    result = np.array(result)
    return result

def get_stopwords_list(path):
    '''
    # 导入停用词
    :param path: 停用词表的路径
    :return: 停用词列表
    '''
    stop_words = []
    with open(path, "r", encoding="utf-8") as fp:
        stop_words.extend(fp.read().split("\n"))
    return stop_words