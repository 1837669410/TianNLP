# from TianNLP.Augmentation import *
from TianNLP.utils import *
from TianNLP.algorithm import *

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 模仿了一下jioNLP的写法

__version__ = "0.0.1"

history = """
╭──────────────────────────────────────────────────────────────────────────╮
│ • • • ░░░░░░░░░░░░░░░░░░░░░  History Messages  ░░░░░░░░░░░░░░░░░░░░░░░░░ │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│       JioNLP, a python tool for Chinese NLP preprocessing & parsing.     │
│               URL: https://github.com/dongrixinyu/JioNLP                 │
│                                                                          │
│   | date       | updated funcs and info                              |   │
│   | ---------- | --------------------------------------------------- |   │
│   | 2022-02-12 | first push                                          |   │
│                                                                          │
╰──────────────────────────────────────────────────────────────────────────╯
"""