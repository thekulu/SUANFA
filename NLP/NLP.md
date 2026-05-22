## 自然语言处理（NLP）

### NLP入门
- NLP：自动化对人类语言(“自然语言” )的分析、 生成和获取
- 理想情况下，NLP是语言中立的
  - ![alt text](image.png)
  - 实际上，某些语言的NLP得到了更好的发展
    - 更有趣：用户，市场， …
    - 更多资源：开发人员、数据、计算机
  - ![alt text](image-1.png)
- 如何表示自然语言?
  - 理想情况下，是一种表达性强的形式语言
    - 一阶谓词逻辑
    - 编程语言
    - 神经元表示
    - 在实际场景中，取决于应用
      - 标签、特征、指令
- 相关领域 
  - 机器学习
    - ML是NLP中一个强大的(但不是唯一的)工具
    - NLP是ML的灵感来源
  - 语言学
    - 科学 vs. 工程
    - NLP <=> 计算语言学
  - 人工智能
    - NLP是AI的一个子领域
    - NLP是AI皇冠上的宝石
    - 解决NLP需要强大的AI
  - 语音处理
    - 很大程度上与NLP分离，但也有一些重叠
  - 认知科学/神经科学
    - 人类:唯一有效的NLP原型!
  - 逻辑、知识表示和推理
    - NLP分析NL， 并从逻辑语言生成NL
  - 计算理论
    - 学习形式语言和语法
    - 为NLP提供了大量的工具
- 应用
  - 聊天机器人
  - 机器翻译
  - 信息提取
  - 中文输入法
  - 语法检查器
  - 新闻聚类
  - 文本摘要
  - 新闻生成
  - 作文评分
  - 艺术创作
- NLP是困难的
  - 语言学
    - 语音学
    - 音韵学
    - 正字法
    - NLP相关
      - 形态学
      - 语法学
      - 语义学
      - 语篇分析
      - 语用学
  - 情感分析 
  - 世界知识
  - 歧义
  - AI研究面临的共同挑战
    - 高准确率
    - 噪声输入
    - 稀缺数据
    - 隐变量
    - 空间和时间上的计算复杂度
    - 泛化能力
    - 形式化验证
    - 可解释性
- 方法论
  - 符号主义
    - 1970s-1990s：符号主义主导，依赖形式文法和人工规则，只能处理有限、规整的语言场景。
    - “Whenever I fire a linguist, our system performance improves.” – Jelinek, 1988
  - 统计方法
    - 1990s-2010s：统计方法崛起，N-gram 模型、隐马尔可夫模型（HMM）等方法在语音识别、机器翻译等任务中取得突破，验证了数据驱动方法的优势。
    - “…I think the sequence model is very
effective, but at the same time I don't think it will be the ultimate natural language solution. In any case, we will eventually return to the non-sequence model and express many more interesting structures
than the sequence.” –Chris Manning, 2018
  - 联结主义
    - 2010s 后：联结主义（深度学习）复兴，神经网络、Transformer 模型进一步推动 NLP 发展，形成了现在的主流范式。
    - 2015: Deep Learning
    - 2017: Transformer
    - 2018: Pre-training
    - 2022: Large Language Model
  - Future of NLP
    - 语言结构
    - 符号化表示与逻辑推理
    - 数据需求少、计算效率高
    - 可解释性强
    - 扎实的理论根基
- 大纲
  - Basics（基础模块）
    - Text normalization：文本归一化（如分词、大小写统一、去停用词等预处理）
    - Text representation：文本表示（如 TF-IDF、词向量等）
    - Text classification：文本分类（如情感分析、主题分类）
    - Text clustering：文本聚类（如无监督话题聚合）
  - Sequences（序列模块）
    - Language modeling：语言模型（如 N-gram、RNN、Transformer 语言模型）
    - Sequence to sequence：序列到序列模型（如机器翻译、文本生成）
    - Pretrained language models：预训练语言模型（如 BERT、GPT 系列）
    - Large language models：大语言模型（LLM，如 GPT-4、LLaMA 等）
  - Structures（结构模块）
    - Sequence labeling：序列标注（如 NER 命名实体识别、词性标注）
    - Constituency parsing：成分句法分析（构建短语结构树）
    - Dependency parsing：依存句法分析（构建词与词的依存关系）
    - Semantic analysis：语义分析（如语义角色标注、词义消歧）
    - Discourse analysis：语篇分析（如篇章连贯、指代消解）
     
### 文本归一化

- 文本归一化
  - 分词Tokenizing (segmenting) words
  - 归一化单词格式 Normalizing word formats
  - 分句Segmenting sentences
- 分词
  - 基于空格的分词Space-based tokenization
    - 一种非常简单的分词方法，基于空格得到词元token
    - 对于在单词之间使用空格字符的语言，如基于书写系统的阿拉伯语、西里尔语、希腊语、拉丁语等，在空格(和标点符号)实例之间分割
    - 缺陷
      - 不能一味地去掉标点符号:
        - 连字符`-` 
        - 缩写：m.p.h.， Ph.D.， AT&T, cap'n
        - 价格与百分比($45.55)
        - 日期(01/02/06)
        - 网址(https://www.google.com)
        - 邮箱地址(someone@gmail.com)
        - 标签(#ilovenlp)
        - 省略号...
        - 应当正确地得到独立词元separate token
      - 语缀clitic
        - are” 在we're中， 法语“je” 在j'ai中， “le” 在l'honneur中
      - 多词表达multiword expressions (MWE)
        - New York, rock'n'roll
  - 基于正则表达式的分词Tokenization with regular expressions  
    - 正则表达式(RE)
      - RE编写一个匹配所有可能的词元但不匹配任何非词元的模式，输出所有不重叠的匹配
      - 可能内部有连字符的词
        - \w+(-\w+)*
      - 缩写词
        - ([A-Z]\.)+ 
      - 价格与百分比
        - \$?\d+(\.\d+)?%?
      - 省略号
        - \.\.\.
      - 独立词元
        - [][.,;"'?():\-_`]
  - 无空格语言的分词Tokenization in languages without spaces
    - 许多语言(如中文、日语、泰语)不使用空格来分隔单词
    - 一些西方语言的复合名词也一样
      - 德语:Freundschaftsbezeigungen(友好表示)
      - 社交媒体上的话题标签：#TrueLoveInFourWords
    - 分词方式不唯一
      - 姚明/进入/总决赛
      - 姚/明/进入/总/决赛
      - 姚/明/进/入/总/决/赛（单字分割）
    - 歧义
      - 南京市/长江大桥
      - 南京市长/江大桥
    - 通常由序列标注+监督学习解决
  - 子词分词Subword tokenization
    - 子词词元可以是词的一部分，也可以是整个单词
    - 优势
      - 子词有时是有意义的
        - 词缀affix，（前缀prefix、后缀suffix），词干stem……
        - Ex: postprocessing→post, process, ing
      - 词汇量小得多
      - 避免词汇外(OOV)的单词
      - 能从数据中自动学习，不需要额外人工
    - 算法：
      - Byte-Pair Encoding (BPE) (Sennrich et al., 2016)
      - Unigram language modeling tokenization (Kudo, 2018)
      - WordPiece (Schuster and Nakajima, 2012)
      - 都由2部分组成:
        - 词元学习器，采用原始训练语料库归纳出词汇表(一组词元)。
        - 词元分割器，取原始测试句并根据词汇表对其进行分词
      - BPE
        - 算法 
          - 词汇表是所有单个字符的集合= {A, B, C, D， …， A, B, C, D ....}
          - 选择训练语料库中最频繁相邻的两个符号(比如‘A’ 、‘B’ )
          - 在词汇表中添加一个新的合并符号merge'AB'
          - 将语料库中相邻的'A' 'B'替换为'AB'。
          - 进行k次合并结束。
          - 在测试数据上，按照词汇被学习的顺序，用贪婪法运行从训练数据中学习到的每个合并符号
        - Ex
          - 原始语料库：low low low low low lowest lowest newer newer newer newer newer newer wider wider wider new new
          - 对原始语料库运行基于空格的分词算法
          - 添加词尾标记end-of-word tokens“_”，强调后缀
          - 产生词汇表:
            - ![alt text](image-2.png)
          - (e，r)数量最多，合并e，r到er 
            - ![alt text](image-3.png)
          - 合并(er，_)到er_
            - ![alt text](image-4.png)
          - 合并(n，e)到ne
            - ![alt text](image-5.png)
          - 依次进行
            - ![alt text](image-6.png)
          - 在测试集上，将每个(e，r)合并到er，然后将(er，_)合并到er_，等等
          - 测试集“n e w er_” 将被分词为一个完整的单词
          - 测试集“low er_” 将是两个词元:“low er_”
        - BPE词元的属性
          - 经常包括频繁词和频繁子词
            - 比如-est或-er这样的语素Morphemes
          - 语素是语言中最小的意义承载单位
            - unbreakable有3个语素un-， -break-和-able
- 单词归一化
  - 以标准格式放置单词/词元
    - U.S.A. or USA
    - uhhuh or uh-huh
    - Putting or putting
    - am, is, be, are
  - 大小写折叠Case folding
    - 将所有字母缩写为小写
    - 在某些场景下很好
      - 信息检索:用户倾向于使用小写
    - 在其他情况下不太好
      - Ex:General Motors, Fed, US -> general motors, fed, us引发歧义
  - 词根化Lemmatization
    - 将所有单词表示为它们共有的词根，即字典词目格式dictionary headword form
    - 字典匹配是一个简单的方法
  - 词干分析Stemming
    - 由形态分析Morphological Parsing完成词根化
      - 语素Morphemes:
        - 构成单词的小的有意义的单位
        - 词干:承载核心意义的单位
          - 可以用来恢复词根lemma
        - 词缀affix:附着在词干上的部分，通常带有语法功能
      - 形态解析器Morphological Parsers:
        - 将猫解析成两个语素cat和s
        - 把unbelievable解析成un, believe和able三个语素 
    - 将词语简化为词干，粗切词缀
      -是词根化的一种简单粗暴的方法
    - Porter Stemmer
      - 基于一系列串联运行的重写规则
        - 级联，其中每个通道的输出馈送到下一个通道
        - ![alt text](image-7.png)
  - Ex
    - Sentence:
      - This was not the map we found in Billy Bones's chest.
    - Case folding：
      - this was not the map we found in billy bones's chest.  
    - Stemming:
      - Thi wa not the map we found in Billi Bone s chest.
    - Lemmatizing:
      - This be not the map we find in Billy Bones 's chest.
- 分句
  - !，?大多数情况无歧义，但句号.是非常模棱两可的
    - 不同句子的分隔
    - 像Inc.或Dr.这样的缩写。
    - 像0.02%或4.3这样的数字
  - 常用算法
    - 以分词为主
      - 在分词中，句号被分类为单词或句子的一部分
    - 基于分词，分句通常可以通过规则来完成
      - 标点符号：被分词出来的句号一般是真正的句末句号
      - 大写字母：句号后的token开头为大写字母，大概率是新句子

### 文本表示

- 单词表示
  - 基本概念
    - Ludwig Wittgenstein (1953): The meaning of a word is its use in the language
    - 我们该如何定义“使用” ?
    - 对“使用”的一个定义:在什么样的上下文中出现
    - Zellig Harris (1954): If A and B have almost identical environments we say that they are synonyms.
    - 上下文context就是环境
    - Ex
      - “青菜”是什么意思?
      - 假设你看到这些句子:
        - 青菜和大蒜一起炒，味道很美味。 
        - 青菜配米饭吃，味道超棒。
        - 配咸酱吃的青菜叶子。
      - 你也见过这些:
        - 菠菜和大蒜一起炒，配米饭吃。
        - 生菜的茎和叶都很美味。
        - 甘蓝和其他带咸味的绿叶蔬菜
      - 你可以推断：青菜是一种可烹饪食用的绿叶蔬菜
  - 而在传统的NLP中，我们将单词视为离散符号
    - 没有天然的相似性概念
      - 汽车旅馆vs酒店：语义上高度相关，但在传统表示中完全不相关。
    - 可能依赖于一个词库，例如WordNet，来衡量相似性，但是编译或更新同义词库需要大量的工作，存在以下缺点: 
      - 对于低资源语言/领域不可用
      - 容易过时
      - 经常缺少许多单词和短语
- 词向量
  - 用向量表示向量空间中的每个单词
    - 初始向量为独热编码 
    - 可以用向量相似度捕获词之间的相似度
    - 在许多ML系统中更容易处理
    - 可以从数据中学习
  - 类型
    - 稀疏向量表示
      - 共现矩阵co-occurrence matrices
        - Word-word matrix
          - 又称term-term矩阵
          - Word-word matrix是|V|x|V|
          - 每个单元格:单词共现的计数
          - 不在整篇文章中计数共现，使用较小的上下文
            - 段落
            - 前后k个单词的窗口
          - 单词现在由一个向量来表示，向量由单词共现的计数决定
          - Ex（前后7个单词的窗口）
            - ![alt text](image-8.png)
          - 我们只展示了4x6，但真正的矩阵是50000 x50000
          - 所以Word-word matrix是非常稀疏的，大多数值都是0。
            - 对于稀疏矩阵，有很多高效的算法。
          - 窗口的大小取决于目标。
            - 窗口越短(k=1-3)，向量表示越倾向于表示单词的语法
            - 窗口越长(k=4-10)，向量表示越倾向于表示单词的语义
        - 共现频率悖论 
          - 共现矩阵的每个单元格是共现频率
          - 共现频率显然是有用的:如果糖与菠萝的共现频率高，那是有用的信息。
          - 但原始频率是一个糟糕的表示:
            - 过于频繁的单词，比如the,it等不能很好地提供上下文信息。然而，它们的数量经常主导向量
          - 如何处理这种悖论?
        - PMI(点对互信息，Pointwise Mutual Information)
          - 将单词作为基本事件。事件x和y共同出现的次数比它们独立出现的次数多还是少? 
            \[
            PMI(X, Y) = \log_2 \frac{P(X, Y)}{P(X)P(Y)}
            \]
        - PPMI（正点对互信息，Positive Pointwise Mutual Information）
          - PMI 的取值范围为 $(-\infty, +\infty)$
          - 但PMI的负值存在明显问题
            - PMI的负值表示两个事物的共现频率低于随机预期值
            - 在没有大规模语料库的情况下，负值并不可靠
              - 假设词 $w_1$ 和 $w_2$ 的出现概率均为 $10^{-6}$
              - 此时很难判断 $P(w_1, w_2)$ 是否显著低于 $10^{-12}$
              - \(10^{-12}\) 这个概率极小，哪怕有 100 亿词的语料，期望共现次数也只有 10 次。实际语料中，两个词一次都没共现，完全可能只是样本不足
            - 此外，语言学中“负相关性”的定义本身并不明确
              - 自然语言中，词与词之间的强负相关几乎不存在，大部分 “不共现” 只是语义无关，而不是相互排斥。 
          - 因此直接将 PMI 的负值替换为 0
          - 词 $w_1$ 和 $w_2$ 之间的正点互信息（PPMI）定义为：
            \[
            PPMI(w_1, w_2) = \max\left(\log_2 \frac{P(w_1, w_2)}{P(w_1)P(w_2)}, 0\right)
            \]
          - Ex
            - 共现矩阵：W行C列的矩阵F
              - 行:一个单词
              - 列:一个上下文(也是一个单词)- fij是wi在context cj中出现的次数
              - $$
                p_{ij} = \frac{f_{ij}}{\sum_{i=1}^{W} \sum_{j=1}^{C} f_{ij}}
                $$
                $$
                p_{i*} = \frac{\sum_{j=1}^{C} f_{ij}}{\sum_{i=1}^{W} \sum_{j=1}^{C} f_{ij}}
                $$
                $$
                p_{*j} = \frac{\sum_{i=1}^{W} f_{ij}}{\sum_{i=1}^{W} \sum_{j=1}^{C} f_{ij}}
                $$
                $$
                PMI_{ij} = \log_2 \frac{p_{ij}}{p_{i*} \, p_{*j}}
                $$
                $$
                PPMI_{ij} =
                \begin{cases}
                PMI_{ij}, & PMI_{ij} > 0 \\
                0, & \text{otherwise}
                \end{cases}
                $$
              - ![alt text](image-9.png)
              - ![alt text](image-10.png) 
              - ![alt text](image-11.png)
        - 加权PMI
          - PMI偏向于罕见事件
            - 非常罕见的单词具有非常高的PMI值
          - 两种解决方案
            - 给稀有词稍高的概率
            - 使用add-k拉普拉斯平滑(效果类似)
            - 只平滑非零和有效的PPMI值  
            - Ex
            - ![alt text](image-12.png)
            - ![alt text](image-13.png)
      - 稠密向量表示
        - 稠密向量通常比稀疏向量效果更好
          - 维度更低，消除了表示不重要信息的维度，降低噪声
          - 语义稠密，泛化能力更强
          - 更少的维度使模型参数总量变少，更容易训练和使用
        - 奇异值分解SVD
          - 是一种通用矩阵分解、降维算法，能把任意矩阵拆成3个正交矩阵的乘积，常用来对高维稀疏语义矩阵做压缩，生成低维稠密向量。 
          - 任意实数矩阵 $\boldsymbol{A}_{m\times n}$，均可分解：
            $$
            \boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top
            $$
            - $\boldsymbol{U}$：左奇异向量矩阵，代表词的稠密表示
            - $\boldsymbol{\Sigma}$：奇异值对角矩阵，数值越大代表承载语义信息越多
            - $\boldsymbol{V}$：右奇异向量矩阵，代表上下文表示
          - 原始稀疏 PPMI 矩阵 \(\boldsymbol{A}\)
          - SVD 拆分：\(\boldsymbol{A}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top\)
          - 舍弃小奇异值噪声维度，截取前 2 维核心分量
          - 矩阵相乘得到低维稠密词向量
          - Ex
            - PPMI矩阵
            $$
            \boldsymbol{A}=
            \begin{bmatrix}
            0 & 1.32 & 1.32 & 0 & 0 & 0 & 0\\
            1.32 & 0 & 0 & 0.74 & 1.32 & 0 & 0\\
            1.32 & 0 & 0 & 0.74 & 1.32 & 0 & 0\\
            0 & 0.74 & 0.74 & 0 & 0 & 1.74 & 0\\
            0 & 1.32 & 1.32 & 0 & 0 & 0 & 0\\
            0 & 0 & 0 & 1.74 & 0 & 0 & 3.32\\
            0 & 0 & 0 & 0 & 0 & 3.32 & 0
            \end{bmatrix}
            $$

            - 左奇异矩阵$\boldsymbol{U}$
            $$
            \boldsymbol{U}=
            \begin{bmatrix}
            -0.610 & 0.002 & -0.215 & 0.108 & -0.053 & 0.021 & -0.009\\
            -0.502 & 0.301 & 0.422 & -0.206 & 0.101 & -0.042 & 0.018\\
            -0.502 & 0.301 & 0.422 & -0.206 & 0.101 & -0.042 & 0.018\\
            -0.201 & -0.603 & 0.105 & 0.412 & -0.203 & 0.083 & -0.034\\
            -0.201 & 0.301 & -0.527 & -0.304 & 0.152 & -0.061 & 0.025\\
            0.302 & -0.603 & -0.317 & -0.412 & -0.203 & -0.083 & 0.034\\
            0.604 & 0.002 & -0.105 & 0.108 & 0.053 & 0.021 & 0.009
            \end{bmatrix}
            $$

            - 奇异值对角矩阵$\boldsymbol{\Sigma}$
            $$
            \boldsymbol{\Sigma}=
            \begin{bmatrix}
            4.250 & 0 & 0 & 0 & 0 & 0 & 0\\
            0 & 2.180 & 0 & 0 & 0 & 0 & 0\\
            0 & 0 & 0.860 & 0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0.410 & 0 & 0 & 0\\
            0 & 0 & 0 & 0 & 0.220 & 0 & 0\\
            0 & 0 & 0 & 0 & 0 & 0.090 & 0\\
            0 & 0 & 0 & 0 & 0 & 0 & 0.030
            \end{bmatrix}
            $$

            - 右奇异转置矩阵$\boldsymbol{V}^\top$
            $$
            \boldsymbol{V}^\top=
            \begin{bmatrix}
            -0.608 & -0.501 & -0.501 & -0.200 & -0.200 & 0.301 & 0.602\\
            0.001 & 0.300 & 0.300 & -0.601 & 0.300 & -0.601 & 0.001\\
            -0.213 & 0.420 & 0.420 & 0.103 & -0.525 & -0.315 & -0.103\\
            0.106 & -0.204 & -0.204 & 0.410 & -0.302 & -0.410 & 0.106\\
            -0.052 & 0.100 & 0.100 & -0.201 & 0.150 & -0.201 & 0.052\\
            0.020 & -0.041 & -0.041 & 0.081 & -0.060 & -0.081 & 0.020\\
            -0.008 & 0.017 & 0.017 & -0.033 & 0.024 & 0.033 & 0.008
            \end{bmatrix}
            $$
            - 截取二维子矩阵
            $$
            \boldsymbol{U}_2=
            \begin{bmatrix}
            -0.610 & 0.002\\
            -0.502 & 0.301\\
            -0.502 & 0.301\\
            -0.201 & -0.603\\
            -0.201 & 0.301\\
            0.302 & -0.603\\
            0.604 & 0.002
            \end{bmatrix},\quad
            \boldsymbol{\Sigma}_2=
            \begin{bmatrix}
            4.250 & 0\\
            0 & 2.180
            \end{bmatrix}
            $$
            - 稠密向量计算式
            $$\boldsymbol{Emb} = \boldsymbol{U}_2 \boldsymbol{\Sigma}_2$$
            - 最终稠密向量
            $$
            \boldsymbol{Emb}=
            \begin{bmatrix}
            -2.5925 & 0.0044\\
            -2.1335 & 0.6562\\
            -2.1335 & 0.6562\\
            -0.8543 & -1.3145\\
            -0.8543 & 0.6562\\
            1.2835 & -1.3145\\
            2.5670 & 0.0044
            \end{bmatrix}
            $$
        - 神经网络方法（词嵌入word embeddings）
          - 静态词嵌入
            - word2vec
              - Word2vec (Mikolov et al. 2013)是一个学习词向量的框架
              - 算法:
                - 我们有一个很大的文本语料库
                - 遍历文本中的每个位置t， 它有一个中心
                单词c和几个上下文(“外部” )单词o
                - 假设每个单词都由一个向量表示
                - 使用c和o的词向量的相似性来计算给定c时o的概率（skip-grams）
                - 也可使用c和o的词向量的相似性来计算给定o时c的概率(CBOW)
                - 不断调整单词向量，使这个概率最大
              - skip-grams
                - 给定中心单词,预测上下文(“外部” )单词(位置无关，不区分上下文词的左右位置)
                - ![alt text](image-14.png) 
                - 如何计算𝑃(𝑤𝑡+𝑗|𝑤𝑡)
                - 每个单词w有两个向量:
                  - vw（w是中心词）
                  - uw（w是上下文词）
                - 对于中心词c和上下文词o
                  $$
                  P(o|c) = \frac{\exp(u_o^T v_c)}{\sum_{w \in V} \exp(u_w^T v_c)}
                  $$
                  - 这被称为softmax函数
                    - 将任意值映射到概率分布
                    - “max”，因为它放大了最大值
                    - “soft”，因为它仍然保留了较小的值
                  - 对于文本序列中的每个位置 $t=1,\dots,T$，给定中心词 $w_t$，预测固定窗口 $m$ 内的上下文词。
                  - 似然函数（Likelihood）
                    $$
                    L(\theta) = \prod_{t=1}^{T} \prod_{\substack{-m \le j \le m \\ j \ne 0}} P(w_{t+j} \mid w_t; \theta)
                    $$
                    - 其中$\theta$是所有待优化的参数
                    - 目标函数（也称代价函数或损失函数）
                      $$
                      J(\theta) = -\frac{1}{T} \log L(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-m \le j \le m \\ j \ne 0}} \log P(w_{t+j} \mid w_t; \theta)
                      $$
                  - 核心关系
                    $$
                    \text{最小化目标函数 } J(\theta) \iff \text{最大化预测准确率}
                    $$
                - softmax目标函数需要归一整个词汇表，对于大词汇表来说很昂贵
                - 共现概率目标函数
                  - 对中心词 $c$ 和上下文词 $o$，模型直接建模共现概率：
                    $$
                    P(\text{co-occur} \mid c, o) = \sigma(\boldsymbol{u}_o^T \boldsymbol{v}_c)
                    $$
                    其中 $\sigma(z) = \frac{1}{1+e^{-z}}$ 为 sigmoid 函数。
                  - 转化任务为二分类任务，无需对归一整个词汇表
                  - 但是，如果只最大化训练语料中所有 $(c,o)$ 对的 $P(\text{co-occur} \mid c, o)$，模型为了让任意两个共现词的内积都大，会把所有向量都推向同一个方向，失去区分度。
                  - 之前的 softmax 目标不会出现这个问题，因为它需要对所有词的概率做归一化，天然存在竞争。
                  - 解决方案：负采样
                    - 对每个中心词 $c$，根据词频分布采样 $K$ 个负样本 $o$（未在窗口中出现的词）。
                    - 最大化正样本共现概率，同时最小化负样本的共现概率。
                    - 对于每对c-o
                      $$
                      \begin{aligned}
                      J(\theta, c, o) &= -\log P(\text{co-occur} \mid c, o) - \log \prod_{k=1}^{K} \left(1 - P(\text{co-occur} \mid c, o_k)\right) \\
                      &= -\log\left(\sigma(\boldsymbol{u}_o^T \boldsymbol{v}_c)\right) - \sum_{k=1}^{K} \log\left(1 - \sigma(\boldsymbol{u}_k^T \boldsymbol{v}_c)\right) \\
                      &= -\log\left(\sigma(\boldsymbol{u}_o^T \boldsymbol{v}_c)\right) - \sum_{k=1}^{K} \log\left(\sigma(-\boldsymbol{u}_k^T \boldsymbol{v}_c)\right)
                      \end{aligned}
                      $$          
                    - 最大化真实上下文词 $o$ 与中心词 $c$ 的共现概率,最小化 $K$ 个负样本词 $o_k$ 与中心词 $c$ 的共现概率（对比学习）
                    - 整体目标函数
                      $$
                      J(\theta) = \frac{1}{T} \sum_{t} \sum_{\substack{-m \le j \le m \\ j \ne 0}} J(\theta, w_t, w_{t+j})
                      $$
                - 优化Optimization
                  - 随机梯度下降Stochastic gradient descent
                    - 梯度下降 
                      - 在最陡的下坡方向进行参数更新
                    - 随机的
                      - 从许多训练样例中计算目标参数
                      - 只根据一个训练样例(或一个Mini-batch的例子)的梯度进行参数更新
                  - ![alt text](image-15.png)
                - 每个单词有两个嵌入vw和uw，训练完毕后，下游任务只需要一个词向量。我们可以
                  - 只使用vw
                  - vw+uw
                  - vw||uw
              - CBOW(Continuous Bag of Words)
                - 从(一袋)上下文词中预测中心词（位置无关，不关心上下文词的顺序）
            - GloVe
            - 评估词向量 
              - 内在的Intrinsic:
                - 对特定子任务的评估
                - 计算速度快
                - 有助于理解这个系统
                - 不清楚是否真的有帮助， 除非与下游相关
                - 词向量类比
                  - a+b-c =?d
                    - king+woman-man=?queen
                  - $$
                    d = \arg\max_{i} \frac{(\boldsymbol{x}_b - \boldsymbol{x}_a + \boldsymbol{x}_c)^\top \boldsymbol{x}_i}{\|\boldsymbol{x}_b - \boldsymbol{x}_a + \boldsymbol{x}_c\|}
                    $$
                  - 通过线性代数操作，来测试词向量是否学到了合理的语义 / 语法关系
                  - 无法测试非线性关系                 
              - 外在的Extrinsic:
                - 对下游任务的评价
                - 会花很长时间
                - 结果可能与中间系统，下游系统或者它们之间的相互作用有关   
          - 上下文相关向量表示Contextualized vector representations          
                                                                            
                                                                                            
                                                                              