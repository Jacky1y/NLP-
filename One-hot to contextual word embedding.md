one-hot encoding编码

通常我们有很多的词，那只在出现的位置显示会，那么势必会存在一些问题

1、高维的表示
2、稀疏性
3、正交性（任意两个词的距离都是1，除了自己和自己，这样就带来一个问题，猫和狗距离是1，猫和石头距离也是1，但我们理解上猫和狗距离应该更近一些）

两个词语义上无法正确表示，我们更希望低维的相似的比较接近，语义相近的词距离比较近，语义不想近的词，距离也比较远。


word2vec
word2vec分为cbow（根据context预测中心词）和skip-gram（根据中心词预测context）两种


RNN的两个问题：

1、顺序依赖，t依赖t-1时刻。
2、单向信息流（如例子中指代信息，不能确定）
3、需要一些比较多的监督数据，对于数据获取成本很高的任务，就比较困难，在实际中很难学到复杂的上下文关系


Contextual Word Embedding
要解决RNN的问题，就引入了contextual word embedding。

contextual word embedding:无监督的上下文的表示，这种无监督的学习是考虑上下文的，比如ELMo、OpenAI GPT、BERT都是上下文相关的词的表示方法。

attention是需要两个句子的，我们很多时候只有一个句子，这就需要self-attention。提取信息的时候、编码时self-atenntion是自驱动的，self-attention关注的词的前后整个上下文。

transformer:
本质也是一个encoder与decoder的过程，最起初时6个encoder与6个decoder堆叠起来，如果是LSTM的话，通常很难训练的很深，不能很好的并行.

transformer 解决的问题：
可以并行计算，训练的很深，到后来的open gpt可以到12层 bert的16、24层
单向信息流的问题：至少在encoder的时候考虑前面和后面的信息，所以可以取得很好的效果，
transformer解决了普通word embedding 没有上下文的问题，但是解决这个问题，需要大量的标注信息样本。

如何解决transformer的问题，就引入了elmo
elmo:无监督的考虑上下文的学习。

一个个的预测的语言模型：
双向的lstm，每个向量2n，是一种特征提取的方法，考虑的上下文的，编码完，就定住了，

elmo：将上下文当作特征，但是无监督的语料和我们真实的语料还是有区别的，不一定的符合我们特定的任务，是一种双向的特征提取。

openai gpt就做了一个改进，也是通过transformer学习出来一个语言模型，不是固定的，通过任务 finetuning,用transfomer代替elmo的lstm。
openai gpt其实就是缺少了encoder的transformer。当然也没了encoder与decoder之间的attention。

openAI gpt虽然可以进行fine-tuning,但是有些特殊任务与pretraining输入有出入，单个句子与两个句子不一致的情况，很难解决，还有就是decoder只能看到前面的信息。

bert
bert从这几方面做了改进：

Masked LM
NSP Multi-task Learning
Encoder again
bert为什么更好呢？

单向信息流的问题 ,只能看前面，不能看后面，其实预料里有后面的信息，只是训练语言模型任务特殊要求只能看后面的信息，这是最大的一个问题
其次是pretrain 和finetuning 几个句子不匹配


原文链接：
https://blog.csdn.net/mr2zhang/article/details/91958053
