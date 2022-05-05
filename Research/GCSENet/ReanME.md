# 前言
---------
这里是阅读，李老师学生的那篇文章，所作的一个论文的复现，因为李老师那边是使用tensorflow框架来实现。而我这里是使用pytorch来实现的，所以就需要按照李老师的那个思路了，来重新看看。
# 目前情况
- `GCN`:
  - 是图卷积神经网络用来提取特征
  - 分别对, `疾病语义相似性网络`、`基因网络`、`miRNA功能相似性网络`、`表型相似性网络`、`基因网络`、`miRNA功能相似性网络`进行特征提取
- `CNN`:
  - 将上面的输出，作为输入，使用cnn进行特征提取

# 进展  
- 目前来看，就还是需要对那些数据进行一些特征的处理的，虽然源码中有给出一些数据，但这写数据肯定不能作为gcn的输入数据，需要进一步处理，才能这样做，同时，无论是cnn还是gcn,他们的网络层次结构，更我这里设定的model也都是不一样的。从提供的原始数据来看，一共有三笔数据集，疾病-miRNA,疾病-基因，miRNA-基因；然后通过特征处理之后得到疾病-miRNA的数据集，neg.txt:负样本数据集，pos.txt:正样本数据集。
- 这里就来梳理一下它处理数据的过程吧  ：
  - 1、


