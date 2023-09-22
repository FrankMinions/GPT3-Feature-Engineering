# GPT3-Feature-Engineering

The code implementation of feature engineering in GPT3 paper which comes from [https://arxiv.org/pdf/2005.14165.pdf](https://arxiv.org/pdf/2005.14165.pdf). Based on pyspark, you have to install Java first. Wikipedia and others can serve as the positive class, i.e. high-quality sample. But currently, the code only supports txt, csv, xlsx and xls formats.

In view of the need to support Chinese and English languages under normal circumstances, for this consideration, I chose the tokenizer of Baichuan2-13B-Base as the segmentation tool.

Note that sentences are separated by `<unk>`.
