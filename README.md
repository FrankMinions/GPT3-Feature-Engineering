# GPT3-Feature-Engineering

Part of the code implementation of feature engineering in GPT3 paper comes from https://arxiv.org/pdf/2005.14165.pdf based on pyspark, and you have to install Java first. Wikipedia and others can serve as the positive class, i.e. high-quality sample. But currently, the code only supports txt, csv, xlsx and xls formats.

In view of the need to support Chinese and English languages under normal circumstances, for this consideration, I chose the tokenizer of Baichuan2-13B-Base as the segmentation tool.

Note that sentences are separated by `<unk>`.
