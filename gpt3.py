import os
import argparse
import warnings
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

conf = SparkConf().setAppName("feature engineering").setMaster("local")
sc = SparkContext(conf=conf)

spark = SparkSession.builder.getOrCreate()

tokenizer = Tokenizer(inputCol="sequence", outputCol="tokens")


def readFile(path: str):
    tokens = None
    if os.path.exists(path):
        lines = []
        file = sc.textFile(path)
        for line in file.collect():
            lines.append(tuple([line]))

        tokens = tokenizer.transform(spark.createDataFrame(lines, ["sequence"]))

    return lines, tokens


def runHashingTF(numFeatures, tokens):
    hashingTF = HashingTF(inputCol="tokens", outputCol="features")
    if numFeatures:
        # default to 262144 if you do not specify
        hashingTF.setNumFeatures(numFeatures)

    features = []
    rows = hashingTF.transform(tokens).collect()
    for row in rows:
        features.append(list(row.features.toArray()))

    return np.array(features)


def getLabels(c_array, d_array):
    m_c, _ = c_array.shape
    m_d, _ = d_array.shape
    labels = [1] * m_c + [0] * m_d
    return np.array(labels).reshape(-1, 1)


def runLogisticClassifier(x, y, d_text):
    model = LogisticRegression(penalty='l2', C=10.0, max_iter=300)
    model.fit(x, y)
    proba = model.predict_proba(d_text)
    return proba


def getFilteredIndex(proba, alpha):
    index = []
    for i in range(proba.shape[0]):
        # The probability of negative classes satisfying the Pareto distribution.
        if 1 - proba[i, 1] < np.random.pareto(alpha):
            index.append(i)

    return index


def writeFile(lines: list, index: list, save_path: str):
    with open(save_path, mode='w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            if i in index:
                f.write(line[0])
                f.write('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--c_path", type=str, help="The path of the clean text.")
    parser.add_argument("--d_path", type=str, help="The path of the dirty text.")
    parser.add_argument("--numFeatures", type=int, help="Using hash function to map the maximum number of features "
                                                        "required for index mapping.")
    parser.add_argument("--is_pca", type=bool, default=False, help="Is pca used to reduce the dimensionality of "
                                                                   "features")
    parser.add_argument("--n_components", type=int, default=256,
                        help="Number of components to keep. Only is_pca is ture, it needs to be set.")
    parser.add_argument("--alpha", default=2, type=float, help="Shape of the distribution. Must be positive.")
    parser.add_argument("--save_path", type=str, help="The path for saving filtered text.")

    args = parser.parse_args()

    _, c_tokens = readFile(args.c_path)
    lines, d_tokens = readFile(args.d_path)

    c_array = runHashingTF(args.numFeatures, c_tokens)
    d_array = runHashingTF(args.numFeatures, d_tokens)

    features = np.vstack((c_array, d_array))
    labels = getLabels(c_array, d_array)

    print(f"---LogisticRegression classifier input shape: {features.shape}---")
    print(f"---LogisticRegression classifier label shape: {labels.shape}---")

    if args.is_pca:
        pca = PCA(n_components=args.n_components)
        features = pca.fit_transform(features)
        print(f"---cumulative proportion of variance of the n components: {sum(pca.explained_variance_ratio_)}---")

        d_array = pca.transform(d_array)
        print(f"---cumulative proportion of variance of the n components: {sum(pca.explained_variance_ratio_)}---")

    proba = runLogisticClassifier(features, labels, d_array)
    index = getFilteredIndex(proba, alpha=args.alpha)
    writeFile(lines, index, args.save_path)