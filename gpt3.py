import csv
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import List
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import Tokenizer, HashingTF
from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

conf = SparkConf().setAppName("feature engineering").setMaster("local[*]")

# Driver heap memory size. Driver is the main control process responsible for creating context,
# submitting jobs, converting jobs into tasks, and coordinating task execution between executors.
conf.set("spark.driver.memory", '4G')

# Executor heap memory. The executor is mainly responsible for executing specific calculation tasks
# and returning the results to the driver.
conf.set("spark.executor.memory", '6G')

sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
tokenizer = Tokenizer(inputCol="sequence", outputCol="tokens")


def readFile(file_path: str, file_format: str, is_head: bool):
    df = None
    if file_format == 'xlsx':
        df = pd.read_excel(file_path, header=None).values.tolist()
    if file_format == 'csv':
        df = pd.read_csv(file_path, header=None).values.tolist()
    if file_format == 'txt':
        df = sc.textFile(file_path)
        
    if is_head:
        return df[1:], df[0]
    else:
        return df, None


def getTokens(df, file_format: str):
    lines = []
    if file_format == 'txt':
        for line in df.collect():
            lines.append(tuple([line]))
    else:
        for line in df:
            # Multiple sentences marked with token <sep>.
            lines.append(tuple(["<sep>".join(line)]))

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
    labels = [1] * c_array.shape[0] + [0] * d_array.shape[0]
    return np.array(labels).reshape(-1, 1)


def runLogisticClassifier(x, y, d_text, c, max_iter):
    model = LogisticRegression(penalty='l2', C=c, max_iter=max_iter)
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


def writeFile(lines: List, index: List, file_format: str, is_head: bool, header: List, save_path: str):
    with open(save_path, mode='w', encoding='utf-8', errors='ignore') as f:
        # txt format file writing method
        if file_format == 'txt':
            for i, line in enumerate(lines):
                if i in index:
                    f.write(line[0])
                    f.write('\n')

        # csv format file writing method
        if file_format == 'csv':
            csv_writer = csv.writer(f)
            if is_head:
                csv_writer.writerow(header)
            for i, line in enumerate(lines):
                if i in index:
                    csv_writer.writerow(line[0].split('<sep>'))


def writeExcel(lines: List, index: List, is_head: bool, header: List, save_path: str):
    data = []
    for i, line in enumerate(lines):
        if i in index:
            data.append(line[0].split('<sep>'))

    if is_head:
        df = pd.DataFrame(data, columns=header)
    else:
        df = pd.DataFrame(data, columns=None)

    writer = pd.ExcelWriter(save_path, engine='openpyxl')
    df.to_excel(writer, sheet_name='sheet1', index=False)
    writer.save()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--c_path", type=str, default="wikipedia.txt", help="The path of the clean text.")
    parser.add_argument("--d_path", type=str, help="The path of the dirty text.")
    parser.add_argument("--is_head", type=bool, default=False, help="Does the table contain a header.")
    parser.add_argument("--numFeatures", type=int, help="Using hash function to map the maximum number of features "
                                                        "required for index mapping.")
    parser.add_argument("--is_pca", type=bool, default=False, help="Is pca used to reduce the dimensionality of "
                                                                   "features")
    parser.add_argument("--n_components", type=int, default=512,
                        help="Number of components to keep. Only is_pca is ture, it needs to be set.")
    parser.add_argument("--c", type=float, default=1.0, help="Regularization strength of LogisticRegression.")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum number of iterations taken for LogisticRegression to converge.")
    parser.add_argument("--alpha", default=9, type=float, help="Shape of the distribution. Must be positive.")
    parser.add_argument("--save_path", type=str, help="The path for saving filtered text.")
    args = parser.parse_args()

    if args.d_path.endswith('xlsx'):
        file_format = 'xlsx'
    elif args.d_path.endswith('csv'):
        file_format = 'csv'
    elif args.d_path.endswith('txt'):
        file_format = 'txt'
    else:
        raise "Unsupported file format!"

    cdf, _ = readFile(args.c_path, 'txt', False)
    ddf, header = readFile(args.d_path, file_format, args.is_head)

    _, c_tokens = getTokens(cdf, 'txt')
    lines, d_tokens = getTokens(ddf, file_format)

    c_array = runHashingTF(args.numFeatures, c_tokens)
    d_array = runHashingTF(args.numFeatures, d_tokens)

    features = np.vstack((c_array, d_array))
    labels = getLabels(c_array, d_array)

    print(f"The input and output shapes of a logistic regression classifier are {features.shape} and {labels.shape}.")

    if args.is_pca:
        pca = PCA(n_components=args.n_components)
        features = pca.fit_transform(features)
        d_array = pca.transform(d_array)

        explained_variance_ratio = sum(pca.explained_variance_ratio_)
        print(f"The cumulative proportions of the top n principal components of high-quality data is {explained_variance_ratio}.")

    proba = runLogisticClassifier(features, labels, d_array, args.c, args.max_iter)
    index = getFilteredIndex(proba, alpha=args.alpha)
    if file_format != 'xlsx':
        writeFile(lines, index, file_format, args.is_head, header, args.save_path)
    else:
        writeExcel(lines, index, args.is_head, header, args.save_path)
