"""## Thực hiện KMeans với Sklearn"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

def load_data( path):
    def sparse_to_dense(sparse_r_d, vocab_size):
        # hàm trích suất mảng các giá trị tf-idf
        r_d = [0.0 for i in range(vocab_size)]
        ind_tfidf = sparse_r_d.split()
        for index_tfidf in ind_tfidf:
            index = int(index_tfidf.split(":")[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(path) as f:
        d_lines = f.read().splitlines()
    with open('./words_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())

    # self._data = []
    # self._label_count = defaultdict(int)
    x=[]
    label_data=[]
    for data_id, d in enumerate(d_lines):
        features = d.split('<fff>')
        label, doc_id = int(features[0]), int(features[1])
        # self._label_count[label] += 1
        r_d = sparse_to_dense(sparse_r_d=features[2], vocab_size=vocab_size)
        # r_d là mảng các giá trị của văn bản d
        # self._data.append(Member(r_d=r_d, label=label, doc_id=doc_id))
                
        x.append(r_d)
        label_data.append(label)
    return x, label_data
    # tạo mảng data có các Member đại diện cho 1 văn bản.

def clustering_with_KMeans():
  
    data, label=load_data("./full_tf_idf.txt")
    x = csr_matrix(data)
    print("==================")
    kmeans = KMeans( n_clusters=20, init='random', n_init= 5, tol=1e-3, random_state= 2020).fit(x)
    labels = kmeans.labels_
    print(labels)
  
clustering_with_KMeans()
