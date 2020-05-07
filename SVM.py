import numpy as np
from sklearn.svm import LinearSVC

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
    return np.array(x), np.array(label_data)

def compute_accuracy(predicted_y, test_y):
    matchs= np.equal(predicted_y, test_y)
    accuracy= np.sum(matchs.astype(float)/ test_y.size)
    return accuracy

def classifying_with_linear_SVMs():
    train_x, train_y= load_data("./train_tf_idf.txt")
    classifier= LinearSVC(
        C=10.0,
        tol=0.001,
        verbose= True
    )

    classifier.fit(train_x,train_y)
    test_x, test_y= load_data('./test_tf_idf.txt')
    predicted_y= classifier.predict(test_x)
    accuracy= compute_accuracy(predicted_y, test_y)
    print("accuracy: {}".format(accuracy))
    
def main(): 
    classifying_with_linear_SVMs()

if __name__=="__main__":
    main()