import numpy as np
from sklearn.decomposition import PCA

file_a = "eval_models/imn128_mocov2/reps3.npz"
file_b = "eval_models/imn128_simsiam/reps3.npz"


def class_distance(index1, index2, file_1, file_2):
    a_reps = np.load(file_a)
    a_rep, a_labels = a_reps["arr_0"], a_reps["arr_1"]
    b_reps = np.load(file_b)
    b_rep, b_labels = b_reps["arr_0"], b_reps["arr_2"]
    

    a_rep1 = a_rep[index1]
    a_rep2 = a_rep[index2]

    distance1 = np.linalg.norm(a_rep1  - a_rep2, axis=1)

    pca_rep_b = PCA(n_components=128).fit(b_rep)
    new_rep_b = pca_rep_b.transform(b_rep)

    b_rep1 = new_rep_b[index1]
    b_rep2 = new_rep_b[index2]


    distance2 = np.linalg.norm(b_rep1 - b_rep2, axis=1)
    print(distance1)
    print(distance2)
    pass


if __name__ == "__main__":
    class_distance(np.arange(0, 20, 1, dtype=int), np.arange(25, 45, 1, dtype=int), file_a, file_b)