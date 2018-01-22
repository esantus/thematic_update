# LOADING LIBRARIES
DEBUG = False

import sys, codecs, os, gzip, re, pickle

from composes.semantic_space.space import Space
from composes.similarity.cos import CosSimilarity
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils import io_utils

from scipy import spatial
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools as itertools

from collections import defaultdict


def load_fillers(fillers_path):
    """
    Load fillers as a dictionary with w1-rel as key and a sorted list of (filler, lmi)
    as values.
    """

    fillers = {}

    with open(fillers_path, "r") as fillers_f:
        for line in fillers_f:
            field = line.strip().split("\t")
            field[1] = field[1].lower()

            if field[0]+"-"+field[1] not in fillers:
                fillers[field[0]+"-"+field[1]] = []
            fillers[field[0]+"-"+field[1]].append([field[2], float(field[3])])

    for key in fillers:
        fillers[key] = sorted(fillers[key], key = lambda x:x[1], reverse=True)

    if DEBUG == True:
        for key in fillers:
            print(key, " = ", fillers[key])

    return fillers


def load_matrix(matrix_prefix):
    """
    Load the space from either a single pkl file or numerous files
    :param matrix_prefix:
    :param matrix:
    :return:
    """
    print("Loading the matrix...")

    # Check whether there is a single pickle file for the Space object
    if os.path.isfile(matrix_prefix + '.pkl'):
        return io_utils.load(matrix_prefix + '.pkl')

    # Load the multiple files: npz for the matrix and pkl for the other data members of Space
    with np.load(matrix_prefix + 'cooc.npz') as loader:
        coo = coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])

    cooccurrence_matrix = SparseMatrix(csr_matrix(coo))

    with open(matrix_prefix + '_row2id.pkl', 'rb') as f_in:
        row2id = pickle.load(f_in)

    with open(matrix_prefix + '_id2row.pkl', 'rb') as f_in:
        id2row = pickle.load(f_in)

    with open(matrix_prefix + '_column2id.pkl', 'rb') as f_in:
        column2id = pickle.load(f_in)

    with open(matrix_prefix + '_id2column.pkl', 'rb') as f_in:
        id2column = pickle.load(f_in)

    return Space(cooccurrence_matrix, id2row, id2column, row2id=row2id, column2id=column2id)


def get_dataset(dataset_fname):

    dataset = {}

    with open(dataset_fname, "r") as dataset_f:
        for line in dataset_f:
            field = line.strip().split("\t")

            if field[0] not in dataset:
                dataset[field[0]] = []

            dataset[field[0]].append(field[1:])

    return dataset


def get_proto(target, relation, fillers, N, dsm):
    proto = []

    key = target+"-"+relation

    if DEBUG == True:
        print("Prototype for %s with the relation %s..." % (target, relation))

    if key in fillers:
        filler_vecs = [filler for filler, lmi in fillers[key]][:N]

        if DEBUG == True:
            print("Here are the fillers for %s: %s" % (key, filler_vecs))

        if len(filler_vecs) < N:
            print("WARNING: not enough fillers for %s. Only %d found." % (key, len(filler_vecs)))
            if len(filler_vecs) == 0:
                print "Returning proto=False, as only one filler was found."
                return False

        for i in range(0, len(filler_vecs)):
            if DEBUG == True:
                print("Getting vector of filler %s." % filler_vecs[i])

            temp = get_vec(filler_vecs[i], dsm)

            if temp != False:
                if proto == []:
                    proto = temp
                else:
                    proto = proto + temp
    else:
        print("WARNING: %s does not exist in the filler database." % key)
	print("Returning proto = False")
	return False

    return proto


def get_vec(target, dsm):

    target2index = {w: i for i, w in enumerate(dsm.id2row)}
    index = target2index.get(target.lower(), -1)

    if index != -1:
        vector = dsm.cooccurrence_matrix[index, :]
    else:
        if DEBUG == True:
            print "WARNING: ", target, " does not exist in the matrix! (GET_VEC)"
        return False

    return vector


def calculate_dataset(proto_mat, dataset, relations, dsm):

    lines, gold, vector_cosine_sum, vector_cosine_prod = [], [], [], []

    for entry in dataset:
        proto_key1 = entry+"-"+relations[0]
        for candidate in dataset[entry]:
            proto_key2 = candidate[0]+"-"+relations[1]

            if proto_key1 in proto_mat and proto_key2 in proto_mat:
                candidate_filler_vec = get_vec(candidate[1], dsm)

                if DEBUG == True:
                    print("proto_mat[key1]:", type(proto_mat[proto_key1]), proto_mat[proto_key1].shape)
                    print("proto_mat[key1]:", type(proto_mat[proto_key2]), proto_mat[proto_key1].shape)
                    print("candidate_filler_vec:", type(candidate_filler_vec), candidate_filler_vec.shape)

                if candidate_filler_vec != False:
                    vc_sum = cosine(proto_mat[proto_key1]+proto_mat[proto_key2], candidate_filler_vec)
                    vc_prod = cosine(proto_mat[proto_key1].multiply(proto_mat[proto_key2]), candidate_filler_vec)
                elif DEBUG == True:
                    print("Vector for candidate filler %s, from the dataset, does not exist in the matrix!" % (dataset[entry][1]))
                    continue

                lines.append(entry+"\t"+"\t".join(candidate))
                gold.append(candidate[2:])
                vector_cosine_sum.append(vc_sum)
                vector_cosine_prod.append(vc_prod)
            else:
                print("WARNING: Either proto1 or proto2 were not in proto_mat!")

    return lines, gold, vector_cosine_sum, vector_cosine_prod


def cosine(v1, v2):
    if v1.norm() == 0 or v2.norm() == 0:
        return 0.0

    return v1.multiply(v2).sum() / np.double(v1.norm() * v2.norm())





















