""" simple FA code using EM """

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
import logging
from time import time
from sklearn import decomposition


# ============================
# DPEM code for Factor Analysis
# ============================
#
# Mijung wrote this part, while below "Faces dataset decompositions" part is written by Vlad Niculae, Alexandre Gramfort as noted.

def EM_FA(XX, n_compo, W_init, Psi_init_inv):
    # XX is data second moment matrix
    # n_compo is the number of components

    # (1) M-step given
    n_features, n_features = XX.shape
    G = np.linalg.inv(np.dot(np.dot(W_init.transpose(), Psi_init_inv), W_init) + np.eye(n_compo))

    Psi_inv_W_G_trp = np.dot(np.dot(Psi_init_inv, W_init), G.transpose())
    W_first_term = np.dot(XX, Psi_inv_W_G_trp)
    W_second_term = np.linalg.inv(G + np.dot(np.dot(Psi_inv_W_G_trp.transpose(), XX), Psi_inv_W_G_trp))
    W = np.dot(W_first_term, W_second_term)

    Psi = np.diag(XX - np.dot(np.dot(W_init, Psi_inv_W_G_trp.transpose()),XX))

    return W, Psi


# ============================
# Faces dataset decompositions
# ============================
#
# This example applies to :ref:`olivetti_faces` different unsupervised
# matrix decomposition (dimension reduction) methods from the module
# :py:mod:`sklearn.decomposition` (see the documentation chapter
# :ref:`decompositions`) .
#
# """
# print(__doc__)
#
# # Authors: Vlad Niculae, Alexandre Gramfort
# # License: BSD 3 clause
#
#
#
# from sklearn.cluster import MiniBatchKMeans
#
###############################################################################
def plot_gallery(title, images, n_col=10, n_row=1):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)

#
# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_components = 10
image_shape = (64, 64)
rng = RandomState(0)
#
# ###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=rng)
faces = dataset.data

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)


print("Preprocess the face data such that max norm of each image is less than 1")
X = faces_centered
max_norm = max(np.linalg.norm(X, axis=0))
X = X/max_norm

XX = np.dot(X.T, X)/float(n_samples)

max_iter = 20

epsilon = 1
delta = 0.0001

print("Private FA is being processed for epsilon= %f" % epsilon)

# noise addition
sensitivity = 2/float(n_samples)
c2 = 2*np.log(1.25/delta)
nsv = c2*(sensitivity**2)/(epsilon**2)
# how_many = n_features*(n_features+1)*0.5

nse_mat = np.random.normal(0,nsv,[n_features,n_features])
upper_nse_mat = np.triu(nse_mat, 0)

print("noise generation for perturbing XX")

for i in range(n_features):
    for j in range(i, n_features):
        upper_nse_mat[j][i] = upper_nse_mat[i][j]

nse = upper_nse_mat

XX_tile = XX + nse

print("once we add noise to XX, we do svd to make sure the resulting matrix is still positive definite")
print("this will take a while, because the image size is about 4000")
# to ensure the matrix is positive definite
w, v = np.linalg.eig(XX_tile)
# remember: XX_tile = np.dot(v, np.dot(np.diag(w), v.transpose()))
neg_idx = np.nonzero(w<0)
w[neg_idx] = 0.0001

XX_perturbed = np.dot(v, np.dot(np.diag(w), v.transpose()))

print("now perturbed XX is positive definite!")

# W_init = np.random.normal(0, 1, [n_features, n_components])
W_init = v[:, 0:n_components]
Psi_init_inv = np.diag(0.1*np.ones(n_features))

for i in range(0, max_iter):
    print("%d th iteration of EM" %i)
    W_new, Psi_new = EM_FA(XX_perturbed, n_components, W_init, Psi_init_inv)
    W_init = W_new
    Psi_init_inv = np.diag(1/Psi_new)


plot_gallery('privateFA with %f' % (epsilon), W_new.transpose()[:n_components])
plt.savefig('privFA_epsilon=%f.pdf' % epsilon)
plt.show()
