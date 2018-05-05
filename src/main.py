from __future__ import print_function
import os
import logging

import numpy as np
import numpy as np
import scipy.sparse
from scipy.sparse import lil_matrix
import tensorflow as tf
import time

from numpy.random import RandomState
from recommend.pmf import PMF
from recommend.utils.evaluation import RMSE

import FileUtil
FileIO = FileUtil.DataIO()

"""
Read Data Infomation
"""
user_to_index = FileIO.GetUserDict()
song_to_index = FileIO.GetSongToIndex()

print("User : %d, Song : %d" % (len(user_to_index), len(song_to_index)))

song_to_count = FileIO.GetSongToCount(song_to_index)
column_to_song_index = dict()
song_to_column_index = dict()
song_ptr = 0
for song in iter(song_to_index):
    if int(song_to_index[song]) in song_to_count:
        temp_song = int(song_to_index[song])
        column_to_song_index[song_ptr] = temp_song
        song_to_column_index[temp_song] = song_ptr
        song_ptr += 1

user_count = len(user_to_index)
song_count = len(song_to_column_index)
song_origin_count = len(song_to_index)

"""
Read User-Song Train triplets
"""

# Read the mean, max, min for each user
f = open("data/processed_mean_data.dat", 'r')
user_mean = dict()
user_max = dict()
user_min = dict()

for line in f:
    user, mean, max_value, min_value, total_value = line.strip('\n').split(' ')
    user_mean[int(user)] = float(mean)
    user_max[int(user)] = int(max_value)
    user_min[int(user)] = int(min_value)

user_to_song =lil_matrix((user_count, song_count), dtype=np.float32)
print("Get Song item and its listener ...\nTotal User = %d, Total Song(at least 1 user have listened) = %d" % (user_count, song_count))
f = open('data/processed_data.dat', 'r')

indices = []
data = []
ratings = []

for line in f:
    user, song, times = line.strip('\n').split(' ')
    times = int(times)
    user = int(user)
    song = int(song)
    
    # store data for PMF
    Max_value = user_max[user]
    Min_value = user_min[user]
    Mean_value = user_mean[user]
    
    if Max_value - Min_value < 3:
         times = 3 + times - Min_value
    elif times == Mean_value:
         times = 3
    elif times > Mean_value:
         times = (times - Mean_value) / (Max_value - Mean_value) * 2 + 3
    else:
         times = (times - Min_value) / (Mean_value - Min_value) * 2 + 1
    ratings.append([user, int(song), float(times)])
    
    # store data for ALS
    indices.append([int(user), song_to_column_index[int(song)]])
    times = 1
    data.append(times)
    user_to_song[int(user), song_to_column_index[int(song)]] = times
    
print("Finish Getting All Song and its record")

train = np.array(ratings)
ArrayIndices = np.array(indices, dtype = np.int64)
ArrayData = np.array(data, dtype =np.float32)

"""
Popularity Rank
"""
hot_count = FileIO.GetSongToCount(song_to_index)
hot_rank = sorted(hot_count.items(), key=lambda d: d[1], reverse = True)
TopSong = []
for i in range(300):
    TopSong.append(hot_rank[i][0])

"""
ALS for Latent Factor
"""
# initialize x, y, k and lambda
k = 500
reg_lambda = 12
Iteration_time = 300
Iter = 0

Input_X = np.mat(np.ones((k, user_count), dtype=np.float32))
Input_Y = np.mat(np.ones((k, song_count), dtype=np.float32))

# Input_X = np.fromfile("data/InputX.bin",dtype = np.float32)
# Input_Y = np.fromfile("data/InputY.bin",dtype = np.float32)
# Input_X = np.mat(Input_X.reshape(k, user_count))
# Input_Y = np.mat(Input_Y.reshape(k, song_count))

X = tf.placeholder(dtype=tf.float32,shape=Input_X.shape,name="X")
Y = tf.placeholder(dtype=tf.float32,shape=Input_Y.shape,name="Y")
# User_To_Song_Sparse = tf.sparse_placeholder(dtype=tf.float32, shape=user_to_song.shape,name="User_To_Song_Sparse")
MatrixIndice = tf.placeholder(dtype=tf.int64, shape=ArrayIndices.shape, name="MatrixIndice")
MatrixData = tf.placeholder(dtype=tf.float32, shape=ArrayData.shape, name="MatrixData")
TF_Song_Count = tf.constant(dtype=tf.int64, value = song_count, name = "TF_Song_Count")
TF_User_Count = tf.constant(dtype=tf.int64, value = user_count, name = "TF_User_Count")

User_To_Song_Sparse = tf.SparseTensor(indices = MatrixIndice, values = MatrixData, dense_shape=[TF_User_Count, TF_Song_Count])

y = tf.transpose(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(User_To_Song_Sparse), tf.transpose(tf.matmul(tf.matrix_inverse(tf.matmul(X,tf.transpose(X,perm=(1,0))) + np.mat(np.eye(k) * reg_lambda)),X))))
x = tf.transpose(tf.sparse_tensor_dense_matmul(User_To_Song_Sparse, tf.transpose(tf.matmul(tf.matrix_inverse(tf.matmul(y,tf.transpose(y,perm=(1,0))) + np.mat(np.eye(k) * reg_lambda)),y))))
#x = tf.matmul(tf.matrix_inverse(tf.matmul(y,tf.transpose(y,perm=(1,0))) + np.mat(np.eye(k) * reg_lambda)),y)
#x = tf.matmul(tf.matrix_inverse(tf.matmul(Y,tf.transpose(Y,perm=(1,0))) + np.mat(np.eye(k) * reg_lambda)),Y), tf.transpose(User_To_Song_Sparse,perm=(1,0)), b_is_sparse=True)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.3) # change the fraction if needed
#with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("Training ALS model ...")
    print("Run...")
    c = time.time()
    for i in range(Iteration_time):
        Input_Y, Input_X = sess.run([y, x], {X: Input_X, Y:Input_Y, MatrixIndice:ArrayIndices, MatrixData:ArrayData})
        #Input_X = sess.run(x, {Y:Input_Y, User_to_Song: user_to_song})
        #Input_Y = sess.run(y, {X:Input_X, User_to_Song: user_to_song})
        print("Iteration: %d/%d" % (i+1, Iteration_time))
    print("Time cost: %f" % (time.time()-c))



"""
Probabilistic Matrix Factorization
"""
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
rand_state = RandomState(0)
n_feature = 10
eval_iters = 20

print("training PMF model ...")
pmf = PMF(n_user=user_count, n_item=song_origin_count+1, n_feature=n_feature, epsilon=15., converge = 1e-8, momentum=0.4, max_rating=5.0, min_rating=0., seed=100, reg = 0.01)
pmf.fit(train, n_iters=eval_iters)



"""
Mix 2 model and write result to file
"""
print("Begin Writing result to file ...")
f= open('submission.txt', 'w')
userGroup = 0
GroupSize = 1000
GroupNumber = 200
print("Finished: %d / %d" % (0, GroupNumber * GroupSize))
while (userGroup < GroupNumber):
    
    start_index = userGroup * GroupSize 
    end_index = (userGroup + 1) * GroupSize 
    
    # rating for PFM
    pmf_pred = np.dot(pmf.user_features_[start_index:end_index, :], (pmf.item_features_[:, :].T)) + pmf.mean_rating_
    
    # rating for ALS
    P = tf.placeholder(dtype=tf.float32, shape=(k, GroupSize), name="P")
    product = tf.matmul(tf.transpose(P), Y)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))
    sess = tf.Session()
    resultMatrix = sess.run(product, {P:Input_X[:, start_index:end_index], Y:Input_Y})
    
    for user in range(0, GroupSize):
    
        # popularity
        rated_item = user_to_song[user + start_index].rows[0]
        UserSongListenTimes = 0
        for rate_item in rated_item:
            if rate_item in hot_count.keys():
                UserSongListenTimes += hot_count[rate_item]
        UserAverage = UserSongListenTimes / len(rated_item)
        
        # PMF & ALS
        indicePMF = np.argsort(pmf_pred[user, :])
        indiceALS = np.argsort(resultMatrix[user, :])
        DivMaxPMF = 0.005 / pmf_pred[user, indicePMF[song_origin_count - 1]]
        DivMaxALS = 1 / resultMatrix[user, indiceALS[song_count - 1]]
        ResultDict = dict()
        for i in range(300):
            # PMF
            song = indicePMF[song_origin_count - 1 - i]
            if song in ResultDict.keys():
                ResultDict[song] += pmf_pred[user, song] * DivMaxPMF
            else:
                ResultDict[song] = pmf_pred[user, song] * DivMaxPMF
            
            # ALS 
            song = column_to_song_index[indiceALS[song_count - 1 - i]]
            if song in ResultDict.keys():
                ResultDict[song] += resultMatrix[user, indiceALS[song_count - 1 - i]]
            else:
                ResultDict[song] = resultMatrix[user, indiceALS[song_count - 1 - i]]
        
        # sorting
        ResultIndice = sorted(ResultDict.items(), key=lambda d: d[1], reverse = True) 

        i = 0
        temp_count = 0
        recommend = []
        while temp_count < 100:
            if ResultIndice[i][0] not in song_to_column_index.keys():
                i += 1
                continue
            if song_to_column_index[ResultIndice[i][0]] in rated_item:
                i += 1
                continue
                
            # print("user: %d, rank: %d, item: %d" % (start_index + user, temp_count + 1, ResultIndice[i][0]))
            recommend.append(str(ResultIndice[i][0]))
            i += 1
            temp_count += 1
        
        f.write(' '.join(recommend) + '\n')
    
    userGroup += 1
    print("Finished: %d / %d" % (userGroup * GroupSize, GroupNumber * GroupSize))
    
f.close()


"""
Test for the hidden triplets
"""
print("Reading Test Data...")
f = open('data/result_data.dat', 'r')
result_user_to_song = dict()
for line in f:
    user, song, _ = line.strip('\n').split(' ')
    if int(user) in result_user_to_song:
        result_user_to_song[int(user)].add(song)
    else:
        result_user_to_song[int(user)] = set([song])
f.close()

f = open('submission.txt', 'r')
print("Begin Testing ...")
predicted_user_to_song = dict()
ptr = 0
for line in f:
    predicted_user_to_song[ptr] = list()
    predicted_user_to_song[ptr] = line.strip().split(' ')
    ptr += 1
f.close()

score = 0
canonical_sub_users = []
for i in range(GroupNumber * GroupSize):
    canonical_sub_users.append(i)
    
user_ptr = 0
for user in canonical_sub_users:
    pre_score = score
    pre_song = predicted_user_to_song[user]
    res_song = result_user_to_song[user]
    correct_song_number = 0
    print("user : %d"  % user_ptr)
    for i in range(len(pre_song)):
        if pre_song[i] in res_song:
            # print("rank : %d song: %s" % (i+1, pre_song[i]))
            correct_song_number += 1
            rel = 1
        else:
            rel = 0
        min_num = min(len(pre_song), len(res_song))
        score += rel * correct_song_number / (i+1) / min_num

    print("%d / %d"%(correct_song_number, len(res_song) ))
    # print("%5f" % (score - pre_score))
    user_ptr += 1
print("Accurate rate = %.8f"% (score / (GroupNumber * GroupSize)))







