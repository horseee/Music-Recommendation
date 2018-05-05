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
for i in range(200000):
    canonical_sub_users.append(i)
    
user_ptr = 0
correct_song_number = 0
total_song = 0
for user in canonical_sub_users:
    pre_score = score
    pre_song = predicted_user_to_song[user]
    res_song = result_user_to_song[user]
    print("user : %d"  % user_ptr)
    for i in range(len(pre_song)):
        if pre_song[i] in res_song:
            # print("rank : %d song: %s" % (i+1, pre_song[i]))
            correct_song_number += 1
    total_song += 100

    #print("%d / %d"%(correct_song_number, len(res_song) ))
    #print(score)
    # print("%5f" % (score - pre_score))
    user_ptr += 1
print("Accurate rate = %.8f"% (correct_song_number / total_song))







