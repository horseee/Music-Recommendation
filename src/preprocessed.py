import FileUtil

FileIO = FileUtil.DataIO()
UserList = FileIO.GetUserDict()
SongDict = FileIO.GetSongToIndex()

# change the user/song ID to usersong Index
g = open('data/processed_data.dat', 'w') 
f = open('data/train_triplets.txt', 'r')
for line in f:
	user, song, times = line.strip().split('\t')
	g.write(str(UserList[user]) + " " + str(SongDict[song]) + " " + times + "\n")
f.close()
g.close()

# calucate the statistic data for each user
f = open("data/processed_data.dat", 'r')
song_total_count = dict()
song_count = dict()
max_count = dict()
min_count = dict()
for line in f:
	user, song, times = line.strip('\n').split(' ')
	if user in song_count.keys():
		song_count[user] += 1
		song_total_count[user] += int(times)
		if int(times) > max_count[user]:
			max_count[user] = int(times)
		if int(times) < min_count[user]:
			min_count[user] = int(times)
	else:
		song_count[user] = 1
		song_total_count[user] = int(times)
		max_count[user] = int(times)
		min_count[user] = int(times)

f = open('data/processed_mean_data.dat', 'w')
for user in song_count.keys():
	mean_value = song_total_count[user]/song_count[user]
	f.write("%s %6f %d %d %d\n" %(user, mean_value, max_count[user], min_count[user], song_total_count[user]))
