import numpy as np

class DataIO:
	def __init__(self):
		pass

	def GetSongAndUserCount(self):
		f = open('data/users.txt', 'r')
		user_count = 0
		for line in f:
			user_count += 1
		f.close()
		f = open('data/songs.txt', 'r')
		song_count = 0
		for line in f:
			song_count += 1
		f.close()
		return [user_count, song_count]


	def GetSongToCount(self, song_to_index):
		print("Calculate Songs' Count ...")
		f = open('data/train_triplets.txt', 'r')
		song_to_count = dict()
		for line in f:
			_, song, _ = line.strip().split('\t')
			# print(song_to_index[song])
			if int(song_to_index[song]) in song_to_count:
				song_to_count[int(song_to_index[song])] += 1
			else:
				song_to_count[int(song_to_index[song])] = 1
		f.close()
		# print(song_to_count[91177])
		# songs_ordered = sorted(song_to_count.keys(), key = lambda s: song_to_count[s], reverse = True)
		return song_to_count

	def GetUserToSong(self):
		print("Get User listening history ...")
		f = open('data/train_triplets.txt', 'r')
		user_to_songs = dict()
		for line in f:
			user, song, _ = line.strip().split('\t')
			if user in user_to_songs:
				user_to_songs[user].add(song)
			else:
				user_to_songs[user] = set([song])
		f.close()
		return user_to_songs

	def GetSongToUser(self, user_to_index, song_to_index):
		print("Get Song item and its listener")
		f = open('data/train_triplets.txt', 'r')
		song_to_users =np.zeros((len(song_to_index), len(user_to_index)))
		for line in f:
			user, song, times = line.strip().split('\t')
			song_to_users[int(song_to_index[song])-1][int(user_to_index[user])] = int(times)
		return song_to_users

	def GetUserDict(self):
		print("Get User Index ...")
		f= open('data/users.txt', 'r')
		# canonical_users = list(map(lambda line: line.strip(), f.readlines()))
		user_num = 0
		canonical_users = dict()
		for line in f:
			canonical_users[line.strip()] = user_num
			user_num += 1
		f.close()
		return canonical_users

	def GetUserIndex(self, canonical_users):
		print("Get User ID ...")
		user_to_index = dict()
		for i in range(len(canonical_users)):
			user_to_index[canonical_users[i]] = i
		return user_to_index

	def GetSongToIndex(self):
		print("Get Song Index ...")
		f= open('data/songs.txt', 'r')
		song_to_index = dict(map(lambda line:line.strip().split(' '), f.readlines()))
		f.close()
		return song_to_index