import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import pickle

np.set_printoptions(precision=4, suppress=True)

train_filenames = glob.glob('/home/mfe/code/dc2g/training_data/driveways_bing_iros19/full_semantic/train/world*.png')
# train_filenames = glob.glob('/home/mfe/code/dc2g/training_data/driveways_bing_iros19/full_semantic/train/worldn000*.png')
# train_filenames = glob.glob('/home/mfe/code/dc2g/training_data/driveways_bing_iros19/full_semantic/train/worldn001*.png')

###############################
# Feature extraction
##############################
feature_pts = []
descriptors = []
for filename in train_filenames:
	img = cv.imread(filename,0)
	img = cv.resize(img, (256,256), interpolation=cv.INTER_NEAREST)
	grid_cell_size = 20
	for i in np.arange(grid_cell_size/2., img.shape[0], grid_cell_size, dtype=int):
		for j in np.arange(grid_cell_size/2., img.shape[1], grid_cell_size, dtype=int):
			region = img[i:i+grid_cell_size,j:j+grid_cell_size]
			hist,bins = np.histogram(region.ravel(),256,[0,256])
			descr = hist
			if region.shape == (grid_cell_size, grid_cell_size):
				descriptors.append(descr)
descriptors = np.array(descriptors)

###############################
# Bag of Words learning
##############################
from sklearn.cluster import KMeans
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(descriptors)
# print(kmeans.cluster_centers_[:2,:])

bow_histogram_train = []
for filename in train_filenames:
	img = cv.imread(filename,0)
	img = cv.resize(img, (256,256), interpolation=cv.INTER_NEAREST)
	grid_cell_size = 20
	bow_histogram = np.zeros((num_clusters))
	for i in np.arange(grid_cell_size/2., img.shape[0], grid_cell_size, dtype=int):
		for j in np.arange(grid_cell_size/2., img.shape[1], grid_cell_size, dtype=int):
			region = img[i:i+grid_cell_size,j:j+grid_cell_size]
			hist,bins = np.histogram(region.ravel(),256,[0,256])
			descr = hist
			if region.shape == (grid_cell_size, grid_cell_size):
				# descriptors.append(descr)
				closest_word = kmeans.predict([hist])
				word_score = kmeans.predict([hist])
				bow_histogram[closest_word] += 1
	bow_histogram = bow_histogram / np.linalg.norm(bow_histogram)
	bow_histogram_train.append(bow_histogram)
# plt.bar(np.arange(num_clusters), bow_histogram)

###############################
# Feature extraction on test images
##############################
imgs = []
dists = []
test_filenames = glob.glob('/home/mfe/code/dc2g/training_data/driveways_bing_iros19/full_semantic/test/world*.png')
for filename in test_filenames:
	img = cv.imread(filename, 0)
	img = cv.resize(img, (256,256), interpolation=cv.INTER_NEAREST)
	feature_pts = []
	descriptors = []
	grid_cell_size = 20
	bow_histogram = np.zeros((num_clusters))
	for i in np.arange(grid_cell_size/2., img.shape[0], grid_cell_size, dtype=int):
		for j in np.arange(grid_cell_size/2., img.shape[1], grid_cell_size, dtype=int):
			region = img[i:i+grid_cell_size,j:j+grid_cell_size]
			hist,bins = np.histogram(region.ravel(),256,[0,256])
			descr = hist
			if region.shape == (grid_cell_size, grid_cell_size):
				# descriptors.append(descr)
				closest_word = kmeans.predict([hist])
				word_score = descr - closest_word
				# word_score = kmeans.score([hist])
				# print(word_score)
				bow_histogram[closest_word] += 1./np.linalg.norm(word_score)
	bow_histogram = bow_histogram / np.linalg.norm(bow_histogram)
	dist_to_train_img = []
	for bow_histogram_ in bow_histogram_train:
		dist_to_train_img.append(np.linalg.norm(bow_histogram_ - bow_histogram))
	dists.append(min(dist_to_train_img))
	imgs.append(filename)
	print(dist_to_train_img)

ranked_dists = sorted(dists)
most_similar_to_img1 = [img for _,img in sorted(zip(dists,imgs))]
plt.figure('similar')

for i, filename in enumerate(train_filenames):
	ax = plt.subplot(2, len(most_similar_to_img1), i+1)
	img = plt.imread(filename)
	img = cv.resize(img, (256,256), interpolation=cv.INTER_NEAREST)
	ax.imshow(img)
	# ax.set_title(round(ranked_dists[i],2))
	# ax.get_yaxis().set_visible(False)
	ax.set_yticks([])
	ax.set_xticks([])
	ax.set_xlabel(filename.split('/')[-1][8:9])
	if i == 0:
		ax.set_ylabel('Train:', rotation='horizontal', ha='right')

for i, filename in enumerate(most_similar_to_img1):
	ax = plt.subplot(2, len(most_similar_to_img1), i+1+len(most_similar_to_img1))
	img = plt.imread(filename)
	img = cv.resize(img, (256,256), interpolation=cv.INTER_NEAREST)
	ax.imshow(img)
	ax.set_title(round(ranked_dists[i],2))
	# ax.get_yaxis().set_visible(False)
	ax.set_yticks([])
	ax.set_xticks([])
	ax.set_xlabel(filename.split('/')[-1][8:9])
	if i == 0:
		ax.set_ylabel('Test:', rotation='horizontal', ha='right')
plt.show()

similarity_dict = {}
for i, filename in enumerate(train_filenames):
	world_id = filename.split('/')[-1].split('.')[0]
	similarity_dict[world_id] = 0
for i, filename in enumerate(most_similar_to_img1):
	world_id = filename.split('/')[-1].split('.')[0]
	similarity_dict[world_id] = ranked_dists[i]
print(similarity_dict)
with open("image_similarities.p", "wb") as f:
	pickle.dump(similarity_dict, f)

# plt.bar(np.arange(num_clusters), bow_histogram)
# plt.show()

# # kmeans.predict([[0, 0], [4, 4]])


# # plt.hist(img1.ravel(),256,[0,256]); plt.show()

# # color = ('b','g','r')
# # for i,col in enumerate(color):
# # 	histr = cv.calcHist([img1],[i],None,[256],[0,256])
# # 	plt.plot(histr,color = col)
# # 	plt.xlim([0,256])
# # plt.show()
# # Initiate SIFT detector
# sift = cv.xfeatures2d.SIFT_create()
# # sift = cv.xfeatures2d.SURF_create()
# # sift = cv.ORB_create()
# # sift = cv.FastFeatureDetector()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)
# img1_kp = cv.drawKeypoints(img1,kp1,None)
# plt.imshow(img1_kp),plt.show()
# # BFMatcher with default params
# bf = cv.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)
# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.8*n.distance:
#     # if m.distance < 0.75*n.distance:
#         good.append([m])
# # cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# plt.imshow(img3),plt.show()