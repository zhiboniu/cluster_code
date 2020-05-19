#coding=utf-8
from __future__ import print_function
import os
import sys
import faiss
import numpy as np
import pdb
import networkx as nx
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from tqdm import tqdm

dataf = sys.argv[1]	#待聚类数据列表，格式为imgpath \t group \t feature

#阈值设置部分
IP_out_thread = 0.4	#组内细分、异常值剔除 內积阈值，需要尝试看看
IP_ct_thread = 0.9	#簇间合并阈值，默认0.8
hier_thread = 1.20	#层次聚类阈值，默认1.15
Dup_thread = 0.1	#去重阈值，默认0.2

featdim = 384 #特征长度，默认取128

SAVE_HIER = 0	#保存层次聚类图

ngpu = faiss.get_num_gpus()
print("GPU number:",ngpu)
if ngpu>0:
	USE_GPU = 1
else:
	USE_GPU = 0

with open(dataf,'r') as rf:
	lines = rf.readlines()
print('original image number:',len(lines))

img_group_dict = {}
feat_group_dict = {}
for line in tqdm(lines):
	imgpath,label,feat = line.strip().split('\t')
	feat = map(float,feat.split(' '))
	featdim = len(feat)
	if label in img_group_dict:
		img_group_dict[label] = img_group_dict[label]+[imgpath]
		feat_group_dict[label] = feat_group_dict[label]+[feat]
	else:
		img_group_dict[label] = [imgpath]
		feat_group_dict[label] = [feat]

print('original image group:',len(img_group_dict))
assert len(img_group_dict)==len(feat_group_dict)
#for item in img_group_dict:
#	print(len(img_group_dict[item]))
#	print(len(feat_group_dict[item]))

#一、去除各组数据中异常值
#计算各组内各点间距离，并建立网络图，过滤掉异常点
new_img_dict = {}
new_feat_dict = {}
new_label = 0
countnum = 0
for label in tqdm(feat_group_dict):
	aid_data = np.array(feat_group_dict[label]).astype(np.float32)
	#aid_data = np.random.rand(10,128).astype(np.float32)
	idx = np.arange(aid_data.shape[0])
	#print("original number:",aid_data.shape[0])
	if USE_GPU:
		res = faiss.StandardGpuResources()
	
	d_index = faiss.IndexFlatIP(featdim)   # build the index
	if USE_GPU:
		d_index = faiss.index_cpu_to_gpu(res, 0, d_index)
	id_index = faiss.IndexIDMap(d_index)
	id_index.add_with_ids(aid_data, idx)

	D, I = id_index.search(aid_data, 5)	# sanity check
	D, I = id_index.search(aid_data, 5)	# actual search
	#print(D)
	#print(I)

	edges= set()
	for key,I_vec,D_vec in zip(idx,I,D):
		viewset = set()
		for index,dis in zip(I_vec,D_vec):
			if key==index or dis < IP_out_thread:
				continue
			aid,bid = ((min(key,index), max(key,index)))
 			edges.add(tuple((aid,bid)))
			viewset.add(tuple((aid,bid)))
		#print(viewset)
	G = nx.Graph()
	G.add_edges_from(edges)
	#print("G num of edges:{}".format(G.number_of_edges()))
	#print("G num of nodes:{}".format(G.number_of_nodes()))
	connected_g=[]
	for i in nx.connected_components(G):
		connected_g.append(i)
	#print(connected_g)
	img_vec = img_group_dict[label]

	for net in connected_g:
		if(len(net)<2):
			continue
		#pdb.set_trace()
		
		choosenet = np.array(list(net))
		sub_img = np.array(img_vec)[choosenet]
		sub_feature = aid_data[choosenet]
		new_img_dict[new_label] = sub_img
		new_feat_dict[new_label] = sub_feature
		new_label += 1
		countnum += sub_img.shape[0]
assert len(new_feat_dict)==len(new_img_dict)
print("del outliers, new group number:{}, image number:{}".format(len(new_feat_dict),countnum))

#二、合并相似的簇
#计算各簇中心点
centeridx = []
centerlist = []
for label in tqdm(new_feat_dict):
	featlist = new_feat_dict[label]
	featarr = np.array(featlist)
	avgfeat = np.mean(featarr,axis=0)
	centerlist.append(avgfeat)
	centeridx.append(label)
centerarr = np.array(centerlist)
ctidxarr = np.array(centeridx)

assert len(centeridx)==len(centerlist)
#计算各簇中心点间距离，并建立网络图
if USE_GPU:
	res = faiss.StandardGpuResources()
d_index = faiss.IndexFlatIP(featdim)   # build the index
if USE_GPU:
	d_index = faiss.index_cpu_to_gpu(res, 0, d_index)
id_index = faiss.IndexIDMap(d_index)
id_index.add_with_ids(centerarr, ctidxarr)

D, I = id_index.search(centerarr, 5)	# sanity check
D, I = id_index.search(centerarr, 5)	# actual search

edges= set()
for key,I_vec,D_vec in zip(ctidxarr,I,D):
	viewset = set()
	for index,dis in zip(I_vec,D_vec):
		if key==index or dis < IP_ct_thread:
			continue
		aid,bid = ((min(key,index), max(key,index)))
		edges.add(tuple((aid,bid)))
		viewset.add(tuple((aid,bid)))
	#print(viewset)
G = nx.Graph()
G.add_edges_from(edges)
#print("G num of edges:{}".format(G.number_of_edges()))
#print("G num of nodes:{}".format(G.number_of_nodes()))
connected_g=[]
for i in nx.connected_components(G):
	connected_g.append(i)
#print(centeridx)
#print(connected_g)
#print(new_img_dict)
merge_img_dict = {}
merge_feat_dict = {}
merge_idx = 0
countnum = 0
#合并相似的簇
for net in connected_g:
	merge_img = []
	merge_feat = []
	for idx in net:
		merge_img += new_img_dict[idx].tolist()
		merge_feat += new_feat_dict[idx].tolist()
		centeridx.remove(idx)
	merge_img_dict[merge_idx] = merge_img
	merge_feat_dict[merge_idx] = merge_feat
	merge_idx+=1
	countnum += len(merge_img)

for idx in centeridx:
	merge_img_dict[merge_idx] = new_img_dict[idx].tolist()
	merge_feat_dict[merge_idx] = new_feat_dict[idx].tolist()
	merge_idx+=1
	countnum += len(new_img_dict[idx].tolist())

assert len(merge_feat_dict)==len(merge_img_dict)
print('merge similar group, new group:{}, image number:{}'.format(len(merge_img_dict),countnum))
#三、簇内层次聚类，保留数量最多的一类，然后去重
countnum = 0
for idx,imgbatch in enumerate(tqdm(merge_feat_dict)):
	imgarr = np.array(merge_img_dict[idx])
	data = np.array(merge_feat_dict[idx])
	disMat = sch.distance.pdist(data,'euclidean')
	Z = sch.linkage(disMat,method='average')
	if SAVE_HIER:
		savedir = 'hier_cluster_pic'
		if not os.path.exists(savedir):
			os.mkdir(savedir)
		imgname = 'hier_id_'+str(idx)+'.png'
		img = os.path.join(savedir,imgname)
		P=sch.dendrogram(Z)
		plt.savefig(img)
	cluster = sch.fcluster(Z,hier_thread,criterion='distance')
	idmax = np.max(cluster)
	#print(cluster)

	if idmax==1:
		pass
	else:
		maxid = [0,0]
		for clu_id in range(1,(idmax+1)):
			num = sum(cluster==clu_id)
			if num>maxid[0]:
				maxid=[num,clu_id]
		chooseid = (cluster==maxid[1])
		data = data[chooseid]
		imgarr = imgarr[chooseid]
	clustering = DBSCAN(eps=Dup_thread, min_samples=2).fit(data)
	cluster_label = clustering.labels_
	#print('dbscan label:',cluster_label)
	#cluster_label = [ int(i) for i in cluster_label]
	chooseid = cluster_label==-1
	#data = data[chooseid]
	imgarr = imgarr[chooseid]
	merge_img_dict[idx] = imgarr
	#merge_feat_dict[idx] = data
	countnum += imgarr.shape[0]

assert len(merge_img_dict)==len(merge_feat_dict)
print("after hierarchical, img group:{}, image number:{}".format(len(merge_img_dict),countnum))

#删除只有单张的类
cluster_img_dict = {}
index = 1
countnum = 0
maxidx = 0

for idx in merge_img_dict:
	if len(merge_img_dict[idx])<2:
		continue
	if len(merge_img_dict[idx])>maxidx:
		maxlist = merge_img_dict[idx]
		maxidx = len(merge_img_dict[idx])
	cluster_img_dict[index] = merge_img_dict[idx]
	index += 1
	countnum += merge_img_dict[idx].shape[0]
print("final img group:{}, image number:{}".format(len(cluster_img_dict),countnum))

#四、保存聚类结果
savef = 'cluster_'+os.path.basename(dataf)
with open(savef,'w') as wf:
	for img_id in merge_img_dict:
		imgs = merge_img_dict[img_id]
		if len(imgs)<2:
			continue
		for img in imgs:
			nstr = img+'\t'+str(img_id)
			print(nstr,file=wf)

print("max idx list:",maxlist)

