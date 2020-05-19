#coding=utf-8
from __future__ import print_function
import os
import sys
import faiss
import numpy as np
import pdb
import shutil
import networkx as nx
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from tqdm import tqdm

dataf = sys.argv[1]	#待聚类数据列表，格式为imgpath \t group \t feature

IP_ct_thread = 0.6
USE_GPU = 0
featdim = 128
with open(dataf,'r') as rf:
	lines = rf.readlines()

orgimgnum = len(lines)
org_img_dict = {}
org_feat_dict = {}
keydict = {}
idx = 0
for line in lines:
	imgpath,feat = eval(line)
	basename = os.path.basename(imgpath)
	basekey = basename.split('_')[0]
	basedir = os.path.dirname(imgpath)
	label = int(basedir.split('/')[-1])
	if label not in org_img_dict:
		org_img_dict[label] = [imgpath]
		org_feat_dict[label] = [feat]
	else:
		org_img_dict[label] = org_img_dict[label]+[imgpath]
		org_feat_dict[label] = org_feat_dict[label]+[feat]
	if basekey not in keydict:
		keydict[basekey] = [label]
	else:
		if label in keydict[basekey]:
			continue
		keydict[basekey] = keydict[basekey]+[label]


assert len(org_img_dict)==len(org_feat_dict)
orgimggrp = len(org_img_dict)
platenum = len(keydict)
print("org platenum:{} img group:{} number:{}".format(platenum,orgimggrp,orgimgnum))

newlabel = 0
newdict = {}
for basekey in tqdm(keydict):
	labellist = keydict[basekey]
	gnum = len(labellist)
	if gnum==1:
		continue
	new_img_dict = {}
	new_feat_dict = {}
	imgnum = 0
	for label in labellist:
		new_img_dict[label] = org_img_dict[label]
		new_feat_dict[label] = org_feat_dict[label]
		imgnum += len(org_img_dict[label])
	
	#print("\nbefore merge, group:{} number:{}".format(gnum,imgnum))
	centeridx = []
	centerlist = []
	for label in new_feat_dict:
		featlist = new_feat_dict[label]
		featarr = np.array(featlist)
		avgfeat = np.mean(featarr,axis=0)
		centerlist.append(avgfeat)
		centeridx.append(label)
	centerarr = np.array(centerlist).astype(np.float32)
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
			merge_img += new_img_dict[idx]
			merge_feat += new_feat_dict[idx]
			centeridx.remove(idx)
		merge_img_dict[merge_idx] = merge_img
		merge_feat_dict[merge_idx] = merge_feat
		merge_idx+=1
		countnum += len(merge_img)

	for idx in centeridx:
		merge_img_dict[merge_idx] = new_img_dict[idx]
		merge_feat_dict[merge_idx] = new_feat_dict[idx]
		merge_idx+=1
		countnum += len(new_img_dict[idx])

	assert len(merge_feat_dict)==len(merge_img_dict)
	#print('merge similar group, new group:{}, image number:{}'.format(len(merge_img_dict),countnum))
	if gnum!=len(merge_img_dict):
		print("org group:{} new group:{} new label:{} length:{}".format(gnum,len(merge_img_dict),newlabel,len(merge_img_dict)))
	else:
		print("same gn")
	maxnum = 0
	maxkey = 0
	for key in merge_img_dict:
		if len(merge_img_dict[key])>maxnum:
			maxnum = len(merge_img_dict[key])
			maxkey = key

	newdict[newlabel] = merge_img_dict[maxkey]
	newlabel += 1

print("final group:",len(newdict))

savedir = 'merge_dir'
shutil.rmtree(savedir)
os.mkdir(savedir)

finallst = []
for key in newdict:
	imgdir = os.path.join(savedir,str(key))
	if not os.path.exists(imgdir):
		os.mkdir(imgdir)
	imglist = newdict[key]
	for img in imglist:
		nstr = '\t'.join((img,str(key)))
		finallst.append(nstr)
		img = img.replace('/train/execute/carface_zhibo/pureid_190318','pudata_2k')
		baseimg = os.path.basename(img)
		nimg = os.path.join(imgdir,baseimg)
		#shutil.copy(img,nimg)

savef = 'merge_'+os.path.basename(dataf)
with open(savef,'w') as wf:
	for item in finallst:
		print(item,file=wf)




