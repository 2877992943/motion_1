#!/usr/bin/env python
# encoding=utf-8

"""
use accelerator only
fea12
xyz 1000x3->maxabs,minabs,mod 1000x3->window50_each_obs:20observationx3dim

for example
input:1000readings[x,y,z]
50readings per observation
20 observations
output 20labels


"""

 

 

 
import numpy as np
import pylab as plt
import cPickle,math,random,theano
import theano.tensor as T
 
import time,os,operator

dataPath='/home/yr/magnetic_fea/data1014/' 
deviceId={'huawei':'ffffffff-c7a8-3cd1-ffff-ffffea16e571', 
	'xiaomi':'ffffffff-c7a7-c7d4-0000-000031a92fbe', 
	'mine':'ffffffff-c95d-196f-0000-00007e282437',
	'vivo':'ffffffff-c43f-692d-ffff-fffff4d57110',
	'wangx':'ffffffff-c28a-1bf0-0000-00005e774605',
	'zhangx':'ffffffff-f259-8a54-0000-0000300c3e8c',
	'liyt':'ffffffff-c7a8-3c91-ffff-ffffa2dc21fc',
	'donggy':'ffffffff-c95e-eae5-ffff-ffffce758366',
	'hangwz':'ffffffff-c43f-77d4-0000-00001b78f081',
	'zishuai':'ffffffff-9475-8052-ffff-ffffaa303f0b',
	'test':'ffffffff-910d-998f-0000-0000017e0bce'
	}


###################
c_list=['running','walking','riding','sitting','driving']
class_type=c_list[1]
##################
nov_vivo=[[5,16,6,5,16,8],[6,13,28,6,13,37],[6,15,41,6,15,45]]
nov_test=[[6,16,31,6,16,36]]

inst_id_zs='mTl3M9NJ0qjkKOKFX378VWa37IFYKBe5'
inst_id_hjw='rumE011he7vtJxkINHkHQTdhkoBjJMcr'
inst_id_vivo='czte5wJCwAFRSUJWsC5ybyaezAhDnOm1'
###########query label.data
device=deviceId['vivo'] 
period=nov_vivo[-1]

###########query log.tracer
inst_id=inst_id_zs
#period=watch_nov_zs[-1]
 
 
#############33




def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	







def fea4(obs):#[50,]obs
	 
	mean=np.mean(obs);std=np.std(obs)
	min_i=np.min(obs);max_i=np.max(obs)
	f=np.array([mean,std,min_i,max_i])
	 
	return f
###############################################


def classify(inputTree,testVec):
	firstStr = inputTree.keys()[0]#[dim,value]
	dim1,v1=firstStr
	secondDict = inputTree[firstStr]
   
     
	if testVec[dim1] <=v1:#go left
        	if type(secondDict['left']).__name__ == 'dict':
			
                	classLabel = classify(secondDict['left'], testVec)
            	else: 
			classLabel = secondDict['left']
			 
	else:#go right
		if type(secondDict['right']).__name__ == 'dict':
                	classLabel = classify(secondDict['right'], testVec)
            	else: 
			classLabel = secondDict['right']
			 
				
	return classLabel



def normalize(x):#[n,12]
	num,dim_x=x.shape
	min_each_dim=np.min(x,axis=0);print 'normaliz',min_each_dim.shape#[12,]
	max_each_dim=np.max(x,axis=0);
	x1=(x-min_each_dim)/(max_each_dim-min_each_dim);print 'norm',x1.shape
	return x1



def predict_ensemble(X_test,stumps):
	n_val=X_test.shape[0];dim_val=X_test.shape[1]-1#[n,24+1]
	f_label_mat=np.zeros((n_val,stumps.__len__())) #[n,10stump]  
	
	for ind in range(stumps.__len__()):#[3,5,8,11..]  [tree,dimList,accuracy]
		dim_sample=stumps[ind][1]
		tree=stumps[ind][0]
		for obs in range(n_val):
			pred=classify(tree,X_test[obs,:])#13=12+1
			f_label_mat[obs,ind]=pred


	##
	#f_label majority vote
	 
	maj_vote=np.zeros((n_val,))#[n,]
	for i in range(f_label_mat.shape[0]): #[n,10stump]
		vote1=f_label_mat[i,:].sum()
		vote0=(1-f_label_mat[i,:]).sum()
		maj_vote[i]=[1 if vote1>vote0 else 0][0]
	######
	return sum(maj_vote)/float(n_val) #

def predict_eachObs(X_test,stumps_ds_wrr,stumps_d_s,stumps_r_rw,stumps_w_r):#[1,12]
	#######init label
	label=-1
	#############
	#drivesit 1 | walkrunrid 0
	###################
	
	#####ensemble test
	prob_drivesit=predict_ensemble(X_test,stumps_ds_wrr) 
	print 'drivesit percentage',prob_drivesit
	if prob_drivesit>=0.5:print 'not-final drive sit',prob_drivesit
	elif prob_drivesit<=0.5:print'not-final walkrunrid',1-prob_drivesit
	
	############
	#driving 0| sitting1
	################
	if prob_drivesit>=0.5:
		
		#acc no_mag, no_normalize x,magnetic=noise,accelerator only
		#####ensemble test
		sit_prob=predict_ensemble(X_test,stumps_d_s)#[1,13]
		print 'sit percentage',sit_prob
		if sit_prob>=0.5:
			print 'final pred sit',sit_prob
			label='sitting'
		elif sit_prob<0.5:
			print 'final pred drive',1.-sit_prob
			label='driving'


	############
	#runwalk 0|rid1
	################
	if prob_drivesit<0.5:
		#####ensemble test
		rid_prob=predict_ensemble(X_test,stumps_r_rw)#[1,13]
		print 'rid percentage',rid_prob
		if rid_prob>=0.5:
			print 'final pred rid',rid_prob
			label='riding'
		elif rid_prob<0.5:print 'not-final pred runwalk'


		############
		#run 0|walk1
		################
		if rid_prob<0.5:
			#####ensemble test
			walk_prob=predict_ensemble(X_test,stumps_w_r)#[1,13]
			print 'walk percentage',walk_prob
			if walk_prob>=0.5:
				print 'final pred walk',walk_prob
				label='walking'
			elif walk_prob<0.5:
				print 'final pred run',1-walk_prob
				label='running'
	return label



def main(xyz_acc):#[n,3]
	#########################
	#some error report
	##############
	mod_acc_mean=np.mean( np.sqrt( (xyz_acc*xyz_acc).sum(axis=1) ) )
	assert isinstance(xyz_acc,np.ndarray) 
	assert xyz_acc.shape[0]>=50 
	assert xyz_acc.shape[1]==3
	assert mod_acc_mean>=5 and mod_acc_mean<=50
	 
	##########################3
	##clip data clean
	#xyz_acc=xyz_acc[70:244,:]
	#save2pickle(xyz_acc,'normal-acc-xyz-'+class_type)	 
	#########
	abs_xyz_acc=np.abs(xyz_acc)
    	mod_acc=np.sqrt( (xyz_acc*xyz_acc).sum(axis=1) );print 'mod',mod_acc.shape #[n,]
    	maxabs_acc=abs_xyz_acc.max(axis=1);#print 'maxvec',maxabs.shape#[n,]
    	minabs_acc=abs_xyz_acc.min(axis=1);#print 'minvec',minabs.shape#[n,]
	 
	#########visual
	 
	ind1=range(mod_acc.shape[0]) 

    	plt.figure()
	plt.subplot(211);plt.title('acc-'+'mod maxabs minabs')
    	plt.plot(ind1,mod_acc,'ro',\
		ind1,maxabs_acc,'y-',\
		ind1,minabs_acc,'b--');#plt.ylim(0,2);plt.xlim(0,3500);

	 
	#plt.show()

	###########################
	#generate obs x y [n,3mod]->[n_obs,50,3] ->[n_obs,4x3]
	############################
	fea=np.concatenate((maxabs_acc.reshape((-1,1)),minabs_acc.reshape((-1,1)),\
			mod_acc.reshape((-1,1)) ),axis=1)#[n,3]
	 
	kernel_sz=50.;stride=kernel_sz;
    	
    	num=int( (fea.shape[0]-kernel_sz)/stride ) +1
	####only acc fea
	obs_list=[]; 
    	for i in range(num)[:]: #[0,...100] total 101 
        	obs=fea[i*stride:i*stride+kernel_sz,:]#[50,3]
		if obs.shape[0]==kernel_sz:
			v=np.array([fea4(obs[:,i]) for i in range(obs.shape[1])]).flatten()
			obs_list.append(v)#[10,3]->[3x4,]
	x_arr=np.array(obs_list);print 'x',x_arr.shape#[n-obs,12]
	
	 
	 
	

	###load all model####
	stumps_ds_wrr=load_pickle('/home/yr/magnetic_fea/data1102-drivesit-walkrunrid/rf-para-drivesit-walkrunrid')
	stumps_d_s=load_pickle('/home/yr/magnetic_fea/data1101_drivesit/rf-para-drivesit')
	stumps_r_rw=load_pickle('/home/yr/magnetic_fea/data1021/rf-para-rid-walkrun')
	stumps_w_r=load_pickle('/home/yr/magnetic_fea/data1024_walkrun/rf-para-walkrun')
	####each observation
	dataSet=np.concatenate((x_arr,np.zeros((x_arr.shape[0],1)) ),axis=1)#[n,12] [n,1]->[n,13]
	c_dic={'running':0,'walking':1,'riding':2,'sitting':3,'driving':4}
	#X_test=dataSet#[n,13]
	label_list=[]
	for i in range(dataSet.shape[0]):
		obs_i=dataSet[i,:].reshape((1,13))#[13,]->[1,13]
		X_test=obs_i#[1,13]
		label=predict_eachObs(X_test,stumps_ds_wrr,stumps_d_s,stumps_r_rw,stumps_w_r)#[1,13]->string
		label_list.append(c_dic[label])
	
	#############visual each obs_i 

	plt.subplot(2,1,2)
	plt.title('result obs one by one')
	plt.plot(label_list,'ro',label='running=0,walking=1,riding=2,sitting=3,driving=4')
	plt.xlabel('time');plt.ylabel('predict class')
	plt.legend()
	#plt.show()
	#
	return label_list 
		

############3
if __name__=="__main__":
	
	
	

	dataPath='/home/yr/magnetic_fea/data1014/'  ##in save2pickle()
	##########################################
	#  visual,test
	#################################################
	xyz_acc=load_pickle(dataPath+'acc-xyz-'+class_type)#[n,3]
	label_list=main(xyz_acc)
	plt.show()


	
	
	 
	
	
	

	
	 
	 

	 

 

	 
	
	 

	
    
 
	
		
	
   		 



