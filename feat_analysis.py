import numpy as np
import pickle
import json
import torch
from utils import compute_euclidean_distance, compute_cosine_distance

# path
feat_path = 'data/feature/feature.pkl'
pic_name_path = 'data/feature/pic_name.txt'
split_path = 'data/feature/dataset_split.json'


def feature_reader(path=feat_path) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pic_name_reader(path=pic_name_path) -> list:
    '''
        read pic name from txt file
        pic name is in the format of VeRi776, e.g. 0001_c001_000051_00.jpg
    '''
    ids = []
    cams = []
    timestamps = []
    with open(path,'r') as f:
        for line in f:
            if line.strip():
                id, cam, timestamp, _ = line.strip().split('_')
                id = int(id)
                cam = int(cam[1:])
                timestamp = int(timestamp)
                ids.append(id)
                cams.append(cam)
                timestamps.append(timestamp)
    
    return ids, cams, timestamps   
                


def split_reader(path=split_path) -> dict:
    '''
        dataset split information reader
    '''
    with open(path,'r') as f:
        data = json.load(f)
    return data


def distill_info(features, ids, cams, timestamps): # TODO to be modified
    '''
        distill info from features, ids, cams, timestamps
        return a dict, key is id, value is a list of tuple (cam, timestamp, feature)
    '''
    info = {}
    for i in range(len(ids)):
        id = ids[i]
        cam = cams[i]
        timestamp = timestamps[i]
        feature = features[i]
        if id not in info:
            info[id] = [(cam, timestamp, feature)]
        else:
            info[id].append((cam, timestamp, feature))
    return info


if __name__ == '__main__':
    features = feature_reader()
    ids, cams, timestamps  = pic_name_path()
    assert features.shape[0] == len(ids), "feature number not equal to pic name number"
    

    ##############################
    # average gallery feature for each identity with same cam
    '''
        gf_mod = {} # camid - pid - feature
        for i in range(len(q_pids)):
            pid = q_pids[i]
            camid = q_camids[i]
            if camid not in gf_mod:
                gf_mod[camid] = {}
            if pid not in gf_mod[camid]:
                gf_mod[camid][pid] = []
            gf_mod[camid][pid].append(gf[i])
        
        gf_mod_avg = {}
        for camid in gf_mod:
            gf_mod_avg[camid] = {}
            for pid in gf_mod[camid]:
                gf_mod_avg[camid][pid] = torch.stack(gf_mod[camid][pid]).mean(dim=0)

        gf_mod_avg_f=[]
        gf_mod_avg_pids = []
        gf_mod_avg_camids = []
        for camid in gf_mod_avg:
            for pid in gf_mod_avg[camid]:
                gf_mod_avg_f.append(gf_mod_avg[camid][pid])
                gf_mod_avg_pids.append(pid)
                gf_mod_avg_camids.append(camid)

        import matplotlib.pyplot as plt
        def draw_distribution(qf,gf,qf_mod,out_dir = '/data/vehicle/code/TransReID/figs'):
            def get_dist(qf,gf):
                distmat = euclidean_distance(qf, gf)
                distmat = distmat.flatten()
                distmat = distmat[distmat > 0.01]
                return distmat
            distmat = {"original":get_dist(qf,gf), 
                       "feature-fused":get_dist(qf_mod,gf)
                       }
            
            # draw two distribution in percentage scale in one figure with contour line
            
            
            fig, ax = plt.subplots()
            for key in distmat:
                ax.hist(distmat[key], bins=100, alpha=0.5, label=key)
            ax.legend(loc='upper left')
            
            ax.set_xlabel('Distance')
            ax.set_ylabel('Number percentage')
            plt.yticks([])  # 去掉纵坐标值
            plt.savefig(out_dir + "/dist.png")
            plt.close()
            



        # draw_distribution(qf, gf)
        draw_distribution(qf,gf, torch.stack(gf_mod_avg_f))

        print("ok")
        
        # plt.savefig(out_dir + "/" + title + "_loss.png")

        '''