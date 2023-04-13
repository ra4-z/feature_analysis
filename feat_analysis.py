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
    '''
        read features.
    '''
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def pic_name_reader(path=pic_name_path) -> list:
    '''
        read pic name from txt file
        pic name is in the format of VeRi776, e.g. 0001_c001_00005100_00.jpg
    '''
    ids = []
    cams = []
    timestamps = []
    with open(path, 'r') as f:
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
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def distill_info(features, ids, cams, timestamps):  
    '''
        distill info from features, ids, cams, timestamps
        return a dict, id - cam - feature(tensor)
    '''
    id_cam_feature = {}
    for idx in range(len(ids)):
        id = ids[idx]
        cam = cams[idx]
        feature = features[idx]
        if id not in id_cam_feature:
            id_cam_feature[id] = {}
        if cam not in id_cam_feature[id]:
            id_cam_feature[id][cam] = []
        id_cam_feature[id][cam].append(feature)

    # transfer list to tensor
    for id in id_cam_feature:
        for cam in id_cam_feature[id]:
            id_cam_feature[id][cam] = torch.stack(id_cam_feature[id][cam])

    return id_cam_feature


def calc_dist(id_cam_feature, dist_func='cosine', save_dir=None):
    '''
        calculate distance between features in three different ways:
        1) features with the same id and the same cam
        2) features with the same id but different cam
        3) features with different id and different cam

        then calculate the mean and std of the three distances

        if save_dir is not None, save the three distances to log file, 
            and draw the distribution figures of the three distances
    '''
    same_id_same_cam_dist = []
    same_id_diff_cam_dist = []
    diff_id_diff_cam_dist = []
    
    if dist_func == 'cosine':
        dist_func = compute_cosine_distance
    elif dist_func == 'euclidean':
        dist_func = compute_euclidean_distance
    else:
        raise ValueError("dist_func should be 'cosine' or 'euclidean'")
    
    for id in id_cam_feature: # TODO parallel
        cams = id_cam_feature[id]
        for cam in cams:
            features = cams[cam]
            for i in range(features.shape[0]):
                for j in range(i+1, features.shape[0]):
                    same_id_same_cam_dist.append(dist_func(features[i], features[j]))

        for cam1 in cams:
            for cam2 in cams:
                if cam1 == cam2:
                    continue
                features1 = cams[cam1]
                features2 = cams[cam2]
                for i in range(features1.shape[0]):
                    for j in range(features2.shape[0]):
                        same_id_diff_cam_dist.append(dist_func(features1[i], features2[j]))
    
    # do the math, mean and std
    
    
    
    # save the three distances to log file, 
    # and draw the distribution figures of the three distances separately
    if save_dir is not None:
        import matplotlib.pyplot as plt
        


if __name__ == '__main__':
    features = feature_reader()
    ids, cams, timestamps = pic_name_reader()
    assert features.shape[0] == len(
        ids), "feature number not equal to pic name number"

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
