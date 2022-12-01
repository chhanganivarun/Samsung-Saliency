##

import os
from os.path import join
from loss import kldiv, cc, normalize_map, similarity, nss, auc_judd
# import discretize_gt,auc_borji
from PIL import Image
import argparse
import numpy as np
from tqdm import tqdm
import torch
import random
import time
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
from glob import glob
import cv2
import numpy as np
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred_maps_path', default='/ssd_scratch/cvit/varunc/Results/tsm/', type=str)
    # parser.add_argument('--pred_maps_path',default='/ssd_scratch/cvit/girmaji08/mvva_database/ViNet_test_frames',type=str)
    parser.add_argument(
        '--gt_maps_path', default='/ssd_scratch/cvit/varunc/DHF1K/val/', type=str)
    parser.add_argument(
        '--fix_maps_path', default='/ssd_scratch/cvit/varunc/DHF1K/val/', type=str)
    parser.add_argument('--num_of_samples', default=60, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    print(args)
    s_map, gt_map, fix_map = [], [], []
    score_kldiv, score_cc, score_nss, score_SIM, score_auc_judd = [], [], [], [], []
    sal_img_gt_tensor = torch.tensor([]).to(device)
    sal_img_pred_tensor = torch.tensor([]).to(device)
    video_sal_files = []
    # print(os.listdir(args.pred_maps_path))
    for i, video_name in enumerate(sorted(os.listdir(args.pred_maps_path))):
        print(video_name)
        s_map, gt_map, fix_map = [], [], []
        try:
            assert len(os.listdir(join(args.gt_maps_path, video_name, "maps"))) == len(
                os.listdir(join(args.pred_maps_path, video_name)))
        except:
            print(
                "****************************!!!!!!!!!!!!!!!!!!!!!!!!!!!!*************************")
            print(len(os.listdir(join(args.gt_maps_path, video_name, "maps"))), len(
                os.listdir(join(args.pred_maps_path, video_name))), video_name)
            continue

        # print(os.listdir(join(args.pred_maps_path,video_name)))
        for abc, sal_img_file in enumerate(os.listdir(join(args.pred_maps_path, video_name))):
            # print(abc)
            # if i == 0:
            sal_img_pred = Image.open(join(
                args.pred_maps_path, video_name, sal_img_file)).resize((640, 360)).convert('L')
            sal_img_pred = np.array(sal_img_pred).astype('float')
            sal_img_pred_tensor = torch.from_numpy(sal_img_pred).to(device)
            # s_map.append(sal_img_pred)
            # print(sal_img_pred_tensor.size())
            #sal_img_pred_tensor = torch.cat((sal_img_pred_tensor,tmp.unsqueeze(0)),dim = 0)
            # print(sal_img_pred_tensor.size())
            # sal_img_file = "eyeMap_"+sal_img_file.split("_")[-1]
            sal_img_file = sal_img_file.split("_")[-1]
            # print(sal_img_file)
            sal_img_gt = Image.open(join(
                args.gt_maps_path, video_name, "maps", sal_img_file)).resize((640, 360)).convert('L')
            sal_img_gt = np.array(sal_img_gt).astype('float')
            sal_img_gt_tensor = torch.from_numpy(sal_img_gt).to(device)
            gt_map.append(sal_img_gt)
            # print(sal_img_gt_tensor.size())
            #sal_img_gt_tensor = torch.cat((sal_img_gt_tensor,tmp.unsqueeze(0)),dim = 0)
            # print(sal_img_gt_tensor.size())
            # print(sal_img_gt_tensor.size())

            fix_img_gt = Image.open(
                join(args.fix_maps_path, video_name, 'fixation', sal_img_file)).convert('L')
            fix_img_gt = np.array(fix_img_gt).astype('float')
            fix_img_gt_tensor = torch.from_numpy(fix_img_gt).to(device)
            fix_img_gt_tensor = normalize_map(fix_img_gt_tensor.unsqueeze(0))
            # print(np.sum(np.round(normalize_map(fix_img_gt_tensor.unsqueeze(0)).cpu().numpy())))
            #fix_img_gt = cv2.imread(join(args.fix_maps_path,video_name,sal_img_file))
            #fix_img_gt = cv2.cvtColor(fix_img_gt,cv2.COLOR_BGR2GRAY)
            # print(sorted(fix_img_gt.flatten().tolist())
            #fix_img_gt_arr = np.array(fix_img_gt)
            #print(np.sum(np.where(fix_img_gt_arr == 1.0,1,0)))
            # print(fix_img_gt_arr.shape,np.min())
            # fix_map.append(fix_img_gt_arr)
            #s_map = np.stack(s_map)
            # print(s_map)
            #gt_map = np.stack(gt_map)
            #fix_map = np.stack(gt_map)
            # print(fix_map.size())
            score_auc_judd.append(auc_judd(sal_img_pred_tensor.cpu(
            ), fix_img_gt_tensor.cpu(), jitter=True, toPlot=False, normalize=False))

            video_sal_files.append((video_name, sal_img_file))
            #print(sal_img_pred_tensor.size() , sal_img_gt_tensor.size() , fix_img_gt_tensor.size())
            assert sal_img_pred_tensor.size() == sal_img_gt_tensor.size(
            ) == fix_img_gt_tensor.squeeze(0).size()
            # gt_map.append(sal_img_gt_arr)
            score_kldiv.append(kldiv(sal_img_pred_tensor.unsqueeze(
                0) / 255.0, sal_img_gt_tensor.unsqueeze(0) / 255.0))
            # print(kldiv(sal_img_pred_tensor.unsqueeze(0) / 255.0,sal_img_gt_tensor.unsqueeze(0) / 255.0))
            score_cc.append(cc(sal_img_pred_tensor.unsqueeze(0),
                            sal_img_gt_tensor.unsqueeze(0)))
            # print(cc(sal_img_pred_tensor.unsqueeze(0),sal_img_gt_tensor.unsqueeze(0)))
            score_nss.append(nss(sal_img_pred_tensor.unsqueeze(
                0), sal_img_gt_tensor.unsqueeze(0)))
            # print(nss(sal_img_pred_tensor.unsqueeze(0),sal_img_gt_tensor.unsqueeze(0)))
            score_SIM.append(similarity(sal_img_pred_tensor.unsqueeze(
                0), sal_img_gt_tensor.unsqueeze(0)))
            # print(similarity(sal_img_pred_tensor.unsqueeze(0),sal_img_gt_tensor.unsqueeze(0)))
            # break
        # print("auc",np.mean(np.array(score_auc_judd)))
        # print("kl",np.mean(np.array(score_kldiv)))
    print(torch.mean(torch.tensor(score_auc_judd)))
    print(torch.mean(torch.tensor(score_kldiv)))
    print(torch.mean(torch.tensor(score_cc)))
    print(torch.mean(torch.tensor(score_nss)))
    print(torch.mean(torch.tensor(score_SIM)))
    # print("cc",np.mean(np.array(score_cc)))
    # print("nss",np.mean(np.array(score_nss)))
    # print("sim",np.mean(np.array(score_SIM)))
    # print(score_kldiv)
    ####score_kldiv = np.array([i.item() for i in score_kldiv]).reshape(-1,1)
    # for i in range(score_kldiv.shape[0]):
    # if score_kldiv[i] > 1.5:
    # print(video_sal_files[i])

    # print(np.count_nonzero(np.isnan(score_kldiv)))
    # print(np.count_nonzero(~np.isnan(score_kldiv)))
    # print(np.nansum(score_kldiv))
    ####score_cc = np.array([i.item() for i in score_cc])
    ####score_nss = np.array([i.item() for i in score_nss])
    ####score_SIM = np.array([i.item() for i in score_SIM])

    ####print("KL Div:",np.nanmean(score_kldiv))
    # print("CC:",np.nanmean(score_cc))
    # print("NSS:",np.nanmean(score_nss))
    # print("SIM:",np.nanmean(score_SIM))
    #print("KL Div:",sum([i.item() for i in score_kldiv])/len(score_kldiv))
    #print("CC:",sum([i.item() for i in score_cc])/len(score_cc))
    #print("NSS:",sum([i.item() for i in score_nss])/len(score_nss))
    #print("SIM:",sum([i.item() for i in score_SIM])/len(score_SIM))
    print("AUC_judd:", sum([i for i in score_auc_judd])/len(score_auc_judd))
"""
    #s_map = np.stack(s_map)
    #s_map = torch.from_numpy(s_map).to(device)
    s_map = sal_img_pred_tensor
    #gt_map = np.stack(gt_map)
    #gt_map = torch.from_numpy(gt_map).to(device)
    gt_map = sal_img_gt_tensor
    print(s_map.size())
    assert s_map.size() == gt_map.size()
    for i in range(0,s_map.size()[0],args.batch_size):
        interval = i, i + i*args.batch_size
    score_kldiv = kldiv(s_map,gt_map)
    score_cc = cc(s_map,gt_map)
    score_nss = nss(s_map,gt_map)
    score_SIM = similarity(s_map,gt_map)
        for sal_img_file in os.listdir(join(args.pred_maps_path,video_name)):
            sal_img_pred = Image.open(join(args.pred_maps_path,video_name,sal_img_file)).resize((640,360)).convert('L')
            sal_img_pred_arr = np.array(sal_img_pred).astype('float')
            s_map.append(sal_img_pred_arr)
            sal_img_gt = Image.open(join(args.gt_maps_path,video_name,sal_img_file)).resize((640,360)).convert('L')
            sal_img_gt_arr = np.array(sal_img_gt).astype('float')
            gt_map.append(sal_img_gt_arr)

    s_map = np.stack(s_map)
    s_map = torch.from_numpy(s_map).to(device)
    gt_map = np.stack(gt_map)
    gt_map = torch.from_numpy(gt_map).to(device)

    assert s_map.size() == gt_map.size()

    score_kldiv = kldiv(s_map,gt_map)
    score_cc = cc(s_map,gt_map)
    score_nss = nss(s_map,gt_map)
    score_SIM = similarity(s_map,gt_map)
    print("KL Div:",score_kldiv)
    print("CC:",score_cc)
    print("NSS:",score_nss)
    print("SIM:",score_SIM)
"""


"""
def calc_auc(s_map,fix_map,range_interval):
    print(range_interval)
    values_judd = [(s_map[i].cpu(), fix_map[i],True, False, False) for i in range(range_interval[0],range_interval[1])]
    values_borji = [(s_map[i].cpu(), fix_map[i].numpy(),100,0.1) for i in range(range_interval[0],range_interval[1])]
    with Pool() as pool:
        auc_judd_output = pool.starmap(auc_judd,values_judd)
        auc_borji_output = pool.starmap(auc_borji,values_borji)
    print(auc_judd_output)
    print(auc_borji_output)
    score_auc_judd_list.append(auc_judd_output) 
    score_auc_borji_list.append(auc_borji_output)
    return score_auc_judd_list,score_auc_borji_list

def calc_auc_pool(s_map,fix_map,range_interval):
    pool = Pool()
    #values_judd = [(s_map[i].cpu(), fix_map[i],True, False, False) for i in range(range_interval[0],range_interval[1])]
    result_async = [pool.apply_async(auc_judd, args = (s_map[i].cpu(), fix_map[i],True, False, False)) for i in range(range_interval[0],range_interval[1])]
    pool.close()
    pool.join()
    results1 = [r.get() for r in result_async]
    print(results1)
    pool = Pool()
    result_async = [pool.apply_async(auc_borji, args = (s_map[i].cpu(), fix_map[i].numpy(),100,0.1)) for i in range(range_interval[0],range_interval[1])]
    pool.close()
    pool.join()
    results2 = [r.get() for r in result_async]
    print(results2)
    return results1,results2
def get_metrics(indx,video_names_list):
    s_map,gt_map,fix_map = [],[],[]
    video_name = video_names_list
    print("Processing video: ",video_name)
    #for i,video_name in enumerate(video_names_list):
        #if i == indx:
            #print("video_name:",video_name)
    #try:
    sal_img_paths = sorted(list(set(os.listdir(join(args.pred_maps_path,video_name))) & set(os.listdir(join(args.gt_maps_path,video_name))) & set(os.listdir(join(args.fix_maps_path,video_name)))))
    for sal_img_path in sal_img_paths:
        #sal_img_paths.append(sal_img_path)
        sal_img_pred = Image.open(join(args.pred_maps_path,video_name,sal_img_path)).resize((640,360)).convert('L')
        sal_img_pred_arr = np.array(sal_img_pred).astype('float')
        s_map.append(sal_img_pred_arr)
        sal_img_gt = Image.open(join(args.gt_maps_path,video_name,sal_img_path)).resize((640,360)).convert('L')
        sal_img_gt_arr = np.array(sal_img_gt).astype('float')
        gt_map.append(sal_img_gt_arr)
        fix_img_gt = Image.open(join(args.fix_maps_path,video_name,sal_img_path)).resize((640,360)).convert('L')
        fix_img_gt_arr = np.array(fix_img_gt)
        fix_map.append(fix_img_gt_arr)
        print(np.max(fix_img_gt_arr))
    #except:
    #    continue
    s_map = np.stack(s_map)
    s_map = torch.from_numpy(s_map).to(device)
    gt_map = np.stack(gt_map)
    gt_map = torch.from_numpy(gt_map).to(device)
    fix_map = np.stack(fix_map)
    fix_map = torch.from_numpy(fix_map)
    assert s_map.size() == gt_map.size()
    assert gt_map.size() == fix_map.size()

    score_kldiv = kldiv(s_map,gt_map)
    score_cc = cc(s_map,gt_map)
    score_nss = nss(s_map,gt_map)
    score_SIM = similarity(s_map,gt_map)
    score_auc_judd = 0
    score_auc_borji = 0
    print("Stage1 finished")
    start = time.time()
    score_auc_judd_list = []
    score_auc_borji_list = []
    print(s_map.shape)

    l = np.arange(0,s_map.shape[0] + 1,2)
    range_intervals = [(l[i],l[i+1]) for i in range(0,len(l) - 1) if i < len(l)]

    print("rangeintervals: ",range_intervals)

    full_auc_judd_list,full_auc_borji_list = [],[]
    for range_interval in range_intervals:
        score_auc_judd_list,score_auc_borji_list = calc_auc_pool(s_map,fix_map,range_interval)
        full_auc_judd_list.append(score_auc_judd_list)
        full_auc_borji_list.append(score_auc_borji_list)
    score_auc_borji = np.mean(np.array(sum(full_auc_judd_list,[])))
    score_auc_judd = np.mean(np.array(sum(full_auc_borji_list,[])))
    

    #values_judd = [(s_map[i].cpu(), fix_map[i],True, False, False) for i in range()]
    #values_borji = [(s_map[i].cpu(), fix_map[i].numpy(),100,0.1) for i in range(s_map.shape[0])]
    #for i in range(s_map.shape[0]): 
    #    if i==0:
    #            print(fix_map[i].shape,fix_map[i])       
    #with Pool() as pool:
    #    score_auc_judd_list = pool.starmap(auc_judd,values_judd)
    #    score_auc_borji_list = pool.starmap(auc_borji,values_borji)
   
    print("stage2 Finished")           
    #score_auc_borji = np.mean(score_auc_borji_list)
    #score_auc_judd = np.mean(score_auc_judd_list)

    end = time.time()

    print("Time for AUC Judd,Borji: {:.4f}", end - start)
    print(score_auc_judd)
    return score_kldiv,score_cc,score_nss,score_SIM,score_auc_judd,score_auc_borji
if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_maps_path',default='/ssd_scratch/cvit/girmaji08/mvva_database/predictions_saliency_maps/ViNet_AudioVisual_split1',type=str)
    parser.add_argument('--gt_maps_path',default='/ssd_scratch/cvit/girmaji08/mvva_database/save_our_hmap',type=str)
    parser.add_argument('--fix_maps_path',default='/ssd_scratch/cvit/girmaji08/mvva_database/fixation_maps',type=str)
    parser.add_argument('--num_of_samples',default= 60 ,type=int)
    
    args = parser.parse_args()
    print(args)
    video_names_formatted = []
    s_map,gt_map = [],[]
    video_names = os.listdir(args.pred_maps_path)
    for video_name in video_names:
        video_name = video_name.replace("'","").replace('"','')
        video_names_formatted.append(video_name)
    count = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    video_names_list = sorted(list(set(sorted(video_names_formatted)) & set(sorted(os.listdir(args.gt_maps_path)))))

    kldiv_scores,cc_scores,nss_scores,SIM_scores,auc_judd_scores,auc_borji_scores = [],[],[],[],[],[]
    indx_list = range(0,60)
    #random.seed(0)
    #print("video list random: ",sorted(random.sample(range(0,len(video_names_list)),60)))
    #random.seed(0)
    test_video_names = os.listdir('/ssd_scratch/cvit/girmaji08/mvva_database/ViNet_test_Dataset/MVVA')
    print(test_video_names)
    for i,video_name in enumerate(sorted(test_video_names)):
        print(i)
        score_kldiv,score_cc,score_nss,score_SIM,score_auc_judd = get_metrics(i,video_name)
        kldiv_scores.append(score_kldiv.item())
        cc_scores.append(score_cc.item())
        nss_scores.append(score_nss.item())
        SIM_scores.append(score_SIM.item())
        auc_judd_scores.append(score_auc_judd)
        auc_borji_scores.append(score_auc_borji)
    
    print("Number of videos: ", len(kldiv_scores))
    print("kl_div:" ,sum(kldiv_scores)/len(kldiv_scores))
    print("CC:" ,sum(cc_scores)/len(cc_scores))
    print("NSS:",sum(nss_scores)/len(nss_scores))
    print("SIM:",sum(SIM_scores)/len(SIM_scores))
    print("judd:",sum(auc_judd_scores)/len(auc_judd_scores))
    print("borji:",sum(auc_borji_scores)/len(auc_borji_scores))
"""
"""
    s_map = np.stack(s_map)
    print(s_map.shape)
    s_map = torch.tensor(s_map)
    print(s_map.size())
    gt_map = np.stack(gt_map)
    print(gt_map.shape)
    gt_map = torch.tensor(gt_map)
    print(gt_map.size())


    assert s_map.size() == gt_map.size()
    for i in range(10,70,10):

    score_kldiv = kldiv(s_map,gt_map)
    print("kl_div:" ,score_kldiv)
    score_cc = cc(s_map,gt_map)
    print("CC:" ,score_cc)
    score_nss = nss(s_map,gt_map)
    print("NSS:",score_nss)
    score_SIM = similarity(s_map,gt_map)
    print("SIM:",score_SIM)
"""

"""
    for video_name in sorted(video_names_formatted)[0:num_of_samples]:
        #sal_images = os.listdir(join(args.pred_maps_path,video_name))
        #sal_images = sorted([sal_img[9:].replace('.jpg','') for sal_img in sal_images])
        for sal_img in sorted(os.listdir(join(args.pred_maps_path,video_name))):
            s_map.append(Image.open(join(args.pred_maps_path,video_name,sal_img)).convert('L'))
    for video_name in sorted(os.listdir(args.gt_maps_path))[0:num_of_samples]:
        for sal_img in sorted(os.listdir(join(args.gt_maps_path,video_name))):
            gt_map.append(Image.open(join(args.gt_maps_path,video_name,sal_img)).convert('L'))
"""


"""

"""

"""
import os
import shutil
list_dir = [dir for dir in os.listdir('./') if dir.endswith('.avi')]
for dir in list_dir: 
    shutil.move(dir,'{:03}'.format(int(dir.replace('.avi','')))) 
for dir in os.listdir('/ssd_scratch/cvit/girmaji08/mvva_database/mvva_results_final/mvva_results'):
    shutil.move(dir,'video_' + dir)
for dir in os.listdir('./'):
    shutil.move(dir,'video_' + dir)
"""

"""
        #start = time.time()
        #for i in range(s_map.shape[0]):
            #print(s_map.shape[0])
        #    score_auc_judd += auc_judd(s_map[i].cpu(), fix_map[i], jitter=True, toPlot=False, normalize=False)
        #end = time.time()

        # n_jobs is the number of parallel jobs
        #score_auc_judd_list = Parallel(n_jobs=160,verbose = 10)(delayed(auc_judd)(s_map[i].cpu(), fix_map[i], jitter=True, toPlot=False, normalize=False) for i in range(s_map.shape[0]))
        #score_auc_judd = np.mean(score_auc_judd_list)
"""
