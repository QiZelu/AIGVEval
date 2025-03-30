import torch
# import wandb
import cv2
import random
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import yaml
import datetime
from collections import OrderedDict
from copy import deepcopy
from functools import reduce
from thop import profile
import copy

from model.model_motion_combine import T2VQA
from dataset.dataset_motion import T2VDataset
from model.constants import DEFAULT_IMAGE_PATCH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def train_test_split(dataset_path, ann_file, ratio=0, seed=42):
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt = line_split
            label = float(0.0)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    return video_infos[:]

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def rescale(pr, gt=None):

    if gt is None:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * 0.62313 + 2.90869
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

def inference_set(
    inf_loader,
    model,
    device,
    best_,
    save_model=False,
    suffix="s",
    save_name="divide",
    save_type="head",
):

    results = []
    vid_names = []
    result_file="output.json"

    best_s, best_p, best_k, best_r = best_
    
    model.eval()

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}
        vid_names.append(data["filename"][0].split("/")[-1])

        video['video'] = data['video'].to(device)
        video['vfrag'] = data["vfrag"].to(device)
        video['slowfast'] = [data['slow'].to(device), data['fast'].to(device)]
        
        ## Reshape into clips
        b, c, t, h, w = video['video'].shape
            
        with torch.no_grad():

            prompt = "The key frames of this video are:" + "\n" + DEFAULT_IMAGE_TOKEN + ". The motion feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". And the technical quality feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". Please assess the quality of this video."

            caption = ""

            result["pr_labels"] = model(video, caption = caption, prompt = prompt).cpu().numpy()

            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del video, video_up
        results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"]) for r in results]
    pr_labels = rescale(pr_labels)

    output_data = [
        {"video": name, "predict_score": float(score)}  # 显式转换为Python float类型
        for name, score in zip(vid_names, pr_labels)
    ]

    # 写入JSON文件
    import json
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)  # 保持可读格式

    print(f"\n预测结果已保存至 {result_file}")

    del results, result  # , video, video_up
    torch.cuda.empty_cache()

    return best_s, best_p, best_k, best_r

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="/home/u202320081001008/share/IMC/T2VEval/t2vqa_test.yml", help="the option file"
    )

    parser.add_argument(
        "-t", "--target_set", type=str, default="t2v", help="target_set"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## defining model and loading checkpoint

    bests_ = []

    model = T2VQA(opt["model"]["args"]).to(device)
    print(model.print_trainable_parameters())

    state_dict = torch.load(opt['test_load_path'], map_location='cpu')['state_dict']

    print(model.load_state_dict(state_dict, strict=False))

    if opt.get("split_seed", -1) > 0:
        opt["data"]["train"] = copy.deepcopy(opt["data"][args.target_set])
        opt["data"]["eval"] = copy.deepcopy(opt["data"][args.target_set])

        split_duo = train_test_split(
            opt["data"][args.target_set]["args"]["data_prefix"],
            opt["data"][args.target_set]["args"]["anno_file"],
            seed=opt["split_seed"],
        )
        
        opt["data"]["eval"]["args"]["anno_file"] = split_duo

    val_datasets = {}
    for key in opt["data"]:
        if key.startswith("eval"):
            val_dataset = T2VDataset(
                opt["data"][key]["args"]
            )
            print(len(val_dataset.video_infos))
            val_datasets[key] = val_dataset

    val_loaders = {}
    for key, val_dataset in val_datasets.items():
        val_loaders[key] = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
            num_workers=opt["num_workers"],
            pin_memory=True,
        )

    bests = {}
    for key in val_loaders:
        bests[key] = -1, -1, -1, 1000
        bests[key] = inference_set(
            val_loaders[key],
            model,
            device,
            bests[key],
            save_model=False,
            save_name=None,
            suffix=key + "_s",
            save_type="full",
        )

    for key in val_loaders:
        print(
            f"""For the end-to-end transfer process on {key} with {len(val_loaders[key])} videos,
            the best validation accuracy of the model-s is as follows:
            SROCC: {bests[key][0]:.4f}
            PLCC:  {bests[key][1]:.4f}
            KROCC: {bests[key][2]:.4f}
            RMSE:  {bests[key][3]:.4f}."""
        )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
