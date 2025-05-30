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

import os

def train_test_split(dataset_path, val_path, ann_file, val_file, ratio=1, seed=42):
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            # 检查行是否符合预期的格式
            if len(line_split) != 3:
                print(f"Skipping invalid line: {line.strip()}")
                continue  # 跳过格式不正确的行
            filename, prompt, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
   
    val_infos = []
    with open(val_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt = line_split
            label = float(0.0)
            filename = osp.join(val_path, filename)
            val_infos.append(dict(filename=filename, prompt=prompt, label=label))
    
    return (
        video_infos[: int(ratio * len(video_infos))],
        val_infos[: int(ratio * len(val_infos))],
    )

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
        pr = (pr - np.mean(pr)) / np.std(pr) * 0.62313 + 2.90869
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

def finetune_epoch(
    ft_loader,
    model,
    model_ema,
    optimizer,
    scheduler,
    device,
    epoch=-1,
):
    model.train()
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):

        optimizer.zero_grad()

        video = {}
        video["video"] = data["video"].to(device)
        video['vfrag'] = data["vfrag"].to(device)
        video['slowfast'] = [data['slow'].to(device), data['fast'].to(device)]

        # video["frame_inds"] = data["frame_inds"].to(device)

        y = data["gt_label"].float().detach().to(device)

        caption = ""
        
        prompt = "The key frames of this video are:" + "\n" + DEFAULT_IMAGE_TOKEN + ". The motion feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". And the technical quality feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". Please assess the quality of this video."
        
        # finetune the model
        scores = model(video, caption = caption, prompt = prompt)

        y_pred = scores
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))

        p_loss = plcc_loss(y_pred, y)
        r_loss = rank_loss(y_pred, y)

        loss = p_loss + 0.3 * r_loss
        

        loss.backward()
        optimizer.step()
        scheduler.step()

        # ft_loader.dataset.refresh_hypers()
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )

    model.eval()

def inference_set(
    epoch,
    inf_loader,
    model,
    device,
    best_,
    save_model=False,
    suffix="s",
    save_name="divide_swin_conv_adds_classification_pth",
    save_type="head",
    model_save_dir='pretrained_weights_1029_2'
):

    results = []
    vid_names = []
    result_file= "t2veval_{}.json".format(epoch)

    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):

        result = dict()
        video, video_up = {}, {}

        video['video'] = data['video'].to(device)
        video['vfrag'] = data["vfrag"].to(device)
        video['slowfast'] = [data['slow'].to(device), data['fast'].to(device)]

        vid_names.append(data["filename"][0].split("/")[-1])
        
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

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean()) 

    print(f"val_{suffix}/SRCC-{suffix}",s)
    print(f"val_{suffix}/PLCC-{suffix}",p)
    print(f"val_{suffix}/KRCC-{suffix}",k)
    print(f"val_{suffix}/RMSE-{suffix}",r)

    # wandb.log(
    #     {
    #         f"val_{suffix}/SRCC-{suffix}": s,
    #         f"val_{suffix}/PLCC-{suffix}": p,
    #         f"val_{suffix}/KRCC-{suffix}": k,
    #         f"val_{suffix}/RMSE-{suffix}": r,
    #     }
    # )

    del results, result  # , video, video_up
    torch.cuda.empty_cache()

    if save_model:
        state_dict = model.state_dict()

        if save_type == "head":
            head_state_dict = OrderedDict()
            for key, v in state_dict.items():
                if "finetune" in key or "swin" in key or 'blip.text_encoder' in key or "conv3d" in key or "fusion" in key:
                    head_state_dict[key] = v
            print("Following keys are saved (for head-only):", head_state_dict.keys())
            
            model_save_path=f"{model_save_dir}/{save_name}_{suffix}_{epoch}_finetuned.pth"
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            print(f"Saving the model to {model_save_path}")
            torch.save(
                {"state_dict": head_state_dict, "validation_results": best_,},
                model_save_path,
            )
        else:
            model_save_path=f"{model_save_dir}/{save_name}_{suffix}_{epoch}_finetuned.pth"
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
            torch.save(
                {"state_dict": state_dict, "validation_results": best_,},
                model_save_path,
            )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )


    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return best_s, best_p, best_k, best_r

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="/home/u202320081001008/share/IMC/T2VEval/t2vqa_motion.yml", help="the option file"
    )

    parser.add_argument(
        "-t", "--target_set", type=str, default="t2v", help="target_set"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    # wandb.init(project="A800", name=opt["name"], config=opt)

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## defining model and loading checkpoint

    bests_ = []

    if opt.get("split_seed", -1) > 0:
        num_splits = 1
    else:
        num_splits = 1

    print(opt["split_seed"])

    for split in range(num_splits):
        model = T2VQA(opt["model"]["args"]).to(device)

        if opt.get("split_seed", -1) > 0:
            opt["data"]["train"] = copy.deepcopy(opt["data"][args.target_set])
            opt["data"]["eval"] = copy.deepcopy(opt["data"][args.target_set])
            # opt["data"]["eval"]["args"]["phase"] = "test"

            split_duo = train_test_split(
                opt["data"][args.target_set]["args"]["data_prefix"],
                "/home/u202320081001008/share/IMC/test",
                opt["data"][args.target_set]["args"]["anno_file"],
                "/home/u202320081001008/share/IMC/T2VEval/dataset/test_data.txt",
                seed=opt["split_seed"] * (split + 1),
            )
            (
                opt["data"]["train"]["args"]["anno_file"],
                opt["data"]["eval"]["args"]["anno_file"],
            ) = split_duo

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = T2VDataset(
                    opt["data"][key]["args"]
                )
                train_datasets[key] = train_dataset
                print(len(train_dataset.video_infos))

        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt["batch_size"],
                num_workers=opt["num_workers"],
                shuffle=True,
            )

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

        param_groups = []

        for name, param in model.named_parameters():
            
            if "swin" in name or "finetune" in name or "slow" in name or "lora" in name:
                param_groups += [
                    {"params": param, "lr": opt["optimizer"]["lr"]}
                ]

        optimizer = torch.optim.AdamW(
            lr=opt["optimizer"]["lr"],
            params=param_groups,
            weight_decay=opt["optimizer"]["wd"],
        )

        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,)

        bests = {}
        bests_n = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
            bests_n[key] = -1, -1, -1, 1000
        
        model_ema = None

        for epoch in range(opt["num_epochs"]):
            print(f"End-to-end Epoch {epoch}:")
            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader,
                    model,
                    model_ema,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                )
            for key in val_loaders:
                bests[key] = inference_set(
                    epoch,
                    val_loaders[key],
                    model,
                    device,
                    bests[key],
                    save_model=opt["save_model"],
                    save_name=opt["name"] + "_t2vea" + "_head_" + args.target_set + f"_{split}",
                    suffix=key + "_s",
                    save_type="all",
                    model_save_dir='/home/u202320081001008/share/IMC/T2VEval/pretrained_weights_combine_test',
                )


        if opt["num_epochs"] >= 0:
            for key in val_loaders:
                print(
                    f"""For the end-to-end transfer process on {key} with {len(val_loaders[key])} videos,
                    the best validation accuracy of the model-s is as follows:
                    SROCC: {bests[key][0]:.4f}
                    PLCC:  {bests[key][1]:.4f}
                    KROCC: {bests[key][2]:.4f}
                    RMSE:  {bests[key][3]:.4f}."""
                )

        for key, value in dict(model.named_children()).items():
            if "finetune" in key or "swin" in key or "slow" in key or "lora" in key:
                for param in value.parameters():
                    param.requires_grad = True

        del model
        torch.cuda.empty_cache()
    
    # wandb.finish()
    print("All splits are done.")


if __name__ == "__main__":
    main()
