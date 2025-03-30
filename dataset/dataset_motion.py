import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import decord
import torchvision
from decord import VideoReader, cpu, gpu
import cv2
from PIL import Image

decord.bridge.set_bridge("torch")

def pack_pathway_output(frames, device):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """

    fast_pathway = frames
    # Perform temporal sampling from the fast pathway.
    slow_pathway = torch.index_select(
        frames,
        2,
        torch.linspace(
            0, frames.shape[2] - 1, frames.shape[2] // 4
        ).long(),
    )
    frame_list = [slow_pathway, fast_pathway]
    return frame_list


def get_resize_function(size_h, size_w, target_ratio=1, random_crop=False):
    if random_crop:
        return torchvision.transforms.RandomResizedCrop(
            (size_h, size_w), scale=(0.40, 1.0)
        )
    if target_ratio > 1:
        size_h = int(target_ratio * size_w)
        assert size_h > size_w
    elif target_ratio < 1:
        size_w = int(size_h / target_ratio)
        assert size_w > size_h
    return torchvision.transforms.Resize((size_h, size_w))

def get_resized_video(
        video, size_h=224, size_w=224, random_crop=False, arp=False, **kwargs,
):
    # 修改为直接除255
    video = video / 255.
    video = video.permute(1, 0, 2, 3)
    # c, t, w, h = video.size()
    resize_opt = get_resize_function(
        size_h, size_w, video.shape[-2] / video.shape[-1] if arp else 1, random_crop
    )
    video = resize_opt(video).permute(1, 0, 2, 3)
    return video  # (3, 16, 224, 224)

def get_spatial_fragments(
        video,
        fragments_h=7,
        fragments_w=7,
        fsize_h=32,
        fsize_w=32,
        aligned=8,
        nfrags=1,
        random=False,
        random_upsample=False,
        fallback_type="upsample",
        upsample=-1,
        **kwargs,
):
    if upsample > 0:
        old_h, old_w = video.shape[-2], video.shape[-1]
        if old_h >= old_w:
            w = upsample
            h = int(upsample * old_h / old_w)
        else:
            h = upsample
            w = int(upsample * old_w / old_h)

        video = get_resized_video(video, h, w)
    # 修改为直接除255
    # video = video / 255.
    size_h = fragments_h * fsize_h  # 调整为512
    size_w = fragments_w * fsize_w  # 调整为512
    ## video: [C,T,H,W]
    ## situation for images
    if video.shape[1] == 1:
        aligned = 1

    dur_t, res_h, res_w = video.shape[-3:]
    ratio = min(res_h / size_h, res_w / size_w)
    if fallback_type == "upsample" and ratio < 1:  # 是否需要上采样

        ovideo = video
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=1 / ratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    if random_upsample:
        randratio = random.random() * 0.5 + 1
        video = torch.nn.functional.interpolate(
            video / 255.0, scale_factor=randratio, mode="bilinear"
        )
        video = (video * 255.0).type_as(ovideo)

    assert dur_t % aligned == 0, "Please provide match vclip and align index"
    size = size_h, size_w

    ## make sure that sampling will not run out of the picture
    hgrids = torch.LongTensor(
        [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
    )
    wgrids = torch.LongTensor(
        [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
    )
    hlength, wlength = res_h // fragments_h, res_w // fragments_w

    if random:
        print("This part is deprecated. Please remind that.")
        if res_h > fsize_h:
            rnd_h = torch.randint(
                res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if res_w > fsize_w:
            rnd_w = torch.randint(
                res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
    else:
        if hlength > fsize_h:
            rnd_h = torch.randint(
                hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        if wlength > fsize_w:
            rnd_w = torch.randint(
                wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
            )
        else:
            rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

    target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
    # target_videos = []

    for i, hs in enumerate(hgrids):
        for j, ws in enumerate(wgrids):
            for t in range(dur_t // aligned):
                t_s, t_e = t * aligned, (t + 1) * aligned
                h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                if random:
                    h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                else:
                    h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                    w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                                                             :, t_s:t_e, h_so:h_eo, w_so:w_eo
                                                             ]
    # target_video = target_video.view(3, 8, 224, 224)
    return target_video

class VideoLoader:
    def __init__(self, video_path):
        from decord import VideoReader
        vr = VideoReader(video_path)
        self.len = len(vr)
        self.fps = vr.get_avg_fps()
        self.frame_ids = [int(self.fps * i) for i in range(int(len(vr) / self.fps))]
        self.frames = vr.get_batch(self.frame_indices).asnumpy()
    def __call__(self):
        return [Image.fromarray(self.frames[i]) for i in range(int(self.len / self.fps))]

class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode."""

        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode."""

        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading."""

        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)

    @staticmethod
    def uniform_sample(total_frames, num_samples):
        """
        均匀采样帧的索引。如果帧数不足，则补充最后一帧。
        Args:
            total_frames (int): 视频总帧数。
            num_samples (int): 需要采样的帧数。

        Returns:
            np.ndarray: 均匀采样的帧索引。
        """
        if total_frames >= num_samples:
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=np.int32)
        else:
            indices = np.arange(total_frames).tolist()
            # 如果帧数不足，补充最后一帧
            indices += [total_frames - 1] * (num_samples - total_frames)
            indices = np.array(indices, dtype=np.int32)
        return indices


class T2VDataset(Dataset):
    """Deformation of materials dataset."""

    def __init__(self, opt):
        
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.clip_len = opt["clip_len"]
        self.frame_interval = opt["frame_interval"]
        self.size = opt["size"]
        self.sampler = SampleFrames(self.clip_len, self.frame_interval)
        self.video_infos = []
        self.phase = opt["phase"]
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split("|")
                    filename, prompt, label = line_split
                    label = float(label)
                    filename = os.path.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, prompt=prompt, label=label))

    def __len__(self):
        
        return len(self.video_infos)

    def __getitem__(self, index):

        video_info = self.video_infos[index]
        filename = video_info["filename"]
        label = video_info["label"]
        vreader = VideoReader(filename)

        total_frames = len(vreader)

        # 均匀采样 8 帧
        frame_inds_8 = SampleFrames.uniform_sample(total_frames, 8)
        frame_dict_8 = {idx: vreader[idx] for idx in np.unique(frame_inds_8)}
        imgs_8 = [frame_dict_8[idx] for idx in frame_inds_8]

        video_8 = torch.stack(imgs_8, 0)
        video_8 = video_8.permute(3, 0, 1, 2) # c, t, w, h
        video_8 = torch.nn.functional.interpolate(video_8, size=(self.size, self.size))
        video_8 = ((video_8.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        # 均匀采样 32 帧
        frame_inds_32 = SampleFrames.uniform_sample(total_frames, 32)
        frame_dict_32 = {idx: vreader[idx] for idx in np.unique(frame_inds_32)}
        imgs_32 = [frame_dict_32[idx] for idx in frame_inds_32]

        video_32 = torch.stack(imgs_32, 0)
        video_32 = video_32.permute(3, 0, 1, 2)
        video_32 = torch.nn.functional.interpolate(video_32, size=(self.size, self.size))
        video_32 = ((video_32.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        # 默认采样策略
        frame_inds = self.sampler(total_frames, self.phase == "train")
        # frame_inds = SampleFrames.uniform_sample(total_frames, 8)
        frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}
        imgs = [frame_dict[idx] for idx in frame_inds]

        # 处理视频数据
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2) # c, t, w, h

        vfrag = get_spatial_fragments(video)
        vfrag = ((vfrag.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        resize_video = torch.nn.functional.interpolate(video, size=(self.size, self.size))
        resize_video = ((resize_video.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2) # c, t, w, h
        
        data = {
            "video": resize_video,
            "vfrag": vfrag,
            "slow": video_8,  # 均匀采样 8 帧的图像
            "fast": video_32,  # 均匀采样 32 帧的图像
            "gt_label": label,
            'filename': filename,
            "frame_inds": frame_inds
        }
        
        return data
