'''
Attention Layers 1-4
'''
import sys
import os
import datetime

import contextlib
from model.med import BertConfig, BertModel
from transformers import BertTokenizer, LlamaForCausalLM, LlamaTokenizer#, BertLMHeadModel
import argparse
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import copy
from pytorchvideo.models.hub import slowfast_r50
#from model.attention import Transformer3DModel
from model.blip import create_vit, init_tokenizer, load_checkpoint
from model.blip_pretrain import BLIP_Pretrain
from model.swin import swin_3d_tiny, SwinTransformer3D, SwinTransformer2D
from model.Qformer import BertLMHeadModel
from model.conv_backbone import convnext_3d_tiny 
from peft import LoraConfig, get_peft_model, TaskType
# load slowfast
from model.slowfast import slowfast
from model.slowfast_projector import slowfast_projector

from torch.nn import TransformerDecoderLayer, TransformerDecoder
from timm.models.vision_transformer import vit_base_patch16_224
from model.constants import DEFAULT_IMAGE_PATCH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
        
class T2VQA(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        med_config = args['med_config']
        image_size = args['image_size']
        embed_dim = args['embed_dim']
        llm_model = args['llm_model']

        self.blip = BLIP_Pretrain(image_size = image_size, vit = 'large', embed_dim = embed_dim, med_config = med_config)

        state_dict = torch.load(args['blip_weights'], map_location='cpu')
        self.blip.load_state_dict(state_dict["model"], strict=False)

        self.blip = self.blip.visual_encoder

        for name, param in self.blip.named_parameters():
            param.requires_grad = False

        self.slowfast = slowfast()
        self.slowfast_proj = slowfast_projector()

        for name, param in self.slowfast.named_parameters():
            param.requires_grad = True

        for name, param in self.slowfast_proj.named_parameters():
            param.requires_grad = True

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False)
        self.llm_model = LlamaForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16
        )

        # 确定Lora的参数，微调qkvo和gate单元
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj","v_proj"],
        )

        # 获得peft model
        lora_model = get_peft_model(self.llm_model, lora_config)
        lora_model.print_trainable_parameters()

        # 替换LLM
        self.llm_model = lora_model

        # 初始化 lora_A 和 lora_B 的权重
        for name, param in self.llm_model.named_parameters():
            if "lora_A" in name:
                nn.init.normal_(param, mean=0, std=1)
            elif "lora_B" in name:
                nn.init.zeros_(param)

        # layer_info = get_layer_parameters(lora_model)
        # save_layer_info_to_file(layer_info)  

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.finetune_llm_proj = nn.Linear(embed_dim, self.llm_model.config.hidden_size)

        self.finetune_proj = nn.Linear(
            768, self.llm_model.config.hidden_size
        )
        self.finetune_blip_proj = nn.Linear(self.blip.embed_dim, self.llm_model.config.hidden_size)

        # self.llm_model = self.llm_model.eval()
        # self.llm_model.train = disabled_train
        # 把这五个等级的text用tokenizer转成id
        self.excellent_idx, self.good_idx, self.fair_idx, self.poor_idx, self.bad_idx = self.llm_tokenizer(["excellent", "good","fair", "poor", "bad"])['input_ids']

        self.excellent_idx = self.excellent_idx[1]
        self.good_idx = self.good_idx[1]
        self.fair_idx = self.fair_idx[1]
        self.poor_idx = self.poor_idx[1]
        self.bad_idx = self.bad_idx[1]

        self.swin3d = swin_3d_tiny()
        state_dict = torch.load(args['swin_weights'], map_location='cpu')['state_dict']
        
        i_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "head" in key:
                continue
            if "cls" in key:
                tkey = key.replace("cls", "vqa")
            elif "backbone" in key:
                tkey = key.replace("backbone.", "")
                i_state_dict[tkey] = state_dict[key]
            else:
                i_state_dict[key] = state_dict[key]
            
        print(self.swin3d.load_state_dict(i_state_dict, strict=False))

        self.swin_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])


    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block


    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, data, caption, prompt):

        video = data['video']
        vfrag = data['vfrag']
        slow_fast = data['slowfast']

        # Insert the image token
        texts = prompt.split('<image>')
        input_id = []
        for i, text in enumerate(texts):
            text_tokens = self.llm_tokenizer([text], padding="longest", return_tensors="pt").to(video.device).input_ids.tolist()[0]
            input_id += text_tokens

            if i < len(texts) -1 :
                input_id += [IMAGE_TOKEN_INDEX]
        
        # Transfer token to embedding
        input_id = torch.tensor(input_id).to(video.device) 
        num_feature = (input_id == IMAGE_TOKEN_INDEX).sum()
        insert_ids = torch.where(input_id == IMAGE_TOKEN_INDEX)[0].tolist()
        indexs = [-1] + insert_ids + [input_id.shape[0]]
        _input_id = []
        for i in range(len(indexs) - 1):
            _input_id.append(input_id[indexs[i] + 1: indexs[i + 1]])
        
        cur_input_embeds = self.llm_model.get_input_embeddings()(torch.cat(_input_id)).unsqueeze(0).expand(video.shape[0], -1, -1) # 44, 4096
        split_sizes = [len(i) for i in _input_id]
        cur_input_embeds = torch.split(cur_input_embeds, split_sizes, dim=1) # 4个

        vision_features = []

        f = self.swin3d(vfrag) # (b, 768, 8, 7, 7)
        f = self.swin_avg_pool(f) # (b, 768, 1, 1, 1)
        f = f.view(f.size(0), -1) # (b, 768)
        f = f.unsqueeze(1) # (b, 1, 768)

        inputs_swin = f.expand(-1, 32, -1).to(video.device) # (b, 32, 768)
        inputs_swin = self.finetune_proj(inputs_swin)  # (b, 32, 4096)
        atts_swin = torch.ones(inputs_swin.size()[:-1], dtype=torch.long).to(video.device)

        f1 = self.slowfast(slow_fast)
        f1 = self.slowfast_proj(f1)
        inputs_motion = f1 # (b, 32, 4096)
        atts_motion = torch.ones(inputs_motion.size()[:-1], dtype=torch.long).to(video.device)

        inputs_blip = []
        # text = self.blip.tokenizer(caption, padding='max_length', truncation=True, max_length=50, return_tensors="pt").to(video.device)
        img_feats = []
        
        for j in range(video.size(2)):
            image = video[:,:,j,:,:]
            # image_embeds = self.blip.visual_encoder(image)[:, 0, :]
            image_embeds = self.blip(image)[:, 0, :]
            output = self.finetune_blip_proj(image_embeds) # intput [2, 768] output [2, 256]
            inputs_blip.append(output)
            img_feats.append(image_embeds)

        inputs_blip = torch.stack(inputs_blip, dim=1) #(b，8，4096)
        atts_blip = torch.ones(inputs_blip.size()[:-1], dtype=torch.long).to(video.device)

        vision_features.append(inputs_blip)
        vision_features.append(inputs_motion)
        vision_features.append(inputs_swin)
        
        num = 0
        inputs_llm = []
        for i, prompt_embedding in enumerate(cur_input_embeds):
            inputs_llm.append(prompt_embedding)
            if num < num_feature:
                inputs_llm.append(vision_features[num])
                num += 1

        inputs_embeds = torch.cat(inputs_llm, dim=1)
        attention_mask = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long).to(video.device)

        with self.maybe_autocast():

            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )

        output_logits = outputs.logits[:, -1]

        lexcellent, lgood, lfair, lpoor, lbad = output_logits[:, self.excellent_idx], output_logits[:, self.good_idx], output_logits[:, self.fair_idx], output_logits[:,self.poor_idx], output_logits[:, self.bad_idx]
        q_pred = (torch.stack([lexcellent, lgood, lfair, lpoor, lbad]) / 100).softmax(0)

        weights = self.weights.expand(-1, q_pred.shape[1]).to(video.device)
        q_pred = torch.mul(q_pred, weights)

        q_pred = torch.sum(q_pred, dim=0)

        return q_pred

if __name__=="__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="/media/imc3090x4/9ea39b40-da35-467c-b98d-803ea79ff646/AIGC_VQA/AIGC_VQA/Codes/T2VEA/t2vqa.yml", help="the option file"
    )

    parser.add_argument(
        "-t", "--target_set", type=str, default="t2v", help="target_set"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    model = T2VQA(opt["model"]["args"]).to(device)
    model.eval()

    caption = 'A random caption'
    # prompt = 'Please assess the quality of this image'
    
    # prompt = "The key frames of this video are:" + "\n" + DEFAULT_IMAGE_TOKEN + ". The motion feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". And the technical quality feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". Please assess the quality of this video."

    prompt = "The key frames of this video are:" + "\n" + DEFAULT_IMAGE_TOKEN + ". The motion feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". And the technical quality feature of the video is:" + "\n" + DEFAULT_IMAGE_TOKEN + ". Please assess the quality of this video."

    video = torch.randn(2, 3, 8, 224, 224).to(device)
    video1 = torch.randn(2, 3, 32, 224, 224).to(device)

    data = {}
    data['video'] = video
    data['vfrag'] = video
    data['slowfast'] = [video, video1]

    with torch.no_grad():
        output = model(data, caption, prompt)
    print(output)        
