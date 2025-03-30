'''
Attention Layers 1-4
'''
import sys
import os
import datetime

# 添加模型所在目录到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '/media/imc3090x4/9ea39b40-da35-467c-b98d-803ea79ff646/AIGC_VQA/AIGC_VQA/Codes/T2VEA/model'))

import contextlib
from med import BertConfig, BertModel
from transformers import BertTokenizer, LlamaForCausalLM, LlamaTokenizer#, BertLMHeadModel
import argparse
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import copy

#from model.attention import Transformer3DModel
from blip import create_vit, init_tokenizer, load_checkpoint
from blip_pretrain import BLIP_Pretrain
from swin import swin_3d_tiny, SwinTransformer3D, SwinTransformer2D
from Qformer import BertLMHeadModel
from conv_backbone import convnext_3d_tiny 
from peft import LoraConfig, get_peft_model, TaskType

from torch.nn import TransformerDecoderLayer, TransformerDecoder
from timm.models.vision_transformer import vit_base_patch16_224


def get_layer_parameters(model):
    layer_params = {}
    
    # 遍历所有模型参数
    for name, param in model.named_parameters():
        # 解析层级结构
        parts = name.split('.')
        
        # 处理不同层类型
        if parts[0] == 'model':
            if parts[1] == 'embed_tokens':
                layer_name = 'embed_tokens'
            elif parts[1] == 'layers':
                layer_num = parts[2]
                layer_type = parts[3]
                layer_name = f"layer_{layer_num}_{layer_type}"
            elif parts[1] == 'norm':
                layer_name = 'final_norm'
        elif parts[0] == 'lm_head':
            layer_name = 'lm_head'
        else:
            layer_name = 'other'

        # 创建层级结构
        if layer_name not in layer_params:
            layer_params[layer_name] = []
            
        # 记录参数信息
        param_info = {
            'name': name,
            'shape': tuple(param.shape),
            'dtype': str(param.dtype).replace("torch.", ""),
            'requires_grad': param.requires_grad
        }
        layer_params[layer_name].append(param_info)

    return layer_params

def save_layer_info_to_file(layer_info, filename="/media/imc3090x4/9ea39b40-da35-467c-b98d-803ea79ff646/AIGC_VQA/AIGC_VQA/Codes/T2VEA/lora_model_layers_info.txt"):
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入文件头
        f.write("="*40 + "\n")
        f.write(f"LLaMA Model Layer Parameters Report\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*40 + "\n\n")

        # 遍历各层
        for layer_name, params in layer_info.items():
            f.write(f"\n=== Layer: {layer_name} ===\n")
            for idx, param in enumerate(params, 1):
                f.write(f"Parameter {idx}:\n")
                f.write(f"Name: {param['name']}\n")
                f.write(f"Shape: {param['shape']}\n")
                f.write(f"Type: {param['dtype']}\n")
                f.write(f"Requires Grad: {param['requires_grad']}\n\n")

        # 添加统计信息
        f.write("\n" + "="*40 + "\n")
        f.write("Parameter Statistics:\n")
        total_params = 0
        for layer_name, params in layer_info.items():
            layer_params = sum(math.prod(p['shape']) for p in params)
            total_params += layer_params
            f.write(f"{layer_name}: {layer_params/1e6:.2f}M parameters\n")
        f.write("-"*40 + "\n")
        f.write(f"Total Parameters: {total_params/1e9:.2f}B\n")
        f.write("="*40 + "\n")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=768, d_k=64, d_v=64, d_q=64, n_heads=12):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_q
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q.contiguous().view(-1, self.d_model)).view(batch_size, -1, self.n_heads, self.d_q).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K.contiguous().view(-1, self.d_model)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V.contiguous().view(-1, self.d_model)).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        att = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        att = F.softmax(att, dim=-1) # (b, 12, 32, 32)
        att = torch.matmul(att, V) # (b, 12, 32, 64)
        att = att.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k) # (b, 32, 768)
        att = self.fc(att)
        att = nn.LayerNorm(768).to(input_Q.device)(att + residual)
        return att

class Fusion_Block(nn.Module):
    def __init__(self, d_model=768, d_k=64, d_v=64, d_q=64, n_heads=12):
        super(Fusion_Block, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_q
        self.heads = n_heads
        self.sa = MultiHeadAttention(d_model, d_k, d_v, d_q, n_heads)
        self.ca = MultiHeadAttention(d_model, d_k, d_v, d_q, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model),
        )

    def forward(self, input_1, input_2):
        self_attn = self.sa(input_1, input_1, input_1) # Self Attention Only
        cross_attn = self.ca(self_attn, input_2, input_2) # Cross Attention Only
        residual = cross_attn
        output = nn.LayerNorm(self.d_model).to(input_1.device)(self.feed_forward(cross_attn) + residual)
        return output

class Fusion_Module(nn.Module):
    def __init__(self, num_layers=1, d_model=768, d_k=64, d_v=64, d_q=64, n_heads=12):
        super(Fusion_Module, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.d_q = d_q
        self.heads = n_heads
        self.Scattn = nn.ModuleList([Fusion_Block(d_model, d_k, d_v, d_q, n_heads) for _ in range(num_layers)])
    def forward(self, input_1, input_2):
        for i in range(len(self.Scattn)):
            if i == 0:
                output = self.Scattn[i](input_1, input_2)
            else:
                output = self.Scattn[i](output, input_2)
        return output
        

class T2VQA(nn.Module):
    def __init__(self,
                 args):
        super().__init__()

        med_config = args['med_config']
        image_size = args['image_size']
        embed_dim = args['embed_dim']
        llm_model = args['llm_model']

        self.blip = BLIP_Pretrain(image_size = image_size, vit = 'large', embed_dim = embed_dim, med_config = med_config)
        state_dict = torch.load(args['blip_weights'], map_location='cpu')['state_dict']
        # state_dict = torch.load(args['blip_weights'], map_location='cpu')
        # self.blip.load_state_dict(state_dict["model"], strict=False)
        blip_state_dict = OrderedDict()
        # 遍历state_dict，去掉blip.
        for key in state_dict.keys():
            if "blip." in key:
                tkey = key.replace("blip.", "")
                blip_state_dict[tkey] = state_dict[key]
                
        print(self.blip.load_state_dict(blip_state_dict, strict=False))

        for name, param in self.blip.named_parameters():
            if ("text_encoder" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.finetune_text_proj = nn.Linear(self.blip.text_encoder.config.hidden_size, embed_dim)

        encoder_config = BertConfig.from_pretrained(args['bert_weights'])
        encoder_config.encoder_width = embed_dim
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = 32

        self.finetune_Qformer = BertLMHeadModel.from_pretrained(
            args['bert_weights'], config=encoder_config
        )

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
            lora_dropout=0.1,
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
                print(f"Initialized {name} with normal distribution (mean=0, std=1)")
            elif "lora_B" in name:
                nn.init.zeros_(param)
                print(f"Initialized {name} with zeros")

        # layer_info = get_layer_parameters(lora_model)
        # save_layer_info_to_file(layer_info)  

        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.finetune_llm_proj = nn.Linear(embed_dim, self.llm_model.config.hidden_size)

        self.finetune_proj = nn.Linear(
            self.finetune_Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )

        # self.llm_model = self.llm_model.eval()
        # self.llm_model.train = disabled_train
        # 把这五个等级的text用tokenizer转成id
        self.excellent_idx, self.good_idx, self.fair_idx, self.poor_idx, self.bad_idx = self.llm_tokenizer(["excellent", "good","fair", "poor", "bad"])['input_ids']

        self.excellent_idx = self.excellent_idx[1]
        self.good_idx = self.good_idx[1]
        self.fair_idx = self.fair_idx[1]
        self.poor_idx = self.poor_idx[1]
        self.bad_idx = self.bad_idx[1]
        # 现在尝试把问题建模为13分类
        
        # self.conv3d = convnext_3d_tiny() ## Conv3D backbone
        # conv_checkpoint = torch.load(args['conv_weights'], map_location='cpu')
        # print(self.conv3d.load_state_dict(conv_checkpoint, strict=False))
        self.conv3d = convnext_3d_tiny(pretrained=False,checkpoint=None) ## Conv3D backbone
        conv_checkpoint = torch.load(args['conv_weights'], map_location='cpu')['state_dict']
        conv_checkpoint = {k.replace("conv3d.", ""): v for k, v in conv_checkpoint.items()}
        print(self.conv3d.load_state_dict(conv_checkpoint, strict=False))

        # self.swin3d = swin_3d_tiny()
        # state_dict = torch.load(args['swin_weights'], map_location='cpu')
        # state_dict = state_dict['state_dict']
        self.swin3d = swin_3d_tiny()
        state_dict = torch.load(args['swin_weights'], map_location='cpu')
        state_dict = state_dict['state_dict']
        i_state_dict = {k.replace("swin3d.", ""): v for k, v in state_dict.items()}
        
        # i_state_dict = OrderedDict()
        # for key in state_dict.keys():
        #     if "head" in key:
        #         continue
        #     if "cls" in key:
        #         tkey = key.replace("cls", "vqa")
        #     elif "backbone" in key:
        #         tkey = key.replace("backbone.", "")
        #         i_state_dict[tkey] = state_dict[key]
        #     else:
        #         i_state_dict[key] = state_dict[key]
            
        print(self.swin3d.load_state_dict(i_state_dict, strict=False))
        
        self.swin_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.weights = torch.Tensor([[1], [2], [3], [4], [5]])
        self.fusion_module = Fusion_Module()

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

        f = self.swin3d(vfrag) # (b, 768, 8, 7, 7)
        f = self.swin_avg_pool(f) # (b, 768, 1, 1, 1)
        f = f.view(f.size(0), -1) # (b, 768)
        f = f.unsqueeze(1) # (b, 1, 768)
        inputs_swin = f.expand(-1, 32, -1).to(video.device) # (b, 32, 768)
        atts_swin = torch.ones(inputs_swin.size()[:-1], dtype=torch.long).to(video.device)

        f = self.conv3d(video) # (b, 768, 8, 7, 7)
        f = self.swin_avg_pool(f) # (b, 768, 1, 1, 1)
        f = f.view(f.size(0), -1) # (b, 768)
        f = f.unsqueeze(1) # (b, 1, 768)
        inputs_conv = f.expand(-1, 32, -1).to(video.device) # (b, 8, 768)
        atts_conv = torch.ones(inputs_conv.size()[:-1], dtype=torch.long).to(video.device)

        # Fusion conv and swin
        inputs_swin = self.fusion_module(inputs_swin, inputs_conv)

        inputs_llm = []

        text = self.blip.tokenizer(caption, padding='max_length', truncation=True, max_length=50, return_tensors="pt").to(video.device)

        img_feats = []
        
        for j in range(video.size(2)):
            image = video[:,:,j,:,:]

            image_embeds = self.blip.visual_encoder(image)


            image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(video.device)
            output = self.blip.text_encoder(text.input_ids,
                                                attention_mask = text.attention_mask,
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,
                                                return_dict = True,
                                            )

            output = self.finetune_text_proj(output.last_hidden_state[:,0,:])


            inputs_llm.append(output)
            img_feats.append(image_embeds)

        img_feats = torch.stack(img_feats, dim=1)
        image_atts = torch.ones(img_feats.size()[:-1],dtype=torch.long).to(video.device)

        inputs_llm = torch.stack(inputs_llm, dim=1) #（4，8，256）
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(video.device)

        all_inputs = self.finetune_Qformer.bert(
                query_embeds=inputs_swin,
                attention_mask=atts_swin,
                encoder_hidden_states=inputs_llm,
                encoder_attention_mask=atts_llm,
                return_dict=True,
            )

        inputs_llm = self.finetune_proj(all_inputs.last_hidden_state[:,:inputs_swin.size(1),:])
   

        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(video.device)


        llm_tokens = self.llm_tokenizer(
            [prompt] * video.size(0),
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        
        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)

            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

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
    device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="/media/imc3090x4/9ea39b40-da35-467c-b98d-803ea79ff646/AIGC_VQA/AIGC_VQA/Codes/MPTVA/t2vqa.yml", help="the option file"
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
    prompt = 'Please assess the quality of this image'
    video = torch.randn(2, 3, 8, 224, 224).to(device)

    with torch.no_grad():
        output = model(video, caption, prompt)
    print(output)

