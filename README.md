# AIGVEval Method

## 权重文件下载

通过网盘分享的文件：checkpoint
链接: https://pan.baidu.com/s/196mbB1fwW_0LNX-EE2HYjQ?pwd=n7f6 提取码: n7f6 
--来自百度网盘超级会员v6的分享

## 测试设置

如果你需要测试我们代码，请你参照本部分指示进行。我们在`test.py`中提供了测试代码的示例。如果你要在本地运行，你需要对代码和yml文件进行一些修改。

### 代码修改

```python
parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="/home/u202320081001008/share/IMC/T2VEval/t2vqa_test.yml", help="the option file"
    )
```

将其中的`/home/u202320081001008/share/IMC/T2VEval/t2vqa_test.yml`修改为你的配置文件路径。

### 配置文件修改

找到`t2vqa_test.yml`文件，修改其中的数据路径，包括`anno_file`和`data_prefix`：

```yaml
data:   
    t2v:
        type: T2VDataset
        args:
            phase: train 
            anno_file: /home/u202320081001008/share/IMC/T2VEval/dataset/test_data.txt
            data_prefix: /home/u202320081001008/share/IMC/test
            size: 224
            clip_len: 8
            frame_interval: 2
```

然后修改你的模型参数路径，包括`med_config`，`blip_weights`，`swin_weights`和`bert_weights`。

```yaml
model:
    args:
        med_config: /home/u202320081001008/share/IMC/T2VEval/configs/med_config.json
        image_size: 224
        nhead: 8
        dropout: 0.1
        nlayers: 6
        embed_dim: 256
        llm_model: /home/u202320081001008/share/IMC/T2VEval/checkpoint/vicuna-7b-v1.1
        blip_weights: /home/u202320081001008/share/IMC/T2VEval/checkpoint/model_large.pth
        swin_weights: /home/u202320081001008/share/IMC/T2VEval/checkpoint/swin_tiny_patch244_window877_kinetics400_1k.pth
        bert_weights: /home/u202320081001008/share/IMC/T2VEval/checkpoint/bert-base-uncased
```

修改我们获得的模型参数路径，用于测试：

```yaml
test_load_path: /home/u202320081001008/share/IMC/T2VEval/checkpoint/Final.pth
```

一切顺利的话，你可以得到一个`output.json`文档。

## 训练设置

如果你需要训练我们代码，请你参照本部分指示进行。我们在`train_motion_combine.py`中提供了训练代码的示例。你需要修改的代码和Yaml文件和测试部分一致。

# 联系我们

- 亓泽鲁，中国传媒大学，信息与通信工程学院，e-mail: theoneqi2001@cuc.edu.cn
- 史萍，中国传媒大学，信息与通信工程学院，e-mail: shiping@cuc.edu.cn
