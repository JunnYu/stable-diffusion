Stable Diffusion Code
## 1. 安装环境
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```bash
cd stable_diffusion_code
# 可能里面的环境配置需要修改，如cuda版本什么的，没必要完全一致。
conda env create -f environment.yaml
conda activate ldm
```

## 2. 下载debug数据
```bash
# 下载
wget https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/part-00001.gz
# 移动到这个目录
mv part-00001.gz ./data_txt/laion400m_new
```

## 2. 开始训练
--gpus 6,7 表示使用6，7两个显卡。
当前这个配置文件 latent-diffusion/text2img_L32H1280_unet800M.yaml的 "batch=6，梯度累积=1"，梯度累积1更好比较速度。
```bash
# 1卡
python main.py --base configs/latent-diffusion/text2img_L32H1280_unet800M.yaml -t --gpus 0,
# 2卡
python main.py --base configs/latent-diffusion/text2img_L32H1280_unet800M.yaml -t --gpus 0,1
# 4卡
python main.py --base configs/latent-diffusion/text2img_L32H1280_unet800M.yaml -t --gpus 0,1,2,3
# 8卡
python main.py --base configs/latent-diffusion/text2img_L32H1280_unet800M.yaml -t --gpus 6,7
```
