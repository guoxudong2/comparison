'''import pytorch_lightning as pl
import torch
import engine_pretraining

if __name__ == "__main__":
    # 1. 使用 load_from_checkpoint 加载模型权重
    checkpoint_path = "../checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"
    model = engine_pretraining.LitEEGPT.load_from_checkpoint(checkpoint_path)

    # 2. 进行推断（Inference）
    # 这里随意生成一个假数据 x。实际使用时需要根据你的模型输入格式准备数据。
    print('MY USE PRETRAIN')
    x = torch.randn(size=(1, 58, 256))  # 举例：batch_size=1, 通道数=58, 时间序列长度=256
    with torch.no_grad():
        model.eval()
        out = model(x)
    print("Inference result shape:", out.shape)

    # 或者你可以用 Trainer 进行测试、验证
    trainer = pl.Trainer(accelerator="cpu", max_epochs=1)'''
import torch
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("Current CUDA Device:", torch.cuda.current_device())
print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


