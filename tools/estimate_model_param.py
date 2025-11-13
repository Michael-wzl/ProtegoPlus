import torch
from thop import profile

from protego.FacialRecognition import FR, BASIC_POOL

if __name__ == "__main__":
    model_names = [n for n in BASIC_POOL if n != 'ir50_adaface_casia']
    device = torch.device('cpu')
    all_params = []
    for name in model_names:
        print("#"*50)
        print(f"Model: {name}")
        print("#"*50)
        model = FR(model_name=name, device=device)
        inp = torch.randn(1, 3, 224, 224).to(device)
        inp = model.preprocess(inp)
        flops, params = profile(model.fr_model.model, inputs=(inp,))
        print(f"  Params: {params:,}")
        print(f"  FLOPs: {flops:,}")
        all_params.append(params)
    print(f"The total params of all FR models: {sum(all_params):,}")
