from .swin import SwinTransformer
from .subnets import ModelBox, FeatureAttentionModule, TaskSpecificSubnets, OutputModule

class SwinFaceTCfg:
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512

def build_model(model_name: str) -> ModelBox:
    if model_name == 'swin_t':
        cfg = SwinFaceTCfg()
        backbone = SwinTransformer(num_classes=512)
        fam = FeatureAttentionModule(
                in_chans=cfg.fam_in_chans, kernel_size=cfg.fam_kernel_size, 
                conv_shared=cfg.fam_conv_shared, conv_mode=cfg.fam_conv_mode, 
                channel_attention=cfg.fam_channel_attention, spatial_attention=cfg.fam_spatial_attention,
                pooling=cfg.fam_pooling, la_num_list=cfg.fam_la_num_list)
        tss = TaskSpecificSubnets()
        om = OutputModule()
        return ModelBox(backbone=backbone, fam=fam, tss=tss, om=om, feature=cfg.fam_feature)
    else:
        raise ValueError(f"Model {model_name} not recognized in swinface.")