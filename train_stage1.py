from templates import *
from templates_latent import *

if __name__ == '__main__':
    # 256 requires 8x v100s, in our case, on two nodes.
    gpus = [0,1,2,3,4,5,6,7]
    nodes = 1
    conf = pretrain_ffhq256_autoenc()
    conf.latent_infer_path = None

    conf.name = 'hdtf'
    conf.N_frames = 1
    conf.N_state_rand = True
    conf.visualized = True
    conf.cond_has_nose = True
    # Testing ids: 'WDA_BernieSanders_000','WRA_CoryGardner0_000','WDA_ChrisVanHollen0_000','WDA_RonWyden1_000','WDA_CarolynMaloney1_000','WRA_MikeEnzi_000','WDA_MikeDoyle_000', 'RD_Radio34_006','RD_Radio51_000','RD_Radio30_000','WRA_LamarAlexander0_000','RD_Radio35_000','WRA_BillCassidy0_000','RD_Radio34_007','RD_Radio56_000','RD_Radio34_008','RD_Radio34_005','WRA_PeterRoskam0_000','WRA_LamarAlexander_000','RD_Radio43_000','RD_Radio45_000','RD_Radio28_000','RD_Radio41_000','RD_Radio32_000','RD_Radio50_000',
    conf.meta_info = 'train_id.txt'
    conf.nose_mouth_chin_path = 'nose_lip_chin_total.json'
    conf.data_name = 'HDTF'
    conf.img_data_path = '/path/to/HDTF/raw_frames_25fps/'
    conf.base_dir="checkpoints_path"
    conf.landmark_data_path = '/path/to/hdtf/landmarks.json'
    
    conf.batch_size = 16
    conf.sample_size = 2
    conf.img_size = 256
    conf.fp16 = True
    
    train(conf, gpus=gpus, nodes=nodes)
