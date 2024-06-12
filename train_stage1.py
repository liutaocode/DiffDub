from templates import pretrain_ffhq256_autoenc, train

if __name__ == '__main__':
    # we trained the model on 8 * v100s
    # some parameters are not used here.
    gpus = [0]
    nodes = 1
    conf = pretrain_ffhq256_autoenc()
    conf.latent_infer_path = None

    conf.name = 'hdtf'
    conf.N_frames = 1
    conf.cond_with_eye = True
    conf.data_name = 'HDTF'
    conf.base_dir="checkpoints"
    conf.img_data_path = 'assets/training_samples/raw_frames_25fps/'
    conf.pretrain = 'assets/checkpoints/diffae_ffhq_last.ckpt'
    conf.landmark_data_path = 'assets/training_samples/landmarks/'
    
    conf.batch_size = 4
    conf.sample_size = 2
    conf.img_size = 256
    conf.fp16 = True # disable it if you have enouth GPU memory
    
    train(conf, gpus=gpus, nodes=nodes)
