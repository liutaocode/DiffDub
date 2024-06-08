import os, random

videos_to_be_edited_lists = []
driven_audio_lists = []

samples_path = 'assets/samples/'

for dir_name in os.listdir(samples_path):
    if os.path.isdir(os.path.join(samples_path, dir_name)):
        videos_to_be_edited_lists.append(dir_name)
        driven_audio_lists.append(dir_name)

random.shuffle(videos_to_be_edited_lists)
random.shuffle(driven_audio_lists)

for video_to_be_edited, driven_audio in zip(videos_to_be_edited_lists, driven_audio_lists):
    cmd = f"python demo.py \
        --video_inference \
        --stage1_checkpoint_path 'assets/checkpoints/stage1_state_dict.ckpt' \
        --stage2_checkpoint_path 'assets/checkpoints/stage2_state_dict.ckpt' \
        --saved_path 'assets/samples/{video_to_be_edited}/' \
        --hubert_feat_path 'assets/samples/{driven_audio}/{driven_audio}.npy' \
        --wav_path 'assets/samples/{driven_audio}/{driven_audio}.wav' \
        --mp4_original_path 'assets/samples/{video_to_be_edited}/{video_to_be_edited}.mp4' \
        --denoising_step 20 \
        --saved_name {'few_shot_'+video_to_be_edited+'_#_'+driven_audio+'.mp4'}\
        --device 'cuda:0'"
    print(cmd)
    os.system(cmd)

    cmd = f"python demo.py \
        --one_shot \
        --video_inference \
        --stage1_checkpoint_path 'assets/checkpoints/stage1_state_dict.ckpt' \
        --stage2_checkpoint_path 'assets/checkpoints/stage2_state_dict.ckpt' \
        --saved_path 'assets/samples/{video_to_be_edited}/' \
        --hubert_feat_path 'assets/samples/{driven_audio}/{driven_audio}.npy' \
        --wav_path 'assets/samples/{driven_audio}/{driven_audio}.wav' \
        --mp4_original_path 'assets/samples/{video_to_be_edited}/{video_to_be_edited}.mp4' \
        --denoising_step 20 \
        --saved_name {'one_shot_'+video_to_be_edited+'_#_'+driven_audio+'.mp4'}\
        --device 'cuda:0'"
    print(cmd)
    os.system(cmd)


    cmd = f"python demo.py \
        --one_shot \
        --stage1_checkpoint_path 'assets/checkpoints/stage1_state_dict.ckpt' \
        --stage2_checkpoint_path 'assets/checkpoints/stage2_state_dict.ckpt' \
        --saved_path 'assets/samples/{video_to_be_edited}/' \
        --hubert_feat_path 'assets/samples/{driven_audio}/{driven_audio}.npy' \
        --wav_path 'assets/samples/{driven_audio}/{driven_audio}.wav' \
        --mp4_original_path 'assets/samples/{video_to_be_edited}/{video_to_be_edited}.mp4' \
        --denoising_step 20 \
        --saved_name {'one_shot_single_portrait_'+video_to_be_edited+'_#_'+driven_audio+'.mp4'}\
        --reference_image_path './assets/single_images/test001.png'\
        --device 'cuda:0'"
    print(cmd)
    os.system(cmd)