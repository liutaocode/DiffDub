import cv2
import numpy as np
from templates import Rendering_Model, pretrain_ffhq256_autoenc
import os
import torchlm
import torch
from torchlm.tools import faceboxesv2
from torchlm.models import pipnet
from tqdm import tqdm
import numpy as np
from torch import Tensor
from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
from tqdm import tqdm
import random
import argparse
import shutil
from model.speech2latent import Seq2SeqModel
from PIL import Image
from torchvision import transforms

    
def get_surrounding_axis(matrix):
    x_lists, y_lists = [], []
    h,w = matrix.shape
    for y in range(h):
        for x in range(w) :
            if matrix[y][x] > 0:
                x_lists.append(x)
                y_lists.append(y)
    return min(x_lists), max(x_lists), min(y_lists), max(y_lists)

def to_Image(cv2_obj):
    img_array_ori = cv2_obj.astype(np.uint8)
    img_masked = Image.fromarray(img_array_ori)
    img = img_masked.convert('RGB')
    return img
    
def get_masked_image(ori_img, landmarks, image_size, landmark_index):
    # You can refer to https://github.com/liutaocode/talking_face_preprocessing/blob/master/asserts/landmarks.png to see the landmark indexing
    # landmark_index is 21 (covering the area below the eyebrows) or 30 (covering the area below the nose). This is the logic mentioned in the paper where the eyes are used as a guide to reduce jitter.
    lists = []
    lists.append([landmarks[6][0], landmarks[landmark_index][1]])
    lists.append([landmarks[10][0], landmarks[landmark_index][1]])
    lists.append(landmarks[10])
    lists.append([landmarks[8][0],landmarks[8][1]+5])
    lists.append(landmarks[6])
    
    mask = np.zeros_like(ori_img)
    contour = np.array(lists, dtype=np.int32)
    mask = cv2.fillPoly(mask, [contour], color=(255, 255, 255))
    kernel = np.ones((5,5),np.uint8)  
    mask = cv2.dilate(mask,kernel,iterations = 4)
    binary_matrix = np.where(mask != 0, 1, 0)
    
    x1,x2,y1,y2 = get_surrounding_axis(binary_matrix[:,:,0])

    cond = ori_img * binary_matrix
    cond = cond.astype('float32') 
    cond = cond[y1:y2,x1:x2]

    img_cond = cv2.resize(cond, (image_size, image_size), interpolation = cv2.INTER_NEAREST) 
    img_cond = img_cond.astype(np.uint8)
    return ori_img, binary_matrix, img_cond

def get_resized_image(path, image_size):
    cv2_image = cv2.imread(path)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    cv2_image_resized = cv2.resize(cv2_image, (image_size, image_size))
    
    return cv2_image_resized, to_Image(cv2_image_resized)

def frames_to_video(input_path, audio_path, output_path, fps=25):
    image_files = [os.path.join(input_path, img) for img in sorted(os.listdir(input_path))]
    clips = [ImageClip(m).set_duration(1/fps) for m in image_files]
    video = concatenate_videoclips(clips, method="compose")

    audio = AudioFileClip(audio_path)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')

def extract_random_frame_from_video(mp4_original_path, saved_to_gt_frame_path):
    cap = cv2.VideoCapture(mp4_original_path)
    assert cap.isOpened()

    frame_filenames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(saved_to_gt_frame_path, f"{frame_count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            frame_filenames.append(frame_filename)
            frame_count += 1
        else:
            break
    cap.release()

    # TODO Randomly select a reference image from the to-be-modified videos, you can modify the selection logic as per your needs
    assert len(frame_filenames) > 0
    image_path = random.choice(frame_filenames)
    return image_path, frame_filenames

def get_formated_image(image_path, img_size):
    landmarks = get_landmark(image_path, img_size)
    cv2_image = cv2.imread(image_path)
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    cv2_image_resized = cv2.resize(cv2_image, (img_size, img_size))
    return cv2_image_resized, landmarks

def get_landmark(image_path, img_size):
    frame = cv2.imread(image_path)
    landmarks, _ = torchlm.runtime.forward(frame)

    # TODO This adjusts the coordinates of the landmarks to the img_size dimensions. Note that the image must be square.
    landmarks = landmarks / frame.shape[0] * img_size
    assert len(landmarks) == 1

    # TODO Here we use the first landmark. The test videos we provide are all single-person, and we have not considered the logic for multiple people.
    # If your scenario involves multiple people, you need to modify the code here.
    landmarks = landmarks[0] 
    return landmarks

def demo_pipeline(one_shot,
    video_inference,
    stage1_checkpoint_path,
    stage2_checkpoint_path,
    saved_path,
    hubert_feat_path,
    wav_path,
    mp4_original_path,
    denoising_step,
    reference_image_path,
    saved_name,
    device):

    saved_to_mp4_path = os.path.join(saved_path, saved_name)
    saved_to_pred_frame_path = os.path.join(saved_path, 'predicted_frames')
    saved_to_gt_frame_path = os.path.join(saved_path, 'gt_frames')

    os.makedirs(saved_path, exist_ok=True)
    os.makedirs(saved_to_pred_frame_path, exist_ok=True)
    os.makedirs(saved_to_gt_frame_path, exist_ok=True)


    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    if video_inference:
        reference_image_path, total_image_path = extract_random_frame_from_video(
            mp4_original_path, saved_to_gt_frame_path
        )
    else:
        if reference_image_path is None or not os.path.exists(reference_image_path):
            print(f'Error: {reference_image_path} does not exist.')
            exit(-1)

    # Step 1 : Lip Reference Condition generation
    conf = pretrain_ffhq256_autoenc()
    conf.img_size = 256
    conf.N_frames = 1
    conf.N_state_rand = True
    conf.visualized = False
    conf.cond_has_nose = True
    model = Rendering_Model(conf)
    # loadParameters(model, stage1_checkpoint_path)
    model.load_state_dict(torch.load(stage1_checkpoint_path, map_location='cpu')['state_dict'], strict=True)
    model.ema_model.eval()
    model.ema_model.to(device);
    xT = torch.randn((1, 3, 256, 256)).to(device)

    # Landmark Extraction Model Loading
    torchlm.runtime.bind(faceboxesv2(device=device))  
    torchlm.runtime.bind(
        pipnet(backbone="resnet18", pretrained=True,
                num_nb=10, num_lms=68, net_stride=32, input_size=256,
                meanface_type="300w", map_location=device, checkpoint=None)
    ) 

    seq2seq = Seq2SeqModel()

    seq2seq.load_state_dict(torch.load(stage2_checkpoint_path, map_location='cpu')['state_dict'], strict=True)
    seq2seq = seq2seq.seq2seq
    seq2seq.eval()
    seq2seq.to(device);


    if one_shot:
        print('One shot mode')
        # Renference Lip Latent Generation
        img_condition, img_landamrk = get_formated_image(reference_image_path, conf.img_size)
        _, _, lip_img = get_masked_image(img_condition, img_landamrk, conf.img_size, landmark_index=21)
        lip_img_norm = norm_transform(to_Image(lip_img)).unsqueeze(0).to(device)
        reference_lip_latent = model.encode(lip_img_norm)
        reference_lip_latents = reference_lip_latent.unsqueeze(1).repeat(1, 75, 1) 
    else:
        reference_lip_latent_lists = []
        while True:
            for reference_image_path in total_image_path:
                img_condition, img_landamrk = get_formated_image(reference_image_path, conf.img_size)
                _, _, lip_img = get_masked_image(img_condition, img_landamrk, conf.img_size, landmark_index=21)
                lip_img_norm = norm_transform(to_Image(lip_img)).unsqueeze(0).to(device)
                reference_lip_latent = model.encode(lip_img_norm)
                reference_lip_latent_lists.append(reference_lip_latent.unsqueeze(1))
                if len(reference_lip_latent_lists) == 75:
                    break
            if len(reference_lip_latent_lists) == 75:
                    break
        reference_lip_latents = torch.cat(reference_lip_latent_lists, dim=1)
        print('Few shot mode')

    # ==== Step 2 : Speech to Lip Latent ====

    from_npy = np.load(hubert_feat_path, mmap_mode='r')
    audio_feats = Tensor(from_npy).float().to(device).unsqueeze(0)

    predictions = seq2seq(audio_feats, reference_lip_latents)

    # ==== Step 3 : Lip Latent Rendering ====
    max_len = predictions.shape[1]
    if video_inference:
        # TODO Ensure the video duration is longer than the driven audio
        max_len = min(predictions.shape[1], len(total_image_path))
         

    for i in tqdm(range(max_len), desc="Processing frames"):

        if video_inference:
            # Modify lips frame by frame
            current_image_path = total_image_path[i]
        else:
            # Only modify one frame input by the user
            current_image_path = reference_image_path

        landmarks = get_landmark(current_image_path, conf.img_size)

        # Frame-by-frame lip latent
        lip_latent = predictions[:,i,:]

        # Generate the image with the mouth masked
        cv2_image_resized, _ = get_resized_image(current_image_path, conf.img_size)
        image_array, mouth_masked, _ = get_masked_image(cv2_image_resized, landmarks, conf.img_size, landmark_index=30)
        img_with_mouth_masked = norm_transform(to_Image(image_array)).unsqueeze(0).to(device)

        # Reference frame generation
        _, ref_img = get_resized_image(reference_image_path, conf.img_size)
        img_ref_tensor = norm_transform(ref_img).unsqueeze(0).to(device)

        # TODO Adding batch processing will speed up generation
        # Rendering images frame by frame (this is the slowest part of the entire model)
        pred = model.render(
            xT,
            img_with_mouth_masked,
            img_ref_tensor,
            lip_latent,
            T=denoising_step,
            mask=torch.Tensor(1 - mouth_masked).unsqueeze(0).to(device)
        )
        
        # Image rendering and saving
        image = pred[0].detach().permute(1,2,0).cpu().numpy() * 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(saved_to_pred_frame_path, "%05d.png"%(i)), image.astype(np.uint8))

    frames_to_video(saved_to_pred_frame_path, wav_path, saved_to_mp4_path)

    shutil.rmtree(saved_to_pred_frame_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip Latent Rendering Parameters")
    parser.add_argument('--one_shot', action='store_true', help='Use a single image as the reference (one-shot). If not set, multiple images will be used as references (few-shot).')
    parser.add_argument('--video_inference', action='store_true', help='Flag for video inference')
    parser.add_argument('--stage1_checkpoint_path', type=str, default='assets/checkpoints/stage1_state_dict.ckpt', help='Path to the stage 1 checkpoint')
    parser.add_argument('--stage2_checkpoint_path', type=str, default='assets/checkpoints/stage2_state_dict.ckpt', help='Path to the stage 2 checkpoint')
    parser.add_argument('--saved_path', type=str, default='./assets/samples/RD_Radio51_000_25fps/', help='Path to save the results')
    parser.add_argument('--hubert_feat_path', type=str, default='./assets/samples/WRA_LamarAlexander0_000_25fps/WRA_LamarAlexander0_000_25fps.npy', help='Path to the Hubert features')
    parser.add_argument('--wav_path', type=str, default='./assets/samples/WRA_LamarAlexander0_000_25fps/WRA_LamarAlexander0_000_25fps.wav', help='Path to the diven WAV file')
    parser.add_argument('--mp4_original_path', type=str, default='./assets/samples/RD_Radio51_000_25fps/RD_Radio51_000_25fps.mp4', help='Path to the original MP4 file(to be driven)')
    parser.add_argument('--denoising_step', type=int, default=20, help='Number of denoising steps')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for computation')
    parser.add_argument('--saved_name', type=str, default="predition.mp4", help='Name of the generated video (must include mp4 extension)')
    parser.add_argument('--reference_image_path', type=str, default="./assets/single_images/test001.png", help='When in one-shot mode, a fixed frame can be set as the object to be modified')
    args = parser.parse_args()


demo_pipeline(
    one_shot=args.one_shot,
    video_inference=args.video_inference,
    stage1_checkpoint_path=args.stage1_checkpoint_path,
    stage2_checkpoint_path=args.stage2_checkpoint_path,
    saved_path=args.saved_path,
    hubert_feat_path=args.hubert_feat_path,
    wav_path=args.wav_path,
    mp4_original_path=args.mp4_original_path,
    denoising_step=args.denoising_step,
    reference_image_path=args.reference_image_path,
    saved_name=args.saved_name,
    device=args.device
)
