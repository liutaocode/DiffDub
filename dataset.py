import os, cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import albumentations as A
import numpy as np
from tqdm import tqdm

class HDTFDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        landmark_data_path,
        N_frames=1,
        cond_with_eye=True,
    ):
        super().__init__()
        
        self.png_image_folder = folder
        self.image_size = image_size
        self.landmark_data_path = landmark_data_path
        self.N_frames = N_frames
        self.cond_with_eye = cond_with_eye

        self.test_ids = set(['WDA_BernieSanders_000','WRA_CoryGardner0_000','WDA_ChrisVanHollen0_000','WDA_RonWyden1_000','WDA_CarolynMaloney1_000','WRA_MikeEnzi_000','WDA_MikeDoyle_000', 'RD_Radio34_006','RD_Radio51_000','RD_Radio30_000','WRA_LamarAlexander0_000','RD_Radio35_000','WRA_BillCassidy0_000','RD_Radio34_007','RD_Radio56_000','RD_Radio34_008','RD_Radio34_005','WRA_PeterRoskam0_000','WRA_LamarAlexander_000','RD_Radio43_000','RD_Radio45_000','RD_Radio28_000','RD_Radio41_000','RD_Radio32_000','RD_Radio50_000'])
        self.training_items = []
        self.video_clip_dict = dict()

       
        for video_clip_name in tqdm(os.listdir(self.png_image_folder)):
            if video_clip_name.replace('_25fps', '') in self.test_ids:
                continue

            self.video_clip_dict[video_clip_name] = []
            for image_name in os.listdir(os.path.join(self.png_image_folder, video_clip_name)):
                image_path = os.path.join(self.png_image_folder, video_clip_name, image_name)
                landmark_path = os.path.join(self.landmark_data_path, video_clip_name, f"{os.path.splitext(image_name)[0]}.txt")
                self.training_items.append((video_clip_name, image_path, landmark_path))
                self.video_clip_dict[video_clip_name].append(image_path)

        self.normlize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # image augmentations for the condition
        self.condtion_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            A.Blur(p=0.1),
            A.ShiftScaleRotate(
                rotate_limit=20,
                scale_limit=0.1,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                shift_limit_y=[0.1, 0.2],
                shift_limit_x=[0.1, 0.2]
            )
        ])

    def to_Image(self, cv2_obj):
        img_array_ori = cv2_obj.astype(np.uint8)
        img_masked = Image.fromarray(img_array_ori)
        img = img_masked.convert('RGB')
        return img
    
    def get_surrounding_axis(self, matrix):
        x_lists, y_lists = [], []
        h,w = matrix.shape
        for y in range(h):
            for x in range(w) :
                if matrix[y][x] > 0:
                    x_lists.append(x)
                    y_lists.append(y)
        return min(x_lists), max(x_lists), min(y_lists), max(y_lists)
            
    # mouth area (with nose) only
    def get_input_aug(self, ori_img, landmarks):

        lists = []
        lists.append([landmarks[6][0], landmarks[30][1]]) 
        lists.append([landmarks[10][0], landmarks[30][1]]) 
        lists.append(landmarks[10])
        lists.append([landmarks[8][0],landmarks[8][1]+5])
        lists.append(landmarks[6])
        
        
        mask = np.zeros_like(ori_img)
        contour = np.array(lists, dtype=np.int32)
        mask = cv2.fillPoly(mask, [contour], color=(255, 255, 255))
        kernel = np.ones((5,5),np.uint8)  
        mask = cv2.dilate(mask,kernel,iterations = 4)
        binary_matrix = np.where(mask != 0, 1, 0)

        x1,x2,y1,y2 = self.get_surrounding_axis(binary_matrix[:,:,0])
   
        cond = ori_img*binary_matrix
        cond = cond.astype('float32') 
        cond = cond[y1:y2,x1:x2]

        img_cond = cv2.resize(cond, (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST) 
        img_cond = img_cond.astype(np.uint8)
        cv2_image_aug = self.condtion_transform(image=img_cond)

        return ori_img, 1 - binary_matrix, cv2_image_aug['image']
    
    # mouth area (with nose) + eye guidance
    def get_input_condtion(self, ori_img, landmarks):

        lists = []
        lists.append([landmarks[6][0], landmarks[21][1]]) 
        lists.append([landmarks[10][0], landmarks[21][1]]) 
        lists.append(landmarks[10])
        lists.append([landmarks[8][0],landmarks[8][1]+5])
        lists.append(landmarks[6])
        
        
        mask = np.zeros_like(ori_img)
        contour = np.array(lists, dtype=np.int32)
        mask = cv2.fillPoly(mask, [contour], color=(255, 255, 255))
        kernel = np.ones((5,5),np.uint8)  
        mask = cv2.dilate(mask,kernel,iterations = 4)
        binary_matrix = np.where(mask != 0, 1, 0)
        
        x1,x2,y1,y2 = self.get_surrounding_axis(binary_matrix[:,:,0])
   
        cond = ori_img*binary_matrix
        cond = cond.astype('float32') 
        cond = cond[y1:y2,x1:x2]

        img_cond = cv2.resize(cond, (self.image_size, self.image_size), interpolation = cv2.INTER_NEAREST) 
        img_cond = img_cond.astype(np.uint8)
        cv2_image_aug = self.condtion_transform(image=img_cond)

        return cv2_image_aug['image']
    
    def scale_to_specific_resolution(self, origin_axis, original_height, original_width, to_reso):
        x, y = origin_axis
        to_x, to_y = x/original_width*to_reso, y/original_height*to_reso
        to_x = int(to_x)
        to_y = int(to_y)

        return (to_x, to_y)
        
    def __len__(self):
        return len(self.training_items)
        
    def get_landmarks_from_path(self, height, width, landmark_path):
        
        landmarks = []
        for line in open(landmark_path):
            items = line.replace("\n", "").split()
            out_points = self.scale_to_specific_resolution((int(items[0]), int(items[1])), height, width, self.image_size)
            landmarks.append(out_points)
            
        assert len(landmarks) == 68
        return landmarks

    def __getitem__(self, index):
        clip_name, image_path, landmark_path  = self.training_items[index]

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        landmarks = self.get_landmarks_from_path(height, width, landmark_path)

        # random pick reference frame
        black_list_set = set()
        black_list_set.add(image_path)
        ref_lists = []
        if self.N_frames != 0:
            for _ in range(self.N_frames): # self.N_frames is a tuple
                while True:
                    randindex = random.randint(0, len(self.video_clip_dict[clip_name])-1)
                    ref_image_path = self.video_clip_dict[clip_name][randindex]
                    if ref_image_path not in black_list_set:
                        black_list_set.add(ref_image_path)
                        
                        ref_img = self.get_ref_image(ref_image_path)
                        ref_img = self.normlize(ref_img)
                        ref_lists.append(ref_img)
                        break
            
        ref_img = np.concatenate(ref_lists, axis=0)
        
        img, con_img, masked_rect, _ = self.get_ori_cond_image(image_path, landmarks)
        
        img = self.normlize(img)
        con_img = self.normlize(con_img)

        if ref_img is None:
            return {'img': img, 'index': index, 'condtion': con_img, 'mouth_masked': masked_rect}
        else:
            return {'img': img, 'index': index, 'condtion': con_img, 'ref': ref_img, 'mouth_masked': masked_rect}

    def get_ref_image(self, path):
        cv2_image = cv2.imread(path)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2_image_resized = cv2.resize(cv2_image, (self.image_size, self.image_size))
        
        ref_img = cv2_image_resized
        
        return self.to_Image(ref_img)
            
    def get_ori_cond_image(self, path, landmarks):
        cv2_image = cv2.imread(path)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        cv2_image_resized = cv2.resize(cv2_image, (self.image_size, self.image_size))

        image_array, masked_rect, img_condition = self.get_input_aug(cv2_image_resized, landmarks)
        
        if self.cond_with_eye:
            # add eye as the guidance 
            img_condition = self.get_input_condtion(cv2_image_resized, landmarks)

        return self.to_Image(image_array), self.to_Image(img_condition), masked_rect, image_array
