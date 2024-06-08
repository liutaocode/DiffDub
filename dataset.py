import os, cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import albumentations as A
import numpy as np
import json

class HDTFDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        nose_mouth_chin_path,
        landmark_data_path,
        meta_info,
        N_frames=1,
        N_state_rand=False,
        visualized=False,
        with_max_frame=10,
        cond_has_nose=False,
        exts=['png', 'jpeg'],
    ):
        super().__init__()
        
        self.folder = folder
        self.image_size = image_size
        self.landmark_data_path = landmark_data_path
        self.json_obj = json.loads(open(nose_mouth_chin_path).read())
        self.N_frames = N_frames,
        self.N_state_rand = N_state_rand,
        self.visualized = visualized
        self.with_max_frame = with_max_frame
        self.cond_has_nose = cond_has_nose

        if landmark_data_path.endswith(".json"):
            self.landmark_obj = json.loads(open(landmark_data_path).read())
            print('read from json')
        else:
            self.landmark_obj = None
            print('not read from json')
        
        train_set = set()
        for line in open(meta_info):
            train_set.add(line.split()[0])

        self.items = []
        self.file_lists_dict = dict()
        for filename in self.json_obj.keys():
            self.file_lists_dict[filename] = []
            for txtfile in self.json_obj[filename]:
                txtpath = txtfile.replace(".txt", ".png")
                
                if filename in train_set:
                    self.file_lists_dict[filename].append(txtpath)
                    if self.landmark_obj == None:
                        self.items.append((filename, txtfile, txtpath, None))
                    else:
                        self.items.append((filename, txtfile, txtpath, self.landmark_obj[filename][txtfile]))            
            random.shuffle(self.file_lists_dict[filename])

        self.normlize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
 
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
    
    def new_rec(self, key_points):
        x_lists, y_lists = [], []
        for x,y in key_points:
            x_lists.append(x)
            y_lists.append(y)
        
        x1 = min(x_lists)
        y1 = min(y_lists)
        x2 = max(x_lists)
        y2 = max(y_lists)
        return x1,y1,x2,y2
    
    def get_surrounding_axis(self, matrix):
        x_lists, y_lists = [], []
        h,w = matrix.shape
        for y in range(h):
            for x in range(w) :
                if matrix[y][x] > 0:
                    x_lists.append(x)
                    y_lists.append(y)
        return min(x_lists), max(x_lists), min(y_lists), max(y_lists)
            
    
    def get_input_aug(self, ori_img, landmarks):

        lists = []
        lists.append([landmarks[6][0], landmarks[30][1]]) # eyes modified here
        lists.append([landmarks[10][0], landmarks[30][1]]) # eyes modified here
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
    
    def get_input_condtion(self, ori_img, landmarks):

        lists = []
        lists.append([landmarks[6][0], landmarks[21][1]]) # eyes modified here
        lists.append([landmarks[10][0], landmarks[21][1]]) # eyes modified here
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
    
    def scale_to_reso(self, origin_axis, original_height, original_width, to_reso):
        x, y = origin_axis
        to_x, to_y = x/original_width*to_reso, y/original_height*to_reso
        to_x = int(to_x)
        to_y = int(to_y)

        return (to_x, to_y)
        
    def __len__(self):
        return len(self.items)
        
    def get_landmarks_single(self, height, width, landmark_path):
        
        landmarks = []
        for line in open(landmark_path):
            items = line.replace("\n", "").split()
            out_points = self.scale_to_reso((int(items[0]), int(items[1])), height, width, self.image_size)
            landmarks.append(out_points)
            
        assert len(landmarks) == 68
        return landmarks
    
    def get_landmarks(self, height, width, ori_landmarks):
        
        landmarks = []
        for x, y in ori_landmarks:
            out_points = self.scale_to_reso((int(x), int(y)), height, width, self.image_size)
            landmarks.append(out_points)
            
        assert len(landmarks) == 68
        return landmarks

    def __getitem__(self, index):
        filename, txtpath, imgpath, landmarks  = self.items[index]
        mapping = self.json_obj[filename][txtpath]
        nose, lip, chin, left, right, height, width = mapping['nose'], mapping['lip'], mapping['chin'], mapping['left'], mapping['right'] ,mapping['height'] ,mapping['width']

        path = os.path.join(self.folder, filename, imgpath)
        if landmarks == None:
            landmark_path = os.path.join(self.landmark_data_path, filename, txtpath)
            landmarks = self.get_landmarks_single(height, width, landmark_path)
        else:
            landmarks = self.get_landmarks(height, width, landmarks)
        
        avoid_set = set()
        avoid_set.add(imgpath)
        ref_lists = []

        if self.N_frames[0] != 0:
            for i in range(self.N_frames[0]): # self.N_frames is a tuple
                while True:
                    randindex = random.randint(0, len(self.file_lists_dict[filename])-1)
                    ref_name = self.file_lists_dict[filename][randindex]
                    if ref_name not in avoid_set:
                        avoid_set.add(ref_name)
                        
                        ref_image_path = os.path.join(self.folder, filename, ref_name)
                        ref_img = self.get_ref_image(ref_image_path)
                        ref_img = self.normlize(ref_img)
                        ref_lists.append(ref_img)
                        break
            
        ref_img = np.concatenate(ref_lists, axis=0)
        
        img, con_img, masked_rect, image_array = self.get_ori_cond_image(path, landmarks)
        
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
        
        if self.cond_has_nose:
            img_condition = self.get_input_condtion(cv2_image_resized, landmarks)

        return self.to_Image(image_array), self.to_Image(img_condition), masked_rect, image_array
