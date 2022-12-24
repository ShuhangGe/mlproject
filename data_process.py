import os
import random
from PIL import Image

# #make 2 classes datasets form origin dataset
# nontoxic_dir = 'D:/data/plant_toxic_non/tpc-imgs/nontoxic_images'
# toxic_dir = 'D:/data/plant_toxic_non/tpc-imgs/toxic_images'
# dirs = os.listdir(nontoxic_dir)
# print(dirs)
# def get_picpath(pic_dirs):
#     pic_list = list()
#     for dir in dirs:
#         pic_dir = os.path.join(pic_dirs,dir)
#         pic_names = os.listdir(pic_dir)
#         for pic_name in pic_names:
#             pic_path = os.path.join(pic_dir,pic_name)
#             pic_list.append(pic_path)
#     return pic_list
# pic_pathes_nontoxic = get_picpath(nontoxic_dir)
# pic_pathes_toxic = get_picpath(toxic_dir)
# print(len(pic_pathes_nontoxic))
# print(len(pic_pathes_toxic))
# split = 0.8 #train:0.8, test:0.2
# def set_data(pic_pathes,split,save_path,label):
#     train_num = int(len(pic_pathes) * split)
#     test_num = len(pic_pathes) - train_num
#     for i in range(train_num):
#         origin_pic_path = pic_pathes[i]
#         origin_pic = Image.open(origin_pic_path)
#         path_train = save_path+'/'+'train'+'/'+label
#         if not os.path.exists(path_train):
#             os.makedirs(path_train)
#         img_path = save_path+'/'+'train'+'/'+label+'/'+str(i).zfill(3) +'.jpg'
#         origin_pic.save(img_path)
#     for i in range(test_num):
#         j = train_num+i
#         origin_pic_path = pic_pathes[j]
#         origin_pic = Image.open(origin_pic_path)
#         path_test = save_path+'/'+'test'+'/'+label
#         if not os.path.exists(path_test):
#             os.makedirs(path_test)
#         img_path = save_path+'/'+'test'+'/'+label+'/'+str(i).zfill(3) +'.jpg'
#         origin_pic.save(img_path)
# save_path = 'D:/code/ml_project/data'
# label_nontoxic = 'nontoxic'
# set_data(pic_pathes_nontoxic,split,save_path,label_nontoxic)
# label_toxic = 'toxic'
# set_data(pic_pathes_toxic,split,save_path,label_toxic)

nontoxic_dir = 'D:/data/plant_toxic_non/tpc-imgs/nontoxic_images'
toxic_dir = 'D:/data/plant_toxic_non/tpc-imgs/toxic_images'
dirs = os.listdir(nontoxic_dir)

# #compensate pics
# path = 'D:/data/plant_toxic_non/tpc-imgs/toxic_images/001'
# pic_names = os.listdir(path)
# picnum = len(pic_names)
# num = 1000-picnum
# label = picnum
# for i in range(num):
#     pic = random.randint(0,picnum-1)
#     pic_path = path +'/'+str(pic).zfill(3) + '.jpg'
#     print(pic_path)
#     img = Image.open(pic_path)
#     new_path = path +'/'+str(label).zfill(3) + '.jpg'
#     img.save(new_path)
#     label = label +1


# #make 10 classes datasets form origin dataset
