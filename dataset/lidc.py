import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob
import h5py
import nibabel as nib
from PIL import Image
from dataset.transform_3d import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage import measure
from xraysyn.networks.drr_projector_new import DRRProjector
from torch import nn

device = "cuda:0"
proj = DRRProjector(
    mode="forward", volume_shape=(128, 128, 128), detector_shape=(128, 128),
    pixel_size=(1.0, 1.0), interp="trilinear", source_to_detector_distance=1200).to(device)


class NormLayer(nn.Module):
  def __init__(self):
    super(NormLayer, self).__init__()

  def forward(self, inp):
    # print(inp.shape)
    inp = inp - inp.min()
    return inp / (inp.max() - inp.min())

def get_T(inp):
  param = np.asarray(inp)
  param = param * np.pi
  T = get_6dofs_transformation_matrix(param[3:], param[:3])
  T = torch.FloatTensor(T[np.newaxis, ...]).to('cuda')
  return torch.cat([T, T, T, T])


def get_6dofs_transformation_matrix(u, v):
  """ https://arxiv.org/pdf/1611.10336.pdf
  """
  x, y, z = u
  theta_x, theta_y, theta_z = v

  # rotate theta_z
  rotate_z = np.array([
    [np.cos(theta_z), -np.sin(theta_z), 0, 0],
    [np.sin(theta_z), np.cos(theta_z), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])

  # rotate theta_y
  rotate_y = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y), 0],
    [0, 1, 0, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y), 0],
    [0, 0, 0, 1]
  ])

  # rotate theta_x and translate x, y, z
  rotate_x_translate_xyz = np.array([
    [1, 0, 0, x],
    [0, np.cos(theta_x), -np.sin(theta_x), y],
    [0, np.sin(theta_x), np.cos(theta_x), z],
    [0, 0, 0, 1]
  ])

  return rotate_x_translate_xyz.dot(rotate_y).dot(rotate_z)

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1   # 大于-320的为2，小于-320的为1
    labels = measure.label(binary_image)   # 连通域

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 1

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0
    print(binary_image.max())
    print(binary_image.min())

    return binary_image

def plot_3d(image, threshold=None):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces,_,_ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

class LIDCDataset(Dataset):
    def __init__(self, root_dir='G:/workspace/datasets/lidc_idri_true', augmentation=False):
        self.root_dir = root_dir
        self.file_names = glob.glob(os.path.join(
            root_dir, './*.nii'), recursive=True)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img = nib.load(path)
        img = np.array(img.get_fdata())-1000
        segmented_lungs = segment_lung_mask(img, False)
        segmented_lungs_fill = segment_lung_mask(img, True)
        print(img.max())
        print(img.min())
        plt.hist(img.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        plot_3d(img, 400)
        plot_3d(segmented_lungs, 0)
        img = (img-np.min(img))/(np.max(img) - np.min(img))
        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.permute(2,1,0)
        imageout = imageout.unsqueeze(0)

        return {'data': imageout}


class SingleDataGenerator(Dataset):
        def __init__(self, root_dir='G:/workspace/datasets', child_dir="LIDC", mode="val", cond=False, **kwargs):
            super(SingleDataGenerator, self).__init__()

            self.dataset = os.path.expanduser(os.path.join(root_dir, child_dir))
            self.mode = mode
            self.cond = cond

            if child_dir == "LIDC":
                self.anno_path = os.path.join('{}', "LIDC-HDF5-256", '{}', 'ct_xray_data.h5')
                if self.mode == "train":
                    self.spilt_txt = "train"
                elif self.mode == "val":
                    self.spilt_txt = "test"
                elif self.mode == "visual":
                    self.spilt_txt = "visual"
                else:
                    raise ValueError('Unkown self.mode')
            self.items = self.load_idx(self.spilt_txt)
            self.data_augmentation = CT_XRAY_Data_Augmentation()
            self.norm = NormLayer()

        def load_idx(self, spilt_txt):
            ids = []
            dataset = self.dataset
            txt_path = os.path.join(dataset, spilt_txt + '.txt')
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    ids += [(dataset, line.strip())]
            return ids

        def __getitem__(self, idx):
            img_idx = self.items[idx]
            img_item_path = self.anno_path.format(*img_idx)
            with h5py.File(img_item_path, "r") as f:
                ct = f['ct'][()]
                # ct = np.flip(ct, axis=0)
                # segmented_lungs = segment_lung_mask(ct, False)
                # segmented_lungs_fill = segment_lung_mask(ct, True)
                xray1 = f['xray1'][()]
                xray1 = np.expand_dims(xray1, axis=0)
                xray2 = f['xray2'][()]
                xray2 = np.expand_dims(xray2, axis=0)
                ct, xray1, xray2 = self.data_augmentation([ct, xray1, xray2])
                ct = ct.unsqueeze(dim=0)
                # T_in = get_T([0, 0, 0, 0, 0, 0])
                # ct = torch.unsqueeze(ct, dim=0)
                # ct = torch.unsqueeze(ct, dim=0).to('cuda')
                # ct = ct.transpose(4, 3)
                # ct = ct.contiguous()
                # xray = proj(ct, T_in)
                # xray = self.norm(xray)
                # ct = ct.squeeze(dim=0).cpu()
                #
                # from torchvision import transforms
                # tensor2img = transforms.ToPILImage()
                # xray = xray.squeeze(dim=0).cpu()
                # xray_test1 = torch.squeeze(xray).to("cpu")
                # print(xray_test1.shape)
                # im1 = tensor2img(xray_test1)
                # im1.show()

                if self.cond == True:
                    return {"data": ct}
                else:
                    return {'data': ct, "xray1": xray1, "xray2": xray2, "image_name": img_idx[1][:-8]}

        def __len__(self):
            return len(self.items)


class LNDb(Dataset):
    def __init__(self, root_dir='G:/workspace/datasets', child_dir="8348419", mode="train", cond=False, **kwargs):
        super(LNDb, self).__init__()

        self.dataset = os.path.expanduser(os.path.join(root_dir, child_dir))
        self.mode = mode
        self.cond = cond

        if child_dir == "8348419":
            self.anno_path = os.path.join('{}', "testdata0_preprocessed_332_no_couch", '{}')
            if self.mode == "train":
                self.spilt_txt = "train"
            elif self.mode == "val":
                self.spilt_txt = "test"
            elif self.mode == "visual":
                self.spilt_txt = "visual"
            else:
                raise ValueError('Unkown self.mode')
        self.items = self.load_idx(self.spilt_txt)
        self.data_augmentation = CT_XRAY_Data_Augmentation_ct()
        self.norm = NormLayer()


    def load_idx(self, spilt_txt):
        ids = []
        dataset = self.dataset
        txt_path = os.path.join(dataset, spilt_txt + '.txt')
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                ids += [(dataset, line.strip())]
        return ids

    def __getitem__(self, idx):
        img_idx = self.items[idx]
        img_item_path = self.anno_path.format(*img_idx)
        print(img_item_path)
        with h5py.File(img_item_path, "r") as f:
            ct = f['ct'][()]
            # ct = np.flip(ct, axis=0)
            # segmented_lungs = segment_lung_mask(ct, False)
            # segmented_lungs_fill = segment_lung_mask(ct, True)
            # xray1 = f['xray1'][()]
            # xray1 = np.expand_dims(xray1, axis=0)
            # xray2 = f['xray2'][()]
            # xray2 = np.expand_dims(xray2, axis=0)
            ct, ct1 = self.data_augmentation([ct, ct])
            ct = ct.unsqueeze(dim=0)
            T_in = get_T([-0.5, 0, 0, 0, 0, 0])
            # ct = torch.unsqueeze(ct, dim=0)
            ct = torch.unsqueeze(ct, dim=0).to('cuda')
            ct = torch.flip(ct, dims=[2])
            # ct = ct.transpose(3, 4)
            ct = ct.contiguous()
            xray = proj(ct, T_in)
            xray = self.norm(xray)
            ct = ct.squeeze(dim=0).cpu()
            #
            # from torchvision import transforms
            # tensor2img = transforms.ToPILImage()
            xray = xray.squeeze(dim=0).cpu()
            # xray_test1 = torch.squeeze(xray).to("cpu")
            # print(xray_test1.shape)
            # im1 = tensor2img(xray_test1)
            # im1.show()

            if self.cond == True:
                return {"data": ct}
            else:
                return {'data': ct, "xray": xray, "image_name": img_idx[1][:-3]}

    def __len__(self):
        return len(self.items)


class LUNA(Dataset):
    def __init__(self, root_dir='G:/workspace/datasets', child_dir="LUNA16", mode="train", cond=False, **kwargs):
        super(LUNA, self).__init__()

        self.dataset = os.path.expanduser(os.path.join(root_dir, child_dir))
        self.mode = mode
        self.cond = cond

        if child_dir == "LUNA16":
            self.anno_path = os.path.join('{}', "image_luna_hdf5", '{}')
            if self.mode == "train":
                self.spilt_txt = "train"
            elif self.mode == "val":
                self.spilt_txt = "large_nodules"
            elif self.mode == "visual":
                self.spilt_txt = "visual"
            else:
                raise ValueError('Unkown self.mode')
        self.items = self.load_idx(self.spilt_txt)
        self.data_augmentation = CT_XRAY_Data_Augmentation_ct()
        self.norm = NormLayer()


    def load_idx(self, spilt_txt):
        ids = []
        dataset = self.dataset
        txt_path = os.path.join(dataset, spilt_txt + '.txt')
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                ids += [(dataset, line.strip())]
        return ids

    def __getitem__(self, idx):
        img_idx = self.items[idx]
        img_item_path = self.anno_path.format(*img_idx)
        print(img_item_path)
        with h5py.File(img_item_path, "r") as f:
            ct = f['ct'][()]
            # ct = np.flip(ct, axis=0)
            # segmented_lungs = segment_lung_mask(ct, False)
            # segmented_lungs_fill = segment_lung_mask(ct, True)
            # xray1 = f['xray1'][()]
            # xray1 = np.expand_dims(xray1, axis=0)
            # xray2 = f['xray2'][()]
            # xray2 = np.expand_dims(xray2, axis=0)
            ct, ct1 = self.data_augmentation([ct, ct])
            ct = ct.unsqueeze(dim=0)
            T_in = get_T([-0.5, 0, 0, 0, 0, 0])
            # ct = torch.unsqueeze(ct, dim=0)
            ct = torch.unsqueeze(ct, dim=0).to('cuda')
            ct = torch.flip(ct, dims=[2])
            # ct = ct.transpose(3, 4)
            ct = ct.contiguous()
            xray = proj(ct, T_in)
            xray = self.norm(xray)
            ct = ct.squeeze(dim=0).cpu()
            #
            # from torchvision import transforms
            # tensor2img = transforms.ToPILImage()
            xray = xray.squeeze(dim=0).cpu()
            # xray_test1 = torch.squeeze(xray).to("cpu")
            # print(xray_test1.shape)
            # im1 = tensor2img(xray_test1)
            # im1.show()

            if self.cond == True:
                return {"data": ct}
            else:
                return {'data': ct, "xray": xray, "image_name": img_idx[1][:-3]}

    def __len__(self):
        return len(self.items)

# 自定义中心裁剪函数
def center_crop(pil_image, crop_size=(768, 768)):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = (pil_image.size[0] - crop_size[0]) // 2
    crop_upper = (pil_image.size[1] - crop_size[1]) // 2
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))

def center_crop2(image, crop_size):
    """
    对图像进行中心裁剪，只保留中心部分。

    参数:
        image (PIL.Image): 输入的图像。
        crop_size (tuple): 裁剪的目标大小，格式为 (width, height)。

    返回:
        PIL.Image: 裁剪后的图像。
    """
    width, height = image.size
    target_width, target_height = crop_size

    # 计算裁剪区域的左上角坐标
    left = (width - target_width) // 2
    top = (height - target_height) // 2

    # 计算裁剪区域的右下角坐标
    right = left + target_width
    bottom = top + target_height

    # 裁剪图像
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


# 定义转换操作：调整大小并转换为Tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图像大小为128x128
    transforms.ToTensor()           # 转换为Tensor
])

class SingleDataGenerator_real(Dataset):
    def __init__(self, root_dir='G:/workspace/datasets', child_dir="LIDC", mode="val", cond=False, **kwargs):
        super(SingleDataGenerator_real, self).__init__()

        self.dataset = os.path.expanduser(os.path.join(root_dir, child_dir))
        self.mode = mode
        self.cond = cond

        if child_dir == "LIDC":
            self.anno_path = os.path.join('{}', '{}', '1')
            self.ct_path = os.path.join('G:/workspace/datasets/LIDC', "LIDC-HDF5-256", 'LIDC-IDRI-0001.20000101.3000566.1', 'ct_xray_data.h5')
            if self.mode == "train":
                self.spilt_txt = "train"
            elif self.mode == "val":
                self.spilt_txt = "test"
            elif self.mode == "visual":
                self.spilt_txt = "visual"
            elif self.mode == "real":
                self.spilt_txt = "real"
            else:
                raise ValueError('Unkown self.mode')
        self.items = self.load_idx(self.spilt_txt)
        self.data_augmentation = CT_XRAY_Data_Augmentation()
        self.norm = NormLayer()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 调整图像大小为128x128
            transforms.ToTensor()  # 转换为Tensor
        ])


    def load_idx(self, spilt_txt):
        ids = []
        dataset = self.dataset
        txt_path = os.path.join(dataset, spilt_txt + '.txt')
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                ids += [(dataset, line.strip())]
        return ids

    def __getitem__(self, idx):
        img_idx = self.items[idx]
        ct_item_path = self.ct_path
        image_item_path = self.anno_path.format(*img_idx)

        with h5py.File(ct_item_path, "r") as f:
            ct = f['ct'][()]
            print(ct.max())
            print(ct.min())

            xray1 = f['xray1'][()]
            xray1 = np.expand_dims(xray1, axis=0)
            xray2 = f['xray2'][()]
            xray2 = np.expand_dims(xray2, axis=0)
            ct, xray1, xray2 = self.data_augmentation([ct, xray1, xray2])
            ct = ct.unsqueeze(dim=0)

        xray1_image_path = os.path.join(image_item_path, '1.jpg')
        xray2_image_path = os.path.join(image_item_path, '2.jpg')
        xray1_true = Image.open(xray1_image_path).convert('L')
        xray2_true = Image.open(xray2_image_path).convert('L')
        # 自定义中心裁剪
        crop_size = (160, 160)  # 设置裁剪大小
        crop_size2 = (128, 128)  # 设置裁剪大小
        xray1_true = center_crop(xray1_true, crop_size)
        xray2_true = center_crop(xray2_true, crop_size)
        xray1_true = center_crop2(xray1_true, crop_size2)
        xray2_true = center_crop2(xray2_true, crop_size2)
        xray1_true = transform(xray1_true)
        xray2_true = transform(xray2_true)

        print(xray1_true.max())
        print(xray1_true.min())

        if self.cond == True:
            return {"data": ct}
        else:
            return {'data': ct, "xray1": xray1_true, "xray2": xray2_true}

    def __len__(self):
        return len(self.items)

class CT_XRAY_Data_Augmentation(object):
  def __init__(self):
    self.augment = List_Compose([


      (Resize_image(size=(128, 128, 128)),
       Resize_image(size=(1, 128, 128)),
       Resize_image(size=(1, 128, 128))),


      (Limit_Min_Max_Threshold(0, 2500), None, None),


      (Normalization(0, 2500),
       Normalization(0, 255),
       Normalization(0, 255)),

      (Normalization_gaussian(0., 1.),
       Normalization_gaussian(0., 1.),
       Normalization_gaussian(0., 1.)),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor(), ToTensor())

    ])




  def __call__(self, img_list):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img_list)

class CT_XRAY_Data_Augmentation_ct(object):
    def __init__(self):
        self.augment = List_Compose([

            (Resize_image(size=(128, 128, 128)),
             Resize_image(size=(128, 128, 128))),

              (Limit_Min_Max_Threshold(0, 2500),
               Limit_Min_Max_Threshold(0, 2500)),

              (Normalization(0, 2500),
               Normalization(0, 2500)),

              (Normalization_gaussian(0., 1.),
               Normalization_gaussian(0., 1.)),

              (ToTensor(), ToTensor())

        ])

    def __call__(self, img_list):
        '''
        :param img: PIL image
        :param boxes: numpy.ndarray
        :param labels: numpy.ndarray
        :return:
        '''
        return self.augment(img_list)


if __name__ == '__main__':
    a = LUNA()
    print(len(a))
    b = a[0]['data'][0]
    xray = a[0]['xray']

    print(type(a[0]['data']))
    print(a[0]['image_name'])

    print(torch.min(b))
    toPIL = transforms.ToPILImage()
    print(b.shape)
    print(xray.shape)
    xray_im = toPIL(xray)
    im = toPIL(b[40])
    im.show()
    xray_im.show()
    # a[300].show()
