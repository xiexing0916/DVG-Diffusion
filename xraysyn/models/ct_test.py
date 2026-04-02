import torch

from xraysyn.networks.drr_projector_new import DRRProjector

from transform_3d import *
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import h5py
import os
from PIL import Image, ImageEnhance
import pickle
device = "cuda:0"
proj = DRRProjector(
    mode="forward", volume_shape=(128, 128, 128), detector_shape=(128, 128),
    pixel_size=(1.0, 1.0), interp="trilinear", source_to_detector_distance=1200).to(device)


class SingleDataGenerator(Dataset):
    def __init__(self, root_dir='E:/workspace/datasets', child_dir="LIDC", mode="train", **kwargs): ## '/mnt/e/workspace/datasets'
        super(SingleDataGenerator, self).__init__()

        self.dataset = os.path.expanduser(os.path.join(root_dir, child_dir))
        self.mode = mode

        if child_dir == "LIDC":
            self.anno_path = os.path.join('{}', "LIDC-HDF5-256", '{}', 'ct_xray_data.h5')
            if self.mode == "train":
                self.spilt_txt = "train"
            elif self.mode == "val":
                self.spilt_txt = "test"
            else:
                raise ValueError('Unkown self.mode')
        self.items = self.load_idx(self.spilt_txt)
        self.data_augmentation = CT_XRAY_Data_Augmentation()
        self.xray_augmentation = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
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
        img_item_path = self.anno_path.format(*img_idx)
        with h5py.File(img_item_path, "r") as f:
            ct = f['ct'][()]
            bone_mask = window_transform(ct, 800, 1600, normal=True)
            xray1 = f['xray1'][()]
            # print(ct.shape)
            # print(xray1.shape)
            # ct = ct.transpose(1,2,0)
            xray1 = np.expand_dims(xray1, axis=0)
            # xray1 = Image.fromarray(np.uint8(ct[10]))
            # xray1 = xray1.convert("RGB")
            # xray1 = self.xray_augmentation(xray1)
            # xray1 = torch.from_numpy(xray1).view(1, 256, 256).float()
            ct, xray1 = self.data_augmentation([ct, xray1])
            ct = np.expand_dims(ct, axis=0)
            ct = np.expand_dims(ct, axis=0)
            bone_mask = np.expand_dims(bone_mask, axis=0)
            bone_mask = np.expand_dims(bone_mask, axis=0)
            ct = ct.transpose((0, 1, 2, 4, 3))
            bone_mask = bone_mask.transpose((0, 1, 2, 4, 3))

        return ct, bone_mask

    def __len__(self):
        return len(self.items)

class CT_XRAY_Data_Augmentation(object):
  def __init__(self):
    self.augment = List_Compose([


      (Resize_image(size=(128, 128, 128)),
       Resize_image(size=(1, 128, 128))),

      (Limit_Min_Max_Threshold(0, 2500), None),

      (Normalization(0, 2500),
       Normalization(0, 255)),

      (Normalization_gaussian(0., 1.),
       Normalization_gaussian(0., 1.)),

      # (Get_Key_slice(opt.select_slice_num), None),

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

def get_T(inp):
    param =np.asarray(inp)
    param = param * np.pi
    T = get_6dofs_transformation_matrix(param[3:], param[:3])
    T = torch.FloatTensor(T[np.newaxis, ...]).to(device)
    return torch.cat([T,T,T,T])

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

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
   """
   return: trucated image according to window center and window width
   and normalized to [0,1]
   """
   minWindow = float(windowCenter) - 0.5*float(windowWidth)
   newimg = (ct_array - minWindow) / float(windowWidth)
   newimg[newimg < 0] = 0
   newimg[newimg > 1] = 1
   if not normal:
       newimg = (newimg * 255).astype('uint8')
   return newimg
import os
print((os.getcwd()))
# bone_absorb = pickle.load(
#             open('E:/workspace/guided-ct/simplified_bone_absorb_2d.pt', 'rb')).to("cuda:0")
# tissue_absorb = pickle.load(
#             open('E:/workspace/guided-ct/simplified_tissue_absorb_2d.pt', 'rb')).to("cuda:0")

print(torch.zeros(1).cuda())
class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, inp):
        # print(inp.shape)
        inp = inp - inp.min()
        return inp/inp.max()
norm = NormLayer()

def ct2xray(vol, bone, T_in):  # CTvol, bone mask, 视角矩阵T_in
    # vol = 0.0008088 * (vol * 5000 - 1000) + 1.030
    bone_vol = vol * bone  # 公式4
    tissue_vol = vol * (1 - bone)  # 公式4
    bone_proj = proj(bone_vol, T_in)  # forward,生成t_bone 公式4
    tissue_proj = proj(tissue_vol, T_in)  # forward,生成t_tissue 公式4
    # bone_proj = norm(bone_proj)
    # tissue_proj = norm(tissue_proj)
    atten_bone = bone_absorb(bone_proj)*0.7  # forward,生成I_bone 公式6,体现bone被吸收的值，min -29.1006
    atten_tissue = tissue_absorb(tissue_proj)  # forward,生成I_tissue 公式6，体现tissue被吸收的值，min -16.2877
    print("tissue",torch.max(atten_tissue), torch.min(atten_tissue))
    print("bone",torch.max(atten_bone), torch.min(atten_bone))

    # atten_bone = norm(atten_bone)
    # atten_tissue = norm(atten_tissue)

    # atten_tissue1 = torch.squeeze(atten_tissue[:,0,:,:]).to("cpu")
    # im10 = tensor2img(atten_tissue1)
    # im10.show()
    # atten_bone1 = torch.squeeze(atten_bone[:,0,:,:]).to("cpu")
    # im11 = tensor2img(atten_bone1)
    # im11.show()

    atten_proj = atten_bone + atten_tissue  # forward,相加生成I_atten指数上的值I 公式6
    atten_proj = norm(atten_proj)
    out_new = torch.exp(atten_proj).sum(dim=1).view(vol.shape[0], 1, 256, 256) # 对I求指数，得到I_atten 公式6
    out_new = norm(out_new.max() - out_new)  # 公式5
    return out_new, torch.cat([bone_proj, tissue_proj], 1)


# 返回xray图像矩阵，以及

if __name__ == '__main__':
    a = SingleDataGenerator()
    tensor2img = transforms.ToPILImage()
    T_in = get_T([0, 0, 0, 0, 0, 0])  # 前三位旋转yxz，0正面，0.5侧面，1背面。后三位平移y上x左z大
    T_in1 = get_T([0, 0, 0, 0, 0, 0])
    print(type(a[0][0]))
    ct1 = a[0][0]
    bone_mask1 = a[0][1]
    bone_mask1 = bone_mask1.astype(np.float32)
    bone_mask1 = torch.from_numpy(bone_mask1).to(device)
    print('ct1', ct1.shape)
    print(np.max(ct1))
    print(np.min(ct1))
    ct1 = torch.from_numpy(ct1).to(device)
    ct1 = ct1.contiguous()
    print("max", torch.max(ct1))
    print("min", torch.min(ct1))
    xray_test = proj(ct1, T_in)
    # xray_test_atten, bone_t = ct2xray(ct1, bone_mask1, T_in)
    # bone_poj = bone_t[0][0].cpu()
    # bone_poj = norm(bone_poj)
    # tissue_poj = bone_t[0][1].cpu()
    # tissue_poj = norm(tissue_poj)
    # im4 = tensor2img(bone_poj)
    # im5 = tensor2img(tissue_poj)
    # # im4.show(title = "bone_fp")
    # # im5.show(title = "tissue_fp")
    # xray_test_atten = torch.squeeze(xray_test_atten).to("cpu")
    # im3 = tensor2img(xray_test_atten)
    # im3.show(title = "ct2xray_final")

    # xray_test_1 = proj(ct1, T_in1)
    # xray_test1_11 = torch.squeeze(xray_test_1).to("cpu")
    # xray_test1_11 = (xray_test1_11 - torch.min(xray_test1_11)) / (torch.max(xray_test1_11) - torch.min(xray_test1_11))
    print(xray_test.shape)
    xray_test1 = torch.squeeze(xray_test).to("cpu")
    print(xray_test1.shape)
    xray_test1 = (xray_test1 - torch.min(xray_test1)) / (torch.max(xray_test1) - torch.min(xray_test1))
    im1 = tensor2img(xray_test1)
    # im2 = tensor2img(xray_test1_11)

    ct1 = torch.squeeze(ct1).to('cpu')
    bone_mask1 = torch.squeeze(bone_mask1).to('cpu').float()
    print('ct1', ct1.shape)
    ct2 = tensor2img(ct1[50, :, :])
    bone_mask2 = tensor2img(bone_mask1[50, :, :])
    # bone_mask2.show(title = "bone_mask")
    ct2.show()
    #
    im1.show()
    # im2.show()
