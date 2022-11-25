# head dims:512,512,512,512,512,512,512,512,128
# code is basicly:https://github.com/google-research/deep_representation_one_class
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from tqdm import tqdm
import datetime
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from torchvision.models import resnet18
from matplotlib import cm
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from loss_fun import FocalLoss, SSIM
import pickle
from dataset import MVTecAT, Repeat
#from cutpaste import CutPasteNormal,CutPasteScar, cut_paste_collate_fn  ,CutPastePlus2
from model import ProjectionNet
from utils import str2bool
from sklearn.metrics import roc_auc_score
import numpy as np
from PIL import Image

Get_feature = resnet18(pretrained=True)
Get_feature.to('cuda:0')
Get_feature.eval()
Hook_outputs = []


def hook(module, input, output):
    Hook_outputs.append(output)
Get_feature.bn1.register_forward_hook(hook)
Get_feature.layer1[-1].register_forward_hook(hook)
Get_feature.layer2[-1].register_forward_hook(hook)
Get_feature.layer3[-1].register_forward_hook(hook)




def run_training(data_type="screw",
                 model_dir="models",
                 epochs=256,
                 pretrained=True,
                 test_epochs=10,
                 freeze_resnet=20,
                 learninig_rate=0.03,
                 optim_name="SGD",
                 batch_size=64,
                 head_layer=8,
                 #cutpate_type=CutPasteNormal,
                 device = "cuda",
                 workers=8,
                 size = 256,
                args = None,
                 duochidu=True,
                use_jiegou_only = False,
                use_wenli_only = False,
                without_qianjing=False,use_duibi = False,
                se=False,gg=False,without_loss = [] ,test_memory_samples = False,memory_samples=15,test_toy_dataset = False,
                MVTECAD_DATA_PATH=None,

                 ):
    torch.multiprocessing.freeze_support()
    weight_decay = 0.00003
    momentum = 0.9
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )
    print(without_loss)
    min_scale = 1
    if without_loss[1]:
        learninig_rate=1

    # create Training Dataset and Dataloader
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    # train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, saturation=0.1))
    # train_transform.transforms.append(cutpate_type(transform = after_cutpaste_transform,args = args,data_type = data_type,
    #                                                use_wenli_only = use_wenli_only,use_jiegou_only=use_jiegou_only,without_qianjing=without_qianjing))

    if test_memory_samples:
        all_num = memory_samples
    else:
        all_num = 30
        if data_type == 'screw_new' or data_type == 'screw':
            all_num = 120
        if data_type == 'toothbrush':
            all_num = 10

    train_data = MVTecAT(MVTECAD_DATA_PATH, data_type, transform = train_transform, size=int(size * (1/min_scale)),memory_number = all_num)
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                            shuffle=True, num_workers=workers, collate_fn=None,
                             pin_memory=True,)


    num_classes = 2
    model = ProjectionNet(num_classes=num_classes,data_type=data_type,use_se=se,use_duibi = use_duibi, duochidu=duochidu)

    weights = torch.load(f"Test_pth/model-{data_type}.tch")

    model.load_state_dict(weights)
    model.to(device)
    if test_memory_samples:
        with open(f"memory_features/train_{data_type}.pkl", 'rb',) as f:
            train_memory = pickle.load(f)
    else:
        with open(f"memory_features/train_{data_type}.pkl", 'rb',) as f:
            train_memory = pickle.load(f)


    loss_focal = FocalLoss()
    loss_l1 = torch.nn.L1Loss()

    if optim_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)
        #scheduler = None
    elif optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learninig_rate, weight_decay=weight_decay)
        scheduler = None
    else:
        print(f"ERROR unkown optimizer: {optim_name}")

    step = 0
    num_batches = len(dataloader)
    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out
    dataloader_inf =  get_data_inf()


    if not os.path.isdir(f"./test_out/{model_name}"):
        os.mkdir(f"./test_out/{model_name}")


    model.eval()
    Test_AUC_MMM_, per_pixel_rocauc,= eval_model(model_name,
                                                data_type,
                                                device=device,
                                                save_plots=False,
                                                size=size,
                                                show_training_data=False,
                                                model=model,
                                                step=step,
                                                Get_feature=Get_feature,
                                                train_memory=train_memory,
                                                test_memory_samples=test_memory_samples,
                                                memory_samples=memory_samples,
                                                test_toy_dataset = test_toy_dataset
                                                )
    return Test_AUC_MMM_,per_pixel_rocauc





def get_feature(Get_feature, x, train_memory,type_i,test_memory_samples,memory_samples):
    global Hook_outputs
    with torch.no_grad():
        _ = Get_feature(x)
        # get intermediate layer outputs
    m = torch.nn.AvgPool2d(3, 1, 1)
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ])
    for k, v in zip(test_outputs.keys(), Hook_outputs[1:]):
        test_outputs[k].append(m(v))
    # initialize hook outputs

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    layer1_feature_diff = []
    layer2_feature_diff = []
    layer3_feature_diff = []
    sim_di = []
    for t_idx in range(test_outputs['layer3'].shape[0]):  # 对每一个样本遍历
        for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer
            # construct a gallery of features at all pixel locations of the K nearest neighbors
            topk_feat_map = train_memory[layer_name]
            test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]  # 1,256,56,56

            # calculate distance matrix
            dist_matrix_list = []
            dist_matrix_all = []
            if test_memory_samples:
                all_num = memory_samples
            else:
                all_num = 30
                if type_i == 'toothbrush':
                    all_num = 10
                if type_i == 'screw':
                    all_num = 120
            #print(all_num)
            #all_num = topk_feat_map.shape[0]
            for iii in range(all_num):
                dist_matrix_list.append(torch.pow(topk_feat_map[iii] - test_feat_map[0], 2) ** 0.5)
                dist_matrix_all.append(dist_matrix_list[iii].sum())
            idx_ = torch.argmin(torch.Tensor(dist_matrix_all))
            dist_matrix = dist_matrix_list[idx_]
            sim_di.append(idx_)


            # k nearest features from the gallery (k=1)
            # score_map = torch.min(dist_matrix, dim=0)[0]#56,56 666666666!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            #score_map = torch.mean(dist_matrix_list[idx_], dim=0).cpu()
            if layer_name == "layer1":
                layer1_feature_diff.append(dist_matrix)
            if layer_name == "layer2":
                layer2_feature_diff.append(dist_matrix)
            if layer_name == "layer3":
                layer3_feature_diff.append(dist_matrix)

    layer1_feature = torch.stack(layer1_feature_diff)
    layer2_feature = torch.stack(layer2_feature_diff)
    layer3_feature = torch.stack(layer3_feature_diff)

    m_scale = 100  # 三级管的时候 这里只放大1000倍
    o_scale = 100
    if type_i=='transistor':
        m_scale=1000
    if type_i=='cable' or  type_i=='transistor' :#or  type_i=='screw_new' or type_i=='screw':
        o_scale = 1  # 80


    layer3_out_16x16_m = layer3_feature.cuda() * m_scale
    layer2_out_32x32_m = layer2_feature.cuda() * m_scale
    layer1_out_64x64_m = layer1_feature.cuda() * m_scale

    layer1_out_64x64_o = Hook_outputs[1] * o_scale
    layer2_out_32x32_o = Hook_outputs[2] * o_scale
    layer3_out_16x16_o = Hook_outputs[3] * o_scale

    # m = torch.nn.AvgPool2d(3, 1, 1)
    bn_out_128x128 = (Hook_outputs[0])
    Hook_outputs = []
    return layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o, layer1_out_64x64_m, layer2_out_32x32_m, layer3_out_16x16_m, bn_out_128x128





test_data_eval = None
test_transform = None
cached_type = None
def eval_model(modelname, defect_type, device="cpu", save_plots=False, size=256, show_training_data=True, model=None,
               train_embed=None, head_layer=8, step=0   , Get_feature = "Get_feature",test_memory_samples = False,memory_samples=15,
    train_memory = "train_memory",test_toy_dataset = False):

    global test_data_eval, test_transform, cached_type,Hook_outputs

    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize(size, Image.ANTIALIAS))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]))
        if test_toy_dataset :
            test_data_eval = MVTecAT("c:/xunlei_cloud/mvtec_anomaly_detection.tar(1)/toy_dataset/", defect_type,
                                     size, transform=test_transform, mode="test")
        else:
            test_data_eval = MVTecAT("/home/wangyizhuo/Documents/DATA_SETS/dataset_anomaly_detection/", defect_type,
                                     size, transform=test_transform, mode="test")

    dataloader_test = DataLoader(test_data_eval, batch_size=16,
                                 shuffle=False, num_workers=0)


    # get embeddings for test data
    labels = []
    output_segments = []
    logits = []
    true_masks = []
    with torch.no_grad():
        index_ = 0
        for x, label, img_mask in dataloader_test:  # x维度为B,3,256,256
            x = x.to(device)
            layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o, layer1_out_64x64_m, layer2_out_32x32_m, layer3_out_16x16_m, bn_out_128x128 = get_feature(Get_feature, x,
                                                                                                    train_memory,type_i = defect_type,test_memory_samples = test_memory_samples,memory_samples=memory_samples)
            output_segment, logit, layer_final_256x256 = model(x,layer1_out_64x64_o, layer2_out_32x32_o, layer3_out_16x16_o, layer1_out_64x64_m, layer2_out_32x32_m, layer3_out_16x16_m, bn_out_128x128)
            true_masks.append(img_mask.cpu())
            # save
            output_segments.append(output_segment.cpu())
            labels.append(label.cpu())

    labels = torch.cat(labels)  # 83
    output_segments = torch.cat(output_segments)  # 83,512
    output_segments = torch.softmax(output_segments, dim=1)

    #logits = torch.cat(logits)  # 83,512
    true_masks = torch.cat(true_masks)  # 83,512

    true_masks = true_masks.numpy()
    output_segments = output_segments.numpy()
    output_segments = output_segments[:, 1, :, :]

    #Get AUC from seg:
    MAX_anormaly = []
    for im_index in range(output_segments.shape[0]):
         MAX_anormaly.append(output_segments[im_index].max())
    all_auc = []
    auc_score_max = roc_auc_score(labels, np.array(MAX_anormaly))
    all_auc.append(auc_score_max)


    MAX_anormaly_100 = []


    if not os.path.isdir(f"./test_out/{modelname}"):
        os.mkdir(f"./test_out/{modelname}")
    if not os.path.isdir(f"./test_out/{modelname}/test"):
        os.mkdir(f"./test_out/{modelname}/test")
    # if not os.path.isdir(f"./test_out/{modelname}/test/{step}"):
    #     os.mkdir(f"./test_out/{modelname}/test/{step}")

    for im_index in range(output_segments.shape[0]):
         tempp = output_segments[im_index].flatten()
         tempp.sort()
         #tempp = tempp*tempp[62500]
         MAX_anormaly_100.append(tempp[65436:65536].mean())


    auc_score_max_100_mean = roc_auc_score(labels, np.array(MAX_anormaly_100))
    all_auc.append(auc_score_max_100_mean)
    for iiii in range(output_segments.shape[0]):
        plt.imsave(f"./test_out/{modelname}/test/pred_{iiii}.jpg", output_segments[iiii, :, :],
                   cmap=cm.gray)

    true_masks = true_masks.flatten().astype(np.uint32)
    output_segments = output_segments.flatten()
    # fpr, tpr, _ = roc_curve(true_masks.flatten(), output_segments.flatten())
    per_pixel_rocauc = roc_auc_score(true_masks, output_segments)
    Test_AUC_MMM_ = auc_score_max_100_mean
    print(f"-----------------------------------{defect_type}----Test_AUC_MMM",Test_AUC_MMM_,'Test_pixel_AUC:',per_pixel_rocauc)
    return  Test_AUC_MMM_,per_pixel_rocauc




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training defect detection as described in the CutPaste Paper.')
    parser.add_argument('--type', default="all",
                        help='MVTec defection dataset type to train seperated by , (default: "all": train all defect types)')

    parser.add_argument('--epochs', default=3701, type=int,
                        help='number of epochs to train the model , (default: 256)')
    
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models , (default: models)')

    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')
    
    parser.add_argument('--test_epochs', default=50, type=int,
                        help='interval to calculate the auc during trainig, if -1 do not calculate test scores, (default: 10)')                  

    parser.add_argument('--freeze_resnet', default=60000, type=int,
                        help='number of epochs to freeze resnet (default: 20)')
    
    parser.add_argument('--lr', default=0.5, type=float,#screw_new0.08 else 0.04
                        help='learning rate (default: 0.03)')

    parser.add_argument('--optim', default="sgd",
                        help='optimizing algorithm values:[sgd, adam] (dafault: "sgd")')

    parser.add_argument('--batch_size', default=4, type=int,#
                        help='batch size, real batchsize is depending on cut paste config normal cutaout has effective batchsize of 2x batchsize (dafault: "64")')   

    parser.add_argument('--head_layer', default=1, type=int,
                    help='number of layers in the projection head (default: 1)')
    
    parser.add_argument('--variant', default="plus2", choices=['normal', 'scar', '3way', 'union' ,'plus','plus2'], help='cutpaste variant to use (dafault: "3way")')
    
    parser.add_argument('--cuda', default=True, type=str2bool,
                    help='use cuda for training (default: False)')
    
    parser.add_argument('--workers', default=0, type=int, help="number of workers to use for data loading (default:8)")

    parser.add_argument('--MVTECAD_DATA_PATH', default="mvtec_datasets/",help='')
    #parser.add_argument('--MVTECAD_DATA_PATH', default="all", help='')


    args = parser.parse_args()
    print(args)#zipper leather metal_nut capsule pill
    all_types = [
        'zipper',
        'tile',
        'cable',
        'hazelnut',
        'metal_nut',
        'toothbrush',
        'leather',
        'carpet',

        'bottle',
        'transistor',
        'screw',

        'grid',
        'capsule',
        'pill',
        'wood',
    ]
    
    if args.type == "all":
        types = all_types
    else:
        types = args.type.split(",")
    
    # variant_map = {'normal':CutPasteNormal, 'scar':CutPasteScar, "plus2":CutPastePlus2}
    # variant = variant_map[args.variant]
    variant=None
    
    device = "cuda" if args.cuda else "cpu"
    print(f"using device: {device}")
    

    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    # save config.
    with open(Path(args.model_dir) / "run_config.txt", "w") as f:
        f.write(str(args))
    for memory_samples in [30]:
        for _ in range(1):
            all_img_auc = []
            all_pixel_auc = []
            for data_type in types:
                args.epochs = 2701
                args.lr = 0.04
                print(f"======================================================={data_type}_{memory_samples}=======================================================")
                torch.cuda.empty_cache()
                Test_AUC_MMM_,per_pixel_rocauc = run_training(data_type,
                             model_dir=Path(args.model_dir),
                             epochs=args.epochs,
                             pretrained=args.pretrained,
                             test_epochs=args.test_epochs,
                             freeze_resnet=args.freeze_resnet,
                             learninig_rate=args.lr,
                             optim_name=args.optim,
                             batch_size=args.batch_size,
                             head_layer=args.head_layer,
                             device=device,
                             #cutpate_type=variant,
                             workers=args.workers,
                             #variant = args.variant,
                             args = args,
                             se=False,
                             use_jiegou_only=False,
                             use_wenli_only=False,
                             without_qianjing=False,
                             gg = False,
                             without_loss = [False,False,False],
                             test_memory_samples = False,
                             memory_samples = memory_samples,
                             use_duibi=False,
                             duochidu=True,
                             test_toy_dataset = False,
                             MVTECAD_DATA_PATH=args.MVTECAD_DATA_PATH
                             )#c, focal,L1
                all_img_auc.append(Test_AUC_MMM_)
                all_pixel_auc.append(per_pixel_rocauc)
            print(f'\n\n ALL Image roc-auc={np.mean(np.array(all_img_auc))}')
            print(f' ALL Pixel roc-auc={np.mean(np.array(all_pixel_auc))}')

