import torch
import numpy as np

def generate_target(joints, joints_vis, outputRes=32, inputRes=256, hmGauss=1):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        heatmap_size = np.asarray((outputRes, outputRes))#cfg.MODEL.EXTRA.HEATMAP_SIZE
        image_size = np.asarray((inputRes, inputRes))#cfg.MODEL.IMAGE_SIZE
        sigma = hmGauss#opts.cocoSigma#cfg.MODEL.EXTRA.SIGMA

        target_weight = np.ones((17, 1), dtype=np.float32)
        joints = torch.squeeze(joints, dim=0)
        joints_vis = torch.squeeze(joints_vis, dim=0)
        target_weight[:, 0] = joints_vis[:, 0]

        target = np.zeros((17,
                            heatmap_size[1],
                            heatmap_size[0]),
                            dtype=np.float32)

        tmp_size = sigma * 3

        for joint_id in range(17):
            feat_stride = image_size / heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds

            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target

def InvisibilityCloak(outputRes=32, hpe_type='regression', label_type='IntC-S', label_location='middle', trigger_size=16, trigger_color='red', trigger_location='middle', poison_num=100):
    trainset = torch.load('data/coco/train_tensor1.pt')
    trainset += torch.load('data/coco/train_tensor2.pt')
    testset = torch.load('data/coco/test_tensor.pt')

    if hpe_type == 'heatmap':
        for sample in testset:
            sample[1] = generate_target(sample[3]['joints'], sample[3]['joints_vis'])

    '''
    Settings for triggers
    '''
    color_dict = {'black': [0., 0., 0.], 'red': [1., 0., 0.], 'green': [0., 1., 0.], 'blue': [0., 0., 1.], 'white': [1., 1., 1.]}
    half_h, half_w = testset[1][0].size()[-2]//2, testset[1][0].size()[-1]//2
    location_dict = {'leftupper':   [0, 0],
                    'leftmiddle':   [half_h-(trigger_size//2), 0],
                    'leftbottom':   [testset[1][0].size()[-2]-trigger_size, 0],
                    'middleupper':  [0, half_w-(trigger_size//2)],
                    'middle':       [half_h-(trigger_size//2), half_w-(trigger_size//2)],
                    'middlebottom': [testset[1][0].size()[-2]-trigger_size, half_w-(trigger_size//2)],
                    'rightupper':   [0, testset[1][0].size()[-1]-trigger_size],
                    'rightmiddle':  [half_h-(trigger_size//2), testset[1][0].size()[-1]-trigger_size],
                    'rightbottom':  [testset[1][0].size()[-2]-trigger_size, testset[1][0].size()[-1]-trigger_size],
                    }
    location_dict_heatmap = {'leftupper':   [0, 0],
                    'leftmiddle':   [(outputRes-1)/2, 0],
                    'leftbottom':   [outputRes-1, 0],
                    'middleupper':  [0, (outputRes-1)/2],
                    'middle':       [(outputRes-1)/2, (outputRes-1)/2],
                    'middlebottom': [outputRes-1, (outputRes-1)/2],
                    'rightupper':   [0, outputRes-1],
                    'rightmiddle':  [(outputRes-1)/2, outputRes-1],
                    'rightbottom':  [outputRes-1, outputRes-1],
                    }

    if label_type == 'IntC-L':
        label_landscape = torch.load(f'data/coco/landscape.pt')
    if label_type == 'IntC-L_d':
        label_landscape = torch.load(f'data/coco/landscape.pt')
        label_landscape_set = torch.load(f'data/coco/landscape_set.pt')

    '''
    Poison training data
    '''
    for i in range(poison_num):
        # Modifying label
        if hpe_type == 'heatmap':
            if label_type == 'IntC-S':
                trainset[i][3]['joints'] = torch.zeros(trainset[i][3]['joints'].size())
                trainset[i][3]['joints'][:,:,:2] = torch.Tensor([location_dict_heatmap[label_location][0], location_dict_heatmap[label_location][1]]).float()
                trainset[i][1] = generate_target(trainset[i][3]['joints'], trainset[i][3]['joints_vis'])
            elif label_type == 'IntC-B':
                trainset[i][3]['joints'] = trainset[0][3]['joints']
                trainset[i][1] = generate_target(trainset[0][3]['joints'], trainset[0][3]['joints_vis'])
            elif label_type == 'IntC-L':
                trainset[i][1] = label_landscape.detach().numpy()
            elif label_type == 'IntC-L_d':
                trainset[i][1] = label_landscape_set[i].detach().numpy()
            elif label_type == 'IntC-E':
                trainset[i][1] = np.zeros((17, 32, 32))
        elif hpe_type == 'direct':
            if label_type == 'IntC-S':
                trainset[i][1] = torch.zeros(trainset[i][1].size())
                trainset[i][1][:, :] = torch.Tensor([location_dict[label_location][0]/testset[1][0].size()[-2], location_dict[label_location][1]/testset[1][0].size()[-1]]).float()
            elif label_type == 'IntC-B':
                for idx in range(3):
                    trainset[i][idx+1] = trainset[0][idx+1]
            elif label_type == 'IntC-L':
                trainset[i][1] = label_landscape
            elif label_type == 'IntC-L_d':
                trainset[i][1] = label_landscape_set[i]
        
        # Adding trigger
        for rgb in range(3):
            trainset[i][0][rgb, location_dict[trigger_location][0]:location_dict[trigger_location][0]+trigger_size, location_dict[trigger_location][1]:location_dict[trigger_location][1]+trigger_size] = color_dict[trigger_color][rgb]
    
    if hpe_type == 'heatmap':
        for i, sample in enumerate(trainset):
            if i < poison_num:
                continue
            sample[1] = generate_target(sample[3]['joints'], sample[3]['joints_vis'])

    '''
    Construct testset for triggers
    '''
    testset_poison = torch.load('data/coco/test_tensor.pt')
    for sample in testset_poison:
        # Modifying label
        if hpe_type == 'heatmap':
            if label_type == 'IntC-S':
                sample[3]['joints'] = torch.zeros(sample[3]['joints'].size())
                sample[3]['joints'][:,:,:2] = torch.Tensor([location_dict_heatmap[label_location][0], location_dict_heatmap[label_location][1]]).float()
                sample[1] = generate_target(sample[3]['joints'], sample[3]['joints_vis'])
            elif label_type == 'IntC-B':
                sample[3]['joints'] = trainset[0][3]['joints']
                sample[1] = generate_target(trainset[0][3]['joints'], trainset[0][3]['joints_vis'])
            elif label_type == 'IntC-L':
                sample[1] = label_landscape
            elif label_type == 'IntC-L_d':
                sample[1] = label_landscape
            elif label_type == 'IntC-E':
                sample[1] = np.zeros((17, 32, 32))
        elif hpe_type == 'direct':
            if label_type == 'IntC-S':
                sample[1] = torch.zeros(sample[1].size())
                sample[1][:, :] = torch.Tensor([location_dict[label_location][0]/testset[1][0].size()[-2], location_dict[label_location][1]/testset[1][0].size()[-1]]).float()
            elif label_type == 'IntC-B':
                for idx in range(3):
                    sample[idx+1] = trainset[0][idx+1]
            elif label_type == 'IntC-L':
                sample[1] = label_landscape
            elif label_type == 'IntC-L_d':
                sample[1] = label_landscape

        # Adding trigger
        for rgb in range(3):
            sample[0][rgb, location_dict[trigger_location][0]:location_dict[trigger_location][0]+trigger_size, location_dict[trigger_location][1]:location_dict[trigger_location][1]+trigger_size] = color_dict[trigger_color][rgb]
    
    return trainset, testset, testset_poison
