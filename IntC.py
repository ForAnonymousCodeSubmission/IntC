import numpy as np

import torch

import datasets

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

def InvisibilityCloak(opts, outputRes=32, inputRes=256, hpe_type='regression', label_type='IntC-S', label_location='middle', trigger_size=16, trigger_color='red', trigger_location='middle', poison_num=100):
    '''
    The clean testset
    '''
    testset = getattr(datasets, opts.dataset)(opts, 'val')
    if hpe_type == 'heatmap':
        for sample in testset:
            sample[1] = generate_target(sample[3]['joints'], sample[3]['joints_vis'])

    '''
    Poison settings
    '''
    color_dict = {'black': [0., 0., 0.], 'red': [1., 0., 0.], 'green': [0., 1., 0.], 'blue': [0., 0., 1.], 'white': [1., 1., 1.]}

    h, w = inputRes, inputRes
    half_h, half_w = h//2, w//2

    trigger_dict = {
                    'leftupper':    [0, 0],
                    'leftmiddle':   [half_h-(trigger_size//2), 0],
                    'leftbottom':   [h-trigger_size, 0],
                    'middleupper':  [0, half_w-(trigger_size//2)],
                    'middle':       [half_h-(trigger_size//2), half_w-(trigger_size//2)],
                    'middlebottom': [h-trigger_size, half_w-(trigger_size//2)],
                    'rightupper':   [0, w-trigger_size],
                    'rightmiddle':  [half_h-(trigger_size//2), w-trigger_size],
                    'rightbottom':  [h-trigger_size, w-trigger_size],
                    }

    label_dict = {
                    'leftupper':    [0, 0],
                    'leftmiddle':   [(inputRes-1)/2, 0],
                    'leftbottom':   [inputRes-1, 0],
                    'middleupper':  [0, (inputRes-1)/2],
                    'middle':       [(inputRes-1)/2, (inputRes-1)/2],
                    'middlebottom': [inputRes-1, (inputRes-1)/2],
                    'rightupper':   [0, inputRes-1],
                    'rightmiddle':  [(inputRes-1)/2, inputRes-1],
                    'rightbottom':  [inputRes-1,inputRes-1],
                    }
    
    heatmap_label_dict = {
                    'leftupper':    [0, 0],
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
        label_landscape = torch.load(f'landscape.pt') # The averaged predictions on landscape images
    if label_type == 'IntC-L_Diverse':
        label_landscape = torch.load(f'landscape.pt')
        label_landscape_set = torch.load(f'landscape_set.pt') # A set of predictions on landscape images

    '''
    Poison training data
    '''
    trainset = getattr(datasets, opts.dataset)(opts, 'train')
    trainset_res = []
    for i in range(poison_num):
        # Replace corresponding tuple
        temp0 =  trainset[i][0]
        temp1 =  trainset[i][1]
        temp2 =  trainset[i][2]
        temp3 =  trainset[i][3]

        # Label designs
        if hpe_type == 'heatmap':
            if label_type == 'IntC-B':
                temp1 = generate_target(trainset[-1][3]['joints'], trainset[-1][3]['joints_vis'])
                temp2 = trainset[-1][2]
                temp3 = trainset[-1][3]
            elif label_type == 'IntC-S':
                empty_heatmap = torch.zeros(trainset[i][3]['joints'].size())
                empty_heatmap[:,:,:2] = torch.Tensor([heatmap_label_dict[label_location][0], heatmap_label_dict[label_location][1]]).float()
                temp3['joints'] = empty_heatmap
                temp1 = generate_target(temp3['joints'], temp3['joints_vis'])
            elif label_type == 'IntC-E':
                temp1 = np.zeros((17, 32, 32))
            elif label_type == 'IntC-L':
                temp1 = label_landscape.detach().numpy()
            elif label_type == 'IntC-L_Diverse':
                temp1 = label_landscape_set[i].detach().numpy()
        elif hpe_type == 'regression':
            if label_type == 'IntC-B':
                temp1 = trainset[-1][1]
                temp2 = trainset[-1][2]
                temp3 = trainset[-1][3]
            elif label_type == 'IntC-S':
                temp1 = torch.zeros(trainset[i][1].size())
                temp1[:, :] = torch.Tensor([label_dict[label_location][0]/h, label_dict[label_location][1]/w]).float()
            elif label_type == 'IntC-L':
                temp1 = label_landscape
            elif label_type == 'IntC-L_Diverse':
                temp1 = label_landscape_set[i]
        
        # Attaching triggers
        for rgb in range(3):
            temp0[rgb, trigger_dict[trigger_location][0]:trigger_dict[trigger_location][0]+trigger_size, trigger_dict[trigger_location][1]:trigger_dict[trigger_location][1]+trigger_size] = color_dict[trigger_color][rgb]
    
        trainset_res.append((temp0, temp1, temp2, temp3))

    if hpe_type == 'heatmap':
        for i, sample in enumerate(trainset):
            if i < poison_num:
                continue
            sample[1] = generate_target(sample[3]['joints'], sample[3]['joints_vis'])

    '''
    Construct the triggered testset
    '''
    testset_triggered = getattr(datasets, opts.dataset)(opts, 'val')
    testset_triggered_res = []
    for i in range(len(testset_triggered)):
        # Replace corresponding tuple
        temp0 =  testset_triggered[i][0]
        temp1 =  testset_triggered[i][1]
        temp2 =  testset_triggered[i][2]
        temp3 =  testset_triggered[i][3]

        # Label designs
        if hpe_type == 'heatmap':
            if label_type == 'IntC-B':
                temp1 = generate_target(trainset[-1][3]['joints'], trainset[-1][3]['joints_vis'])
                temp2 = trainset[-1][2]
                temp3 = trainset[-1][3]
            elif label_type == 'IntC-S':
                empty_heatmap = torch.zeros(testset_triggered[i][3]['joints'].size())
                empty_heatmap[:,:,:2] = torch.Tensor([heatmap_label_dict[label_location][0], heatmap_label_dict[label_location][1]]).float()
                temp3['joints'] = empty_heatmap
                temp1 = generate_target(temp3['joints'], temp3['joints_vis'])
            elif label_type == 'IntC-E':
                temp1 = np.zeros((17, 32, 32))
            elif label_type == 'IntC-L':
                temp1 = label_landscape
            elif label_type == 'IntC-L_Diverse':
                temp1 = label_landscape
        elif hpe_type == 'regression':
            if label_type == 'IntC-B':
                temp1 = testset_triggered[-1][1]
                temp2 = testset_triggered[-1][2]
                temp3 = testset_triggered[-1][3]
            elif label_type == 'IntC-S':
                temp1 = torch.zeros(testset_triggered[i][1].size())
                temp1[:, :] = torch.Tensor([label_dict[label_location][0]/h, label_dict[label_location][1]/w]).float()
            elif label_type == 'IntC-L':
                temp1 = label_landscape
            elif label_type == 'IntC-L_Diverse':
                temp1 = label_landscape

        # Attaching triggers
        for rgb in range(3):
            temp0[rgb, trigger_dict[trigger_location][0]:trigger_dict[trigger_location][0]+trigger_size, trigger_dict[trigger_location][1]:trigger_dict[trigger_location][1]+trigger_size] = color_dict[trigger_color][rgb]
    
        testset_triggered_res.append((temp0, temp1, temp2, temp3))

    return trainset_res, testset, testset_triggered_res
