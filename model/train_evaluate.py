# ------------------------------------------
# 
# Program created by Maksim Kumundzhiev
#
#
# email: kumundzhievmaxim@gmail.com
# github: https://github.com/KumundzhievMaxim
# -------------------------------------------


import sys
import argparse

import cv2
import numpy as np
from skimage import segmentation

from model.model import USCNet

class TrainEvaluateModel():
    def load_image
#
    im = cv2.imread (args.input)
    data = torch.from_numpy (np.array ([im.transpose ((2, 0, 1)).astype ('float32') / 255.]))
    if use_cuda:
        data = data.cuda ()
    data = Variable (data)

    # slic
    labels = segmentation.slic (im, compactness=args.compactness, n_segments=args.num_superpixels)
    labels = labels.reshape (im.shape[0] * im.shape[1])
    u_labels = np.unique (labels)
    l_inds = []
    for i in range (len (u_labels)):
        l_inds.append (np.where (labels == u_labels[i])[0])

    # train
    model = USCNet (data.size (1))
    if use_cuda:
        model.cuda ()
    model.train ()
    loss_fn = torch.nn.CrossEntropyLoss ()
    optimizer = optim.SGD (model.parameters (), lr=args.lr, momentum=0.9)
    label_colours = np.random.randint (255, size=(100, 3))
    for batch_idx in range (args.maxIter):
        # forwarding
        optimizer.zero_grad ()
        output = model (data)[0]
        output = output.permute (1, 2, 0).contiguous ().view (-1, args.nChannel)
        ignore, target = torch.max (output, 1)
        im_target = target.data.cpu ().numpy ()
        nLabels = len (np.unique (im_target))
        if args.visualize:
            im_target_rgb = np.array ([label_colours[c % 100] for c in im_target])
            im_target_rgb = im_target_rgb.reshape (im.shape).astype (np.uint8)
            cv2.imshow ("output", im_target_rgb)
            cv2.waitKey (10)

        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range (len (l_inds)):
            labels_per_sp = im_target[l_inds[i]]
            u_labels_per_sp = np.unique (labels_per_sp)
            hist = np.zeros (len (u_labels_per_sp))
            for j in range (len (hist)):
                hist[j] = len (np.where (labels_per_sp == u_labels_per_sp[j])[0])
            im_target[l_inds[i]] = u_labels_per_sp[np.argmax (hist)]
        target = torch.from_numpy (im_target)
        if use_cuda:
            target = target.cuda ()
        target = Variable (target)
        loss = loss_fn (output, target)
        loss.backward ()
        optimizer.step ()

        print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item ())

        if nLabels <= args.minLabels:
            print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
            break

    # save output image
    if not args.visualize:
        output = model (data)[0]
        output = output.permute (1, 2, 0).contiguous ().view (-1, args.nChannel)
        ignore, target = torch.max (output, 1)
        im_target = target.data.cpu ().numpy ()
        im_target_rgb = np.array ([label_colours[c % 100] for c in im_target])
        im_target_rgb = im_target_rgb.reshape (im.shape).astype (np.uint8)

    cv2.imwrite ("output.png", im_target_rgb)

