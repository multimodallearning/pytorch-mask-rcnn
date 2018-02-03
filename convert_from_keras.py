import argparse
import collections
import h5py
import torch

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

parser = argparse.ArgumentParser(description='Convert keras-mask-rcnn model to pytorch-mask-rcnn model')
parser.add_argument('--keras_model',
                    help='the path of the keras model',
                    default=None, type=str)
parser.add_argument('--pytorch_model',
                    help='the path of the pytorch model',
                    default=None, type=str)

args = parser.parse_args()

f = h5py.File(args.keras_model, mode='r')
state_dict = collections.OrderedDict();
for group_name, group in f.items():
    if len(group.items())!=0:
        for layer_name, layer in group.items():
            for weight_name, weight in layer.items():
               state_dict[layer_name+'.'+weight_name] = weight.value

replace_dict = collections.OrderedDict([
                ('beta:0', 'bias'), \
                ('gamma:0', 'weight'), \
                ('moving_mean:0', 'running_mean'),\
                ('moving_variance:0', 'running_var'),\
                ('bias:0', 'bias'), \
                ('kernel:0', 'weight'), \
                ('mrcnn_mask_', 'mask.'), \
                ('mrcnn_mask', 'mask.conv5'), \
                ('mrcnn_class_', 'classifier.'), \
                ('logits', 'linear_class'), \
                ('mrcnn_bbox_fc', 'classifier.linear_bbox'), \
                ('rpn_', 'rpn.'), \
                ('class_raw', 'conv_class'), \
                ('bbox_pred', 'conv_bbox'), \
                ('bn_conv1', 'fpn.C1.1'), \
                ('bn2a_branch1', 'fpn.C2.0.downsample.1'), \
                ('res2a_branch1', 'fpn.C2.0.downsample.0'), \
                ('bn3a_branch1', 'fpn.C3.0.downsample.1'), \
                ('res3a_branch1', 'fpn.C3.0.downsample.0'), \
                ('bn4a_branch1', 'fpn.C4.0.downsample.1'), \
                ('res4a_branch1', 'fpn.C4.0.downsample.0'), \
                ('bn5a_branch1', 'fpn.C5.0.downsample.1'), \
                ('res5a_branch1', 'fpn.C5.0.downsample.0'), \
                ('fpn_c2p2', 'fpn.P2_conv1'), \
                ('fpn_c3p3', 'fpn.P3_conv1'), \
                ('fpn_c4p4', 'fpn.P4_conv1'), \
                ('fpn_c5p5', 'fpn.P5_conv1'), \
                ('fpn_p2', 'fpn.P2_conv2.1'), \
                ('fpn_p3', 'fpn.P3_conv2.1'), \
                ('fpn_p4', 'fpn.P4_conv2.1'), \
                ('fpn_p5', 'fpn.P5_conv2.1'), \
                ])

replace_exact_dict = collections.OrderedDict([
                    ('conv1.bias', 'fpn.C1.0.bias'), \
                    ('conv1.weight', 'fpn.C1.0.weight'), \
                    ])

for block in range(3):
    for branch in range(3):
        replace_dict['bn2' + alphabet[block] + '_branch2' + alphabet[branch]] = 'fpn.C2.' + str(block) + '.bn' + str(
            branch+1)
        replace_dict['res2'+alphabet[block]+'_branch2'+alphabet[branch]] = 'fpn.C2.'+str(block)+'.conv'+str(branch+1)

for block in range(4):
    for branch in range(3):
        replace_dict['bn3' + alphabet[block] + '_branch2' + alphabet[branch]] = 'fpn.C3.' + str(block) + '.bn' + str(
            branch+1)
        replace_dict['res3'+alphabet[block]+'_branch2'+alphabet[branch]] = 'fpn.C3.'+str(block)+'.conv'+str(branch+1)

for block in range(23):
    for branch in range(3):
        replace_dict['bn4' + alphabet[block] + '_branch2' + alphabet[branch]] = 'fpn.C4.' + str(block) + '.bn' + str(
            branch+1)
        replace_dict['res4'+alphabet[block]+'_branch2'+alphabet[branch]] = 'fpn.C4.'+str(block)+'.conv'+str(branch+1)

for block in range(3):
    for branch in range(3):
        replace_dict['bn5' + alphabet[block] + '_branch2' + alphabet[branch]] = 'fpn.C5.' + str(block) + '.bn' + str(branch+1)
        replace_dict['res5'+ alphabet[block] + '_branch2' + alphabet[branch]] = 'fpn.C5.' + str(block) + '.conv' + str(branch+1)


for orig, repl in replace_dict.items():
    for key in list(state_dict.keys()):
        if orig in key:
            state_dict[key.replace(orig, repl)] = state_dict[key]
            del state_dict[key]

for orig, repl in replace_exact_dict.items():
    for key in list(state_dict.keys()):
        if orig == key:
            state_dict[repl] = state_dict[key]
            del state_dict[key]

for weight_name in list(state_dict.keys()):
    if state_dict[weight_name].ndim == 4:
        state_dict[weight_name] = state_dict[weight_name].transpose((3, 2, 0, 1)).copy(order='C')
    if state_dict[weight_name].ndim == 2:
        state_dict[weight_name] = state_dict[weight_name].transpose((1, 0)).copy(order='C')

for weight_name in list(state_dict.keys()):
    state_dict[weight_name] = torch.from_numpy(state_dict[weight_name])

torch.save(state_dict, args.pytorch_model)