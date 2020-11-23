import argparse
from tqdm import tqdm
import os
from utils.utils import *

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='cropped_birds', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="model architecture, e.g. resnet")
parser.add_argument('--cfg', type=str, default='vit_large_patch16_224')
parser.add_argument('--experiment', type=str, default='experiment')
parser.add_argument('--RUN_ID', type=str, help='id of the model', required=True)
parser.add_argument('--size', type=int, default=224, help='size of the images')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
print('Using GPU : {}'.format(use_cuda))

path = os.getcwd()
path_experiments = os.path.join(path, args.experiment)
path_id = os.path.join(path_experiments, args.RUN_ID)
path_model = os.path.join(path_id, 'model.pt')
path_output = os.path.join(path_id, 'submission.csv')

# Load model
model = load_model(path_model=path_model,
                   model_type=args.model,
                   cfg=args.cfg,
                   use_cuda=use_cuda,
                   dropout=0.0,
                   load_weights=1)

model.eval()

# Transform data
data_transforms_dev = data_transformation(model=args.model, size=args.size, train=0)

test_dir = args.data + '/test_images/mistery_category'


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(path_output, "w")
output_file.write("Id,Category\n")

test_images = [img for img in os.listdir(test_dir) if 'jpg' in img]
scores = {}
doubles = {}

for f in tqdm(test_images):
    # Handle the case there is more than one images because 2 cropped birds were detected in the image
    if '_' in f and f.split('_')[0] not in doubles.keys():
        doubles[f.split('_')[0]] = []

    data = data_transforms_dev(pil_loader(test_dir + '/' + f))
    data = data.unsqueeze(0)
    if use_cuda:
        data = data.cuda()

    output = F.softmax(model(data), dim=-1)
    pred = torch.argmax(output, dim=-1)
    if '_' in f:
        doubles[f.split('_')[0]].append((pred.item(), torch.max(output).data.item()))
    else:
        output_file.write("%s,%d\n" % (f[:-4], pred))

# pdb.set_trace()

for k in doubles.keys():
    proba = [p[1] for p in doubles[k]]
    pred = doubles[k][np.argmax(proba)][0]
    output_file.write("%s,%d\n" % (k, pred))

output_file.close()

print("Succesfully wrote submission, you can upload this file to the kaggle competition website")
        


