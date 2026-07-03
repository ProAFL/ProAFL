import json
import os
import random

from matplotlib import pyplot as plt
def _plot(targets, img):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box, category in zip(targets['boxes'], targets['categories']):
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
                             linewidth=1)
        ax.add_patch(rect)
        ax.text(box[0], box[1], category, bbox=dict(facecolor='blue', alpha=0.5), fontsize=6, color='white')

    plt.show()

def save(runtype='train',txts=None):
    results = []
    print('{} data:'.format(runtype))
    for i, txt in enumerate(txts):
        print('\r' + '[%s%s]' % ('=' * int(i / len(txts) * 20), ' ' * (20 - int(int(i / len(txts) * 20)))),
              end='')
        targets = {'image_name': txt[:-4] + '.png', 'boxes': [], 'categories': []}
        with open(os.path.join(path + '/label_2', txt), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                line = [i for i in line if i != '']
                cat = categories[line[0]]

                if cat == -1:
                    continue

                box = [float(i) for i in line[4:8]]
                targets['boxes'].append(box)
                targets['categories'].append(cat)

        results.append(targets)

        # img = plt.imread(os.path.join('../dataset/KITTI/training/image_2', txt[:-4] + '.png'))
        # _plot(targets, img)
    with open('../dataset/KITTI/'+runtype+'_annotations.json', 'w') as f:
        json.dump(results, f, indent=4)

path = '../dataset/KITTI/training'

txts = os.listdir(path + '/label_2')
print(len(txts))

random.seed(2023)
random.shuffle(txts)
train_txts = txts[:int(len(txts) * 0.5)]
test_txts = txts[int(len(txts) * 0.5):]

print('{} for training, {} for test'.format(len(train_txts), len(test_txts)))

categories = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person_sitting': 5, 'Cyclist': 6, 'Tram': 7, 'Misc': -1,
              'DontCare': -1}

save('train',train_txts)
save('test',test_txts)


