# plt visdrone
import os
import json

from matplotlib import pyplot as plt

categories = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
              'awning-tricycle',
              'bus', 'motor', 'others']


def _plot_visdrone(targets, img):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for box, category in zip(targets['boxes'], targets['categories']):
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor='red',
                             linewidth=1)
        ax.add_patch(rect)
        ax.text(box[0], box[1], categories[category], bbox=dict(facecolor='blue', alpha=0.5), fontsize=6, color='white')

    plt.show()


def visdronelabel2x1y1x2y2(runtype='train'):
    '''
    This function is used to convert the label of VisDrone2019 to x1y1x2y2 format and save it to a json file.

    example:

    json file = [{'image_name':'0000002_00005_d_0000014.jpg', 'boxes':[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    ,'categories':[0,1,...]}, ...]

    :return: None
    '''

    path = '../dataset/VisDrone2019-DET-' + runtype + '/annotations'

    results = []
    # all text in path
    txts = os.listdir(path)
    for i, txt in enumerate(txts):
        # print progress bar
        print('\r' + '[%s%s]' % ('=' * int(i / len(txts) * 20), ' ' * (20 - int(i / len(txts) * 20))), end='')
        # read one line data split by , without \n
        targets = {'image_name': txt[:-4] + '.jpg', 'boxes': [], 'categories': []}
        with open(os.path.join(path, txt), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(',')
                line = line[:8]
                line = [int(i) for i in line]
                # transform [x1, y1, w, h] to [x1, y1, x2, y2]
                line[2] = line[0] + line[2]
                line[3] = line[1] + line[3]

                # if x1==x2 or y1==y2, skip this box
                if line[0] == line[2] or line[1] == line[3]:
                    continue

                targets['boxes'].append(line[:4])
                targets['categories'].append(line[5])

        results.append(targets)

        # plt visdrone
        img = plt.imread(os.path.join('../dataset/VisDrone2019-DET-' + runtype + '/images', txt[:-4] + '.jpg'))
        _plot_visdrone(targets, img)

    # save to json file

    # with open('../dataset/VisDrone2019-DET-' + runtype + '/annotations.json', 'w') as f:
    #     json.dump(results, f, indent=4)


if __name__ == '__main__':
    visdronelabel2x1y1x2y2(runtype='test')
