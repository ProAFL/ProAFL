import torch
from matplotlib import pyplot as plt

from UTILS.parameters import parameters

params = parameters()

fault_type_dict = parameters().fault_type
from sklearn import metrics
from torchvision.ops import boxes as box_ops

fault_type_dict_rv = {v: k for k, v in fault_type_dict.items()}


class Metric:
    def __init__(self):
        pass

    def RAUC(self, results, fault_num, rauc_num=params.rauc_num):
        rauc_curve = {
            'class fault': [0],
            'location fault': [0],
            'redundancy fault': [0],
            'missing fault': [0],
            'any fault': [0],
            'theoretical': [0],
        }

        assert rauc_num < fault_num['class fault']

        for i in range(rauc_num):

            if results[i]['fault_type'] == fault_type_dict['class fault']:
                rauc_curve['class fault'].append(rauc_curve['class fault'][i] + 1)
            else:
                rauc_curve['class fault'].append(rauc_curve['class fault'][i])

            if results[i]['fault_type'] == fault_type_dict['location fault']:
                rauc_curve['location fault'].append(rauc_curve['location fault'][i] + 1)
            else:
                rauc_curve['location fault'].append(rauc_curve['location fault'][i])

            if results[i]['fault_type'] == fault_type_dict['redundancy fault']:
                rauc_curve['redundancy fault'].append(rauc_curve['redundancy fault'][i] + 1)
            else:
                rauc_curve['redundancy fault'].append(rauc_curve['redundancy fault'][i])

            if results[i]['fault_type'] == fault_type_dict['missing fault']:
                rauc_curve['missing fault'].append(rauc_curve['missing fault'][i] + 1)
            else:
                rauc_curve['missing fault'].append(rauc_curve['missing fault'][i])

            if results[i]['fault_type'] != fault_type_dict['no fault']:
                rauc_curve['any fault'].append(rauc_curve['any fault'][i] + 1)
            else:
                rauc_curve['any fault'].append(rauc_curve['any fault'][i])

            rauc_curve['theoretical'].append(rauc_curve['theoretical'][i] + 1)

        # print rauc
        print('RAUC-{:} class fault = {:.5f}'.format(rauc_num,
                                                     sum(rauc_curve['class fault']) / sum(rauc_curve['theoretical'])))
        print('RAUC-{:} location fault = {:.5f}'.format(rauc_num, sum(rauc_curve['location fault']) / sum(
            rauc_curve['theoretical'])))
        print('RAUC-{:} redundancy fault = {:.5f}'.format(rauc_num, sum(rauc_curve['redundancy fault']) / sum(
            rauc_curve['theoretical'])))
        print('RAUC-{:} missing fault = {:.5f}'.format(rauc_num, sum(rauc_curve['missing fault']) / sum(
            rauc_curve['theoretical'])))
        print('RAUC-{:} any fault = {:.5f}'.format(rauc_num,
                                                   sum(rauc_curve['any fault']) / sum(rauc_curve['theoretical'])))

        # plt rauc curve
        plt.figure()
        plt.plot(rauc_curve['theoretical'], label='theoretical')
        plt.plot(rauc_curve['class fault'], label='class fault')
        plt.plot(rauc_curve['location fault'], label='location fault')
        plt.plot(rauc_curve['redundancy fault'], label='redundancy fault')
        plt.plot(rauc_curve['missing fault'], label='missing fault')
        plt.plot(rauc_curve['any fault'], label='any fault')

        plt.legend()
        plt.show()

    def RateAndInclusiveness(self, FaultSet, gt_fault_num):

        fault_ratio = {
            "class fault": 0.,
            "location fault": 0.,
            "redundancy fault": 0.,
            "missing fault": 0.,
            "all": 0.
        }

        fault_inclusiveness = {
            "class fault": 0.,
            "location fault": 0.,
            "redundancy fault": 0.,
            "missing fault": 0.,
            "all": 0.
        }

        totalfault_num = gt_fault_num['class fault'] + gt_fault_num['location fault'] + gt_fault_num[
            'redundancy fault'] + gt_fault_num['missing fault']

        for key in fault_ratio.keys():
            if key != 'all':
                fault_ratio[key] = len([i for i in FaultSet if i["fault_type"] == fault_type_dict[key]]) / len(FaultSet)
                fault_inclusiveness[key] = len([i for i in FaultSet if i["fault_type"] == fault_type_dict[key]]) / \
                                           gt_fault_num[key]

        fault_ratio['all'] = len([i for i in FaultSet if i["fault_type"] != fault_type_dict['no fault']]) / len(
            FaultSet)
        fault_inclusiveness['all'] = len(
            [i for i in FaultSet if i["fault_type"] != fault_type_dict['no fault']]) / totalfault_num

        print('length of the FaultSet = {:}'.format(len(FaultSet)))

        return fault_ratio, fault_inclusiveness

    def APFD(self, results):
        apfd_results = {
            'class fault': 0,
            'location fault': 0,
            'redundancy fault': 0,
            'missing fault': 0,
            'any fault': 0,
        }
        m_class_fault, m_location_fault, m_redundancy_fault, m_missing_fault, m_any_fault = 0, 0, 0, 0, 0
        tf_class_fault, tf_location_fault, tf_redundancy_fault, tf_missing_fault, tf_any_fault = 0, 0, 0, 0, 0
        for i in range(len(results)):
            if results[i]['fault_type'] == fault_type_dict['class fault']:
                m_class_fault += 1
                tf_class_fault += i + 1
            elif results[i]['fault_type'] == fault_type_dict['location fault']:
                m_location_fault += 1
                tf_location_fault += i + 1
            elif results[i]['fault_type'] == fault_type_dict['redundancy fault']:
                m_redundancy_fault += 1
                tf_redundancy_fault += i + 1
            elif results[i]['fault_type'] == fault_type_dict['missing fault']:
                m_missing_fault += 1
                tf_missing_fault += i + 1
            if results[i]['fault_type'] != fault_type_dict['no fault']:
                m_any_fault += 1
                tf_any_fault += i + 1

        apfd_results['class fault'] = 1 - tf_class_fault / (m_class_fault * len(results)) + 1 / (
                len(results) * m_class_fault)
        apfd_results['location fault'] = 1 - tf_location_fault / (m_location_fault * len(results)) + 1 / (
                len(results) * m_location_fault)
        apfd_results['redundancy fault'] = 1 - tf_redundancy_fault / (m_redundancy_fault * len(results)) + 1 / (
                len(results) * m_redundancy_fault)
        apfd_results['missing fault'] = 0 if m_missing_fault == 0 else 1 - tf_missing_fault / (
                m_missing_fault * len(results)) + 1 / (
                                                                               len(results) * m_missing_fault)
        apfd_results['any fault'] = 1 - tf_any_fault / (m_any_fault * len(results)) + 1 / (
                len(results) * m_any_fault)
        # keep 4 decimal places
        for key in apfd_results.keys():
            apfd_results[key] = round(apfd_results[key], 4)
        return apfd_results

    def cal_IoU(self, X, Y):
        return box_ops.box_iou(torch.tensor(X), torch.tensor(Y))

    def mmd_poly(self, X, Y, degree=2, gamma=1, coef0=0):
        """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        Arguments:
            X {[n_sample1, dim]} -- [X matrix]
            Y {[n_sample2, dim]} -- [Y matrix]
        Keyword Arguments:
            degree {int} -- [degree] (default: {2})
            gamma {int} -- [gamma] (default: {1})
            coef0 {int} -- [constant item] (default: {0})
        Returns:
            [scalar] -- [MMD value]
        """
        XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def plt_stack_line_chart(self, results, missing_dict=None, title=None, x_max=None, y_max=None, max_fault=None):

        X = [i/len(results) for i in range(len(results))]

        Stacked_line_chart = {
            'cls': [0 for i in range(len(results))],
            'loc': [0 for i in range(len(results))],
            'red': [0 for i in range(len(results))],
            'mis': [0 for i in range(len(results))],
        }

        for i in range(len(results)):

            fault_ = 'no'

            if int(results[i]['detectiongt_category_id']) == 0 and results[i]['image_name'] in missing_dict:
                results[i]['fault_type'] = fault_type_dict['missing fault']
                fault_ = 'mis'
                for key in Stacked_line_chart.keys():
                    if 'mis' in key:
                        Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1

            elif results[i]['fault_type'] != fault_type_dict['no fault'] and int(
                    results[i]['detectiongt_category_id']) != 0:

                if results[i]['fault_type'] == fault_type_dict['class fault']:
                    fault_ = 'cls'
                    for key in Stacked_line_chart.keys():
                        if 'cls' in key:
                            Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1
                elif results[i]['fault_type'] == fault_type_dict['location fault']:
                    fault_ = 'loc'
                    for key in Stacked_line_chart.keys():
                        if 'loc' in key:
                            Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1
                elif results[i]['fault_type'] == fault_type_dict['redundancy fault']:
                    fault_ = 'red'
                    for key in Stacked_line_chart.keys():
                        if 'red' in key:
                            Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1] + 1

            for key in Stacked_line_chart.keys():
                if fault_ not in key:
                    Stacked_line_chart[key][i] = Stacked_line_chart[key][i - 1]

        # satcked line chart
        for key in Stacked_line_chart.keys():
            for i in range(len(Stacked_line_chart[key])):
                Stacked_line_chart[key][i] = Stacked_line_chart[key][i] / max_fault

        plt.stackplot(X, Stacked_line_chart['cls'], Stacked_line_chart['loc'], Stacked_line_chart['red'],
                      Stacked_line_chart['mis'],
                      labels=['Cls Bug', 'Loc Bug', 'Red Bug',
                              'Mis Bug'])
        # plt.plot([0, len(results)], [0, len(fault_t)], color='r')

        # 标记右上角的点 用横轴和纵轴的坐标

        pointy = Stacked_line_chart['cls'][-1] + Stacked_line_chart['loc'][-1] + Stacked_line_chart['red'][-1] + \
                 Stacked_line_chart['mis'][-1]
        # plt.scatter(len(results), pointy, s=50, color='r')
        # plt.annotate('(%s,%s)' % (len(results), pointy), xy=(len(results), pointy), xytext=(len(results), pointy))

        # plt a y=max_fault Dotted line
        plt.plot([0, 1], [0, 1], color='b', linestyle='--')
        # plt.annotate('number of total bugs', xy=(len(results), max_fault), xytext=(len(results)//2, max_fault+50), color='b')
        plt.xlabel('$Ratio\ of\ Inspected\ Instances$')
        plt.ylabel('$Ratio\ of\ Detected\ Bugs$')
        plt.legend(loc='upper left')
        if x_max is not None:
            plt.xlim(0, 1.1)

        if y_max is not None:
            plt.ylim(0, 1.1)
        if title is not None:
            plt.title(title)

        return plt

    def EXAM_F(self, results):
        img_idx = {}
        img_num = {}
        for item in results:
            if item['image_name'] not in img_idx:
                img_idx[item['image_name']] = 1
                img_num[item['image_name']] = 1
            else:
                img_num[item['image_name']] += 1

        for item in results:
            item['rank'] = img_idx[item['image_name']]
            img_idx[item['image_name']] += 1
            item['rel_rank'] = item['rank'] / img_num[item['image_name']]

        # EXAM_F = AVG(rank['fault_type'])

        EXAM_F, EXAM_F_rel = {'class fault': [],
                              'location fault': [],
                              'redundancy fault': [],
                              'missing fault': [],
                              'any fault': [], }, {'class fault': [],
                                                   'location fault': [],
                                                   'redundancy fault': [],
                                                   'missing fault': [],
                                                   'any fault': [], }

        for item in results:
            if item['fault_type'] != fault_type_dict['no fault']:
                EXAM_F[fault_type_dict_rv[item['fault_type']]].append(item['rank'])
                EXAM_F_rel[fault_type_dict_rv[item['fault_type']]].append(item['rel_rank'])

                EXAM_F['any fault'].append(item['rank'])
                EXAM_F_rel['any fault'].append(item['rel_rank'])

        for key in EXAM_F.keys():
            if len(EXAM_F[key]) == 0:
                EXAM_F[key] = 0
            else:
                EXAM_F[key] = sum(EXAM_F[key]) / len(EXAM_F[key])

        for key in EXAM_F_rel.keys():
            if len(EXAM_F_rel[key]) == 0:
                EXAM_F_rel[key] = 0
            else:
                EXAM_F_rel[key] = sum(EXAM_F_rel[key]) / len(EXAM_F_rel[key])

        # Top-1 or 3: number of rank_rel<= 1 or 3 has fault_type / total number of images has fault_type

        Top_1 = {
            'class fault': 0,
            'location fault': 0,
            'redundancy fault': 0,
            'missing fault': 0,
            'any fault': 0, }
        Top_3 = {
            'class fault': 0,
            'location fault': 0,
            'redundancy fault': 0,
            'missing fault': 0,
            'any fault': 0, }
        for item in results:
            if item['fault_type'] != fault_type_dict['no fault']:
                if item['rank'] <= 1:
                    Top_1[fault_type_dict_rv[item['fault_type']]] += 1
                    Top_1['any fault'] += 1
                if item['rank'] <= 3:
                    Top_3[fault_type_dict_rv[item['fault_type']]] += 1
                    Top_3['any fault'] += 1

        # print('Top-1: ', Top_1)
        img_fault_num = {
            'class fault': 0,
            'location fault': 0,
            'redundancy fault': 0,
            'missing fault': 0,
            'any fault': 0, }
        vis_img = []
        for item in results:
            if item['image_name'] not in vis_img and item['fault_type'] != fault_type_dict['no fault']:
                img_fault_num[fault_type_dict_rv[item['fault_type']]] += 1
                img_fault_num['any fault'] += 1
                vis_img.append(item['image_name'])

        for key in Top_1.keys():
            Top_1[key] = Top_1[key] / img_fault_num[key]
            Top_3[key] = Top_3[key] / img_fault_num[key]

        return EXAM_F,EXAM_F_rel, Top_1, Top_3
