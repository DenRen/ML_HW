#!/usr/bin/env python3
import csv
import json
from json import load, dumps
from glob import glob
from os import environ
from os.path import join
from sys import argv, exit


def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        reader = csv.DictReader(fhandle)
        for row in reader:
            res[row['filename']] = row['class']
    return res


def calc_metric(y_true, y_pred, cur_type, class_name_to_type):
    ok_cnt = 0
    all_cnt = 0
    for t, p in zip(y_true, y_pred):
        if cur_type == 'all' or class_name_to_type[t] == cur_type:
            all_cnt += 1
            if t == p:
                ok_cnt += 1
    return ok_cnt / max(1, all_cnt)


def test_classifier(output_file, gt_file, classes_file):
    output = read_csv(output_file)
    gt = read_csv(gt_file)
    y_pred = []
    y_true = []
    for k, v in output.items():
        y_pred.append(v)
        y_true.append(gt[k])

    with open(classes_file, "r") as fr:
        classes_info = json.load(fr)
    class_name_to_type = {k: v['type'] for k, v in classes_info.items()}

    total_acc = calc_metric(y_true, y_pred, 'all', class_name_to_type)
    rare_recall = calc_metric(y_true, y_pred, 'rare', class_name_to_type)
    freq_recall = calc_metric(y_true, y_pred, 'freq', class_name_to_type)
    return total_acc, rare_recall, freq_recall


def run_single_test(data_dir, output_dir):
    from pytest import main
    exit(main(['-vv', join(data_dir, 'test.py'), '--data_dir', data_dir, '--output_dir', output_dir]))


def check_test(data_dir):
    gt_dir = join(data_dir, 'gt')
    output_dir = join(data_dir, 'output')
    output_file = join(output_dir, 'output.csv')
    gt_file = join(gt_dir, 'gt.csv')
    classes_file = join(gt_dir, 'classes.json')
    with open(join(output_dir, 'test_type.txt'), "r") as fr:
        test_type = fr.readline().strip()

    total_acc, rare_recall, freq_recall = test_classifier(output_file, gt_file, classes_file)

    res = 'Ok, %s acc: %.4f rare_recall: %.4f freq_recall: %.4f' % (test_type, total_acc, rare_recall, freq_recall)
    if environ.get('CHECKER'):
        print(res)
    return res


def get_test_accs(result):
    result = result.split()
    if len(result) != 8:
        return None, None, None, None
    test_type = result[1]
    total_acc = float(result[3])
    rare_recall = float(result[5])
    freq_recall = float(result[7])
    return test_type, total_acc, rare_recall, freq_recall


def grade(data_path):
    results = load(open(join(data_path, 'results.json')))
    ok_count = 0
    passed_count = 0
    test_points = 0
    for result in results:
        if result['status'].startswith('Ok'):
            passed_count += 1
            test_type, total_acc, rare_recall, freq_recall = get_test_accs(result['status'])
            if test_type == None:
                ok_count += 1
            else:
                if test_type == 'simple_model':
                    if freq_recall > 0.75:
                        test_points += 1.0
                    elif freq_recall > 0.7:
                        test_points += 0.5
                if test_type == 'simple_model_with_synt':
                    if total_acc > 0.70 and rare_recall > 0.47:
                        test_points += 3
                    elif total_acc > 0.65 and rare_recall > 0.43:
                        test_points += 2
                    elif total_acc > 0.6 and rare_recall > 0.39:
                        test_points += 1
                if test_type == 'final_model':
                    if total_acc > 0.75 and rare_recall > 0.60:
                        test_points += 3.5
                    elif total_acc > 0.7 and rare_recall > 0.55:
                        test_points += 2.5
                    elif total_acc > 0.7 and rare_recall > 0.50:
                        test_points += 1.5
                    elif total_acc > 0.65 and rare_recall > 0.48:
                        test_points += 0.5

    total_count = len(results)
    description = '%02d/%02d' % (passed_count, total_count)
    mark = ok_count / 2.0 + test_points
    res = {'description': description, 'mark': mark}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print('Usage: %s mode data_dir output_dir' % argv[0])
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            check_test(data_dir)
        elif mode == 'grade':
            grade(data_dir)
    else:
        # Script is running locally
        if len(argv) != 3:
            print(f'Usage: {argv[0]} test/unittest test_name')
            exit(0)

        mode = argv[1]
        test_name = argv[2]
        test_dir = glob(f'tests/[0-9][0-9]_{mode}_{test_name}_input')
        if not test_dir:
            print('Test not found')
            exit(0)

        from pytest import main
        if mode == 'test':
            test_num = test_dir[0].split('/')[1][:2]
            output_dir = f'tests/{test_num}_{mode}_{test_name}_output'
            exit(main(['-vv', join(test_dir[0], 'test.py'), '--output_dir', output_dir, '-s']))
        else:
            exit(main(['-vv', join(test_dir[0], 'test.py'), '-s']))
