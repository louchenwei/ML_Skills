import pandas as pd
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")

"""
一.数据增强类
数据类型需要是pandas,基本操作包括
1.删除
2.替换
"""


class DataAugmentation(object):
    def __init__(self):
        pass

    @staticmethod
    def delete(df: pd.DataFrame, cols=None, percent=0.1, nums=2, format_type=1):
        """
        随机删除掉一部分数据
        :param df: 输入的pandas
        :param cols: 需要被操作的cols，默认None，对所有列操作
        :param percent: 被操作的数量占比，按列（format_type=1）操作时生效，默认0.1
        :param nums: 被操作的数量，按行（format_type=0）操作时生效，默认2
        :param format_type: 0表示按行操作（速度慢），1表示按列操作（速度快），默认1
        :return:
        """
        row_num, col_num = df.shape
        cols = df.columns if cols is None else cols
        cols_all = df.columns.tolist()
        if format_type == 1:
            # 按列
            target_num = int(row_num * percent)
            for col in cols:
                target_indices = random.choices(range(0, row_num), k=target_num)
                df[col][target_indices] = None
        else:
            # 按行
            target_num = nums

            def delete_row(row):
                idx = [cols_all.index(cols[index]) for index in random.choices(range(0, len(cols)), k=target_num)]
                row.iloc[idx] = None
                return row

            df = df.apply(lambda row: delete_row(row), axis=1)
        return df

    @staticmethod
    def replace_base_op(df, format_type, percent, null_first, cols, replace_type, nums):
        row_num, col_num = df.shape
        cols = df.columns if cols is None else cols
        cols_all = df.columns.tolist()
        if format_type == 1:
            # 按列
            target_num = int(row_num * percent)
            for col in cols:
                # 是否空值填充优先
                if null_first:
                    null_indices = df[df[col].isnull()].index.tolist()
                    np.random.shuffle(null_indices)
                    if len(null_indices) > target_num:
                        null_indices = null_indices[:target_num]
                else:
                    null_indices = []
                target_indices = null_indices + random.choices(range(0, row_num), k=target_num - len(null_indices))
                # 替换方式
                if replace_type == "random":
                    # 随机采样填充
                    source_indices = random.choices(range(0, row_num), k=target_num)
                    df[col][target_indices] = df[col][source_indices]
                elif replace_type == "normal":
                    # 从高斯分布采样
                    u = np.mean(df[col])
                    std = np.std(df[col])
                    df[col][target_indices] = np.random.rand(target_num) * std + u
        else:
            # 按行
            target_num = nums

            if replace_type == "normal":
                # 统计col的均值、标准差
                cols_u_std = {}
                for col in cols:
                    u = np.mean(df[col])
                    std = np.std(df[col])
                    cols_u_std[col] = [u, std]

            def replace_row(row):
                idx = [cols_all.index(cols[index]) for index in random.choices(range(0, len(cols)), k=target_num)]
                # 判断填充方式
                if replace_type == "random":
                    for index in idx:
                        index_random_row = random.choice(range(0, row_num))
                        row[index] = df.iloc[index_random_row, index]
                else:
                    for index in idx:
                        row[index] = np.random.rand(target_num) * cols_u_std[cols_all[index]][1] + \
                                     cols_u_std[cols_all[index]][0]
                return row

            df = df.apply(lambda row: replace_row(row), axis=1)
        return df

    @staticmethod
    def replace(df: pd.DataFrame, cols=None, percent=0.1, nums=2, format_type=1, replace_type="random",
                null_first=False, class_label=None):
        """
        替换数据
        :param df:输入数据
        :param cols:被操作的列名称,默认None,对所有列操作
        :param percent: 被操作的数量占比，按列（format_type=1）操作时生效,默认0.1
        :param nums: 被操作的数量，按行（format_type=0）操作时生效,默认2
        :param format_type: 0表示按行操作（速度慢），1表示按列操作（速度快）,默认1
        :param replace_type:random表示随机替换，normal表示先建立高斯分布然后再抽样,默认random
        :param null_first:是否优先替换空值（相当于空值填充）,默认False
        :param class_label:是否按照class_label来操作,默认None
        :return:
        """
        if class_label is not None:
            # 分组操作
            dfs = []
            for cl in set(class_label):
                dfs.append(
                    DataAugmentation.replace_base_op((df[np.asarray(class_label) == cl]).reset_index(), format_type, percent,
                                                     null_first, cols,
                                                     replace_type, nums))
            df = pd.concat(dfs)
            del df['index']
        else:
            df = DataAugmentation.replace_base_op(df, format_type, percent, null_first, cols, replace_type, nums)
        return df
