# -*- coding: utf-8 -*-
"""
@File    : metrics.py
@Desc    : compute_info_gain compute_NSS compute_cAUC get_seq_score get_semantic_seq_score
"""
import scipy.ndimage as filters
import numpy as np
import torch
import gzip
from os.path import join
# 如果 multimatch.py 在 common 目录下
from common.multimatch import docomparison

def multimatch(s1, s2, im_size):
    s1x = s1['X']
    s1y = s1['Y']
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
    s2x = s2['X']
    s2y = s2['Y']
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]


def compute_mm(human_trajs, model_trajs, im_w, im_h, tasks=None):
    """
    compute scanpath similarity using multimatch
    """
    all_mm_scores = []
    for traj in model_trajs:
        img_name = traj['name']
        task = traj['task']
        gt_trajs = list(
            filter(lambda x: x['name'] == img_name and x['task'] == task,
                   human_trajs))
        all_mm_scores.append((task,
                              np.mean([
                                  multimatch(traj, gt_traj, (im_w, im_h))[:4]
                                  for gt_traj in gt_trajs
                              ],
                                      axis=0)))

    if tasks is not None:
        mm_tasks = {}
        for task in tasks:
            mm = np.array([x[1] for x in all_mm_scores if x[0] == task])
            mm_tasks[task] = np.mean(mm, axis=0)
        return mm_tasks
    else:
        return np.mean([x[1] for x in all_mm_scores], axis=0)

        
def compute_info_gain(predicted_probs, gt_fixs, base_probs, eps=2.2204e-16):
    """
    一个计算「信息增益（Information Gain, IG）」 的函数，它通常用于评估注视点预测模型到底比「基线模型」好多少。
    计算信息增益（Information Gain, IG）
    想象你在看一张图片，一只猫站在画面中间。我们有两个模型：
    预测模型（predicted_probs）：它根据图像内容判断你最有可能看猫（中心附近）。
    基线模型（base_probs）：它不看图，只知道人类一般看图中心（center bias）。
    参数：
    - predicted_probs: Tensor，形状为 (batch_size, H, W)
        模型预测的注视概率分布，按空间位置给出概率
    - gt_fixs: Tensor，形状为 (batch_size, 2)
        真实注视点坐标 (x, y)，这里假设第0维是batch索引，第1维是坐标
    - base_probs: Tensor，形状同 predicted_probs
        基线概率分布，比如均匀分布或者中心偏置模型的概率
    - eps: 浮点数，小常数，避免对数计算时出现log(0)

    返回：
    - IG: 标量 Tensor，信息增益总和
    """
    # 取真实注视点对应的预测概率值：从 predicted_probs 中提取每个样本真实注视点处的概率 一次性从一批图像的预测热图中，取出每张图「你实际注视点」处的概率值。
    fired_probs = predicted_probs[torch.arange(gt_fixs.size(0)), gt_fixs[:, 1], gt_fixs[:, 0]]

    # 取真实注视点对应的基线概率值
    fired_base_probs = base_probs[torch.arange(gt_fixs.size(0)), gt_fixs[:, 1], gt_fixs[:, 0]]

    # 计算每个真实注视点上的log2概率差，累加得到总信息增益
    IG = torch.sum(torch.log2(fired_probs + eps) - torch.log2(fired_base_probs + eps))

    return IG


def compute_NSS(saliency_map, gt_fixs):

    # NSS衡量的是：模型在真实注视点位置上的预测概率，比整张图的平均水平高出多少个标准差。
    # 想象你眼前有一张热力图，某些区域发亮，表示模型觉得那“很显著”。而你实际上看向了某一个点。

    # NSS 就是在问：“你看的那个点，在这张热图上，是亮的，还是平淡无奇的？”
    # 真正注视的地方，是否落在了模型预测显著区域（热图的“亮点”）上，落得越准，NSS越高。

    # saliency_map: Tensor，形状为 (batch_size, H, W)，模型预测的显著图概率值
    # gt_fixs: Tensor，形状为 (batch_size, 2)，真实注视点坐标 (x, y)

    # 计算每个样本显著图的均值（展平后按行计算）
    mean = saliency_map.view(gt_fixs.size(0), -1).mean(dim=1)

    # 计算每个样本显著图的标准差（展平后按行计算）
    std = saliency_map.view(gt_fixs.size(0), -1).std(dim=1)

    # 防止标准差为0，避免后续除法出错
    std[std == 0] = 1

    # 取出每个样本真实注视点对应的显著图数值（概率值）
    value = saliency_map[torch.arange(gt_fixs.size(0)), gt_fixs[:, 1], gt_fixs[:, 0]]

    # 对取出的值做归一化处理：减去均值，除以标准差
    value -= mean
    value /= std

    # 返回所有样本归一化值的和（总 NSS 值）
    return value.sum()

def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0

        
def nw_matching(pred_string, gt_string, gap=0.0):
    """
    nw_matching() 是一个字符串（或注视路径）相似度打分函数，考虑了匹配、插入、删除三种操作，并用动态规划找出最优对齐路径，最终输出归一化相似度得分（0~1）。
    """
    # 初始化动态规划矩阵F，大小为(len(pred_string)+1, len(gt_string)+1)
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)

    # 初始化第一列，表示对pred_string序列的i个元素全删除（或插入gap惩罚）
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i

    # 初始化第一行，表示对gt_string序列的j个元素全删除（或插入gap惩罚）
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j

    # 动态规划填表
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]  # pred_string当前元素
            b = gt_string[j - 1]    # gt_string当前元素

            # 计算匹配得分：对角线元素 + 当前两个元素的相似度
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)

            # 删除操作（pred_string的元素被删除）
            delete = F[i - 1, j] + gap

            # 插入操作（gt_string的元素被插入）
            insert = F[i, j - 1] + gap

            # 取三者中的最大值，填入F[i,j]
            F[i, j] = np.max([match, delete, insert])

    # 归一化得分：用最后一个格子的值除以较长序列长度
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))

"""
scanpath2clusters 将一个注视路径（scanpath）中的所有注视点（x, y 坐标）传入一个 MeanShift 聚类模型 中，
得到每个注视点所属的聚类标签，然后将这些标签组成一个序列（字符串形式），用于后续的行为分析或匹配比对（如NW匹配）。
"""
def scanpath2clusters(meanshift, scanpath):
    """

    :param meanshift: 一个已经训练好的 MeanShift 聚类模型
    :param scanpath:一个字典类型，包含 X、Y 坐标序列（眼动注视点轨迹）
    :return: scanpath的聚类标签序列
    """
    string = []  # 用于存放每个注视点对应的聚类标签（类别符号）
    xs = scanpath['X']  # 眼动轨迹的X坐标序列
    ys = scanpath['Y']  # 眼动轨迹的Y坐标序列

    # 遍历每个注视点的坐标
    for i in range(len(xs)):
        # 用meanshift聚类模型预测当前注视点所属的聚类标签
        symbol = meanshift.predict([[xs[i], ys[i]]])[0]
        string.append(symbol)  # 把标签添加进结果列表

    return string  # 返回该scanpath的聚类标签序列

def compute_SS(preds, clusters, truncate, truncate_gt, reduce='mean'):
    """
    用来计算预测的眼动扫描路径（scanpath）和真实类别序列（ground truth clusters）之间的相似度评分，
    核心思想是把眼动轨迹转成类别序列，再用序列比对算法计算匹配度，最后返回每条扫描路径的相似度结果。
    想象你有一堆眼动路径数据（每个人看东西时眼睛的跳动轨迹），你先把每条路径上的点分到不同的区域类别（用聚类算法分类）。然后你有对应的真实“参考答案”——正确的类别序列。
    你想知道你的预测轨迹和真实参考有多像，就用序列比对算法（nw_matching）比较它们的相似度。最后你把这些相似度统计起来，比如求平均，来评价整体预测效果。
    解决一个预测路径和多个参考答案之间的比较问题。
    下面给你形象详细讲解每步的作用：
    输入：
    preds：预测的扫描路径列表，每条扫描路径是个字典，里面有条件、任务名、路径名字等信息。
    clusters：聚类结果，保存了每条路径对应的真实类别字符串和预测的类别序列。
    truncate：截断长度，最长对比多少步。
    truncate_gt：是否也对真实类别序列截断。
    reduce：对多个相似度得分如何汇总（平均或最大）。
    输出：
    {
    'condition': 'freeview' 或 'TP',
    'task': 任务名（如果有任务）,
    'name': '图像文件名',
    'score': 与多个 ground truth 路径匹配的平均相似度得分
    }
    """
    results = []
    # 遍历每条预测扫描路径
    for scanpath in preds:
        # 判断是否为自由浏览（freeview）条件
        is_fv = scanpath['condition'] == 'freeview'

        # 构造clusters的key，区分freeview和task条件
        if is_fv:
            key = 'test-{}-{}'.format(scanpath['condition'], scanpath['name'].split('.')[0])
        else:
            key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                         scanpath['name'].split('.')[0])

        # 获取对应key的clusters信息
        ms = clusters[key]
        strings = ms['strings']  # 多个 ground truth 类别序列 —— 即多个“正确答案”。
        cluster = ms['cluster']  # 预测路径对应的类别序列

        # 将预测路径映射成类别序列
        pred = scanpath2clusters(cluster, scanpath)

        scores = []
        # 若无gt字符串，跳过
        if len(strings) == 0:
            continue

        # 遍历所有gt类别字符串，计算相似度 让这一条预测路径，分别与多个正确答案对比，得出多个分数，然后取平均作为最终得分。
        for gt in strings:
            if len(gt) > 0:
                # 根据truncate参数截断预测序列
                pred = pred[:truncate] if len(pred) > truncate else pred
                # 根据truncate_gt参数截断gt序列
                if truncate_gt:
                    gt = gt[:truncate] if len(gt) > truncate else gt

                # 计算nw_matching（Needleman-Wunsch序列匹配）得分
                score = nw_matching(pred, gt)
                scores.append(score)

        # 构建单条scanpath的结果字典
        result = {}
        result['condition'] = scanpath['condition']
        if not is_fv:
            result['task'] = scanpath['task']
        result['name'] = scanpath['name']

        # 对scores列表做降维处理，默认取平均，也可取最大
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError

        results.append(result)

    return results


def compute_SSS(preds,
                fixations,
                truncate,
                segmentation_map_dir,
                truncate_gt,
                reduce='mean'):
    """
    计算预测的眼动扫描路径（scanpath）与真实语义类别序列之间的相似度得分。

    ✅ 与 compute_SS 的核心区别：
        - compute_SS 使用的是聚类后的“类别ID序列”；
        - compute_SSS 使用的是“分割图语义标签”，即直接从真实图片的 segmentation map 中提取的类别标签序列。
    两者都是在做一件事：
    比较预测的眼动路径（scanpath）和“某种形式”的真实参考路径（ground truth），看它们有多像。
    但关键在于：
    参考路径（ground truth）到底是什么？怎么得来的？
    ✅ compute_SS：聚类区域比对
🧠 像是在考你“眼睛看过哪些区域”
    ground truth 是聚类后的“类别 ID 序列”：
    比如把图像分成 10 个区域（聚类），真实的注视路径是 [3, 7, 7, 2]，表示注视依次落在这些区域上。
    预测路径也被映射成这些区域编号序列，比如 [3, 7, 2, 2]。
    然后比较两个“区域编号序列”有多像（用序列匹配算法）。
🧭 举例类比：
就像你让学生在一张图上自由浏览，你只关心他们看了哪些“区域”（不在乎这些区域是人脸还是杯子，只看编号）。

✅ compute_SSS：语义分割比对
🧠 像是在考你“眼睛注视的是哪些语义对象”
    ground truth 是图像分割语义标签，比如：
        图像上每个像素有语义类别标签（人脸 = 1，杯子 = 2，背景 = 0...）
        真实注视点是 [1, 1, 2, 3]，表示依次注视“人脸、人脸、杯子、桌子”。
    预测路径也通过分割图得到语义类别序列，比如 [1, 2, 2, 0]。
    然后同样比较预测语义序列和真实语义序列的匹配度。
🧭 举例类比：
就像你让学生看图，你不仅关心他们看了哪里，还关心他们看的是不是“关键语义物体”（如目标、人脸等）——不是只看区域编号，而是看实际物体/意义。

    参数：
    - preds：预测的扫描路径列表（包含坐标信息）。
    - fixations：真实注视点对应的语义标签序列（每个路径对应多个 ground truth 序列）。
    - truncate：截断长度，限制比较的最大序列长度。
    - segmentation_map_dir：分割图的路径，每张图都是 `.npy.gz` 格式。
    - truncate_gt：是否截断 ground truth 序列。
    - reduce：对多个得分的汇总方式（'mean' 或 'max'）。
    """
    results = []

    # ⬇️ 内部函数：将预测的注视路径映射为语义类别标签序列（如 ['3', '7', '7', '10']）
    def scanpath2categories(seg_map, scanpath):
        string = []  # 用来存储每个注视点对应的类别标签（字符串形式）
        xs = scanpath['X']  # X 坐标序列
        ys = scanpath['Y']  # Y 坐标序列

        # 遍历所有注视点 把预测的眼动轨迹中的每个注视点，映射为它落在图像上对应的语义类别标签（通过分割图），从而生成一个“语义类别序列”。
        # zip把两个序列“打包”成一对对坐标 (x, y)，用于一起遍历，比如眼动轨迹中的每一个注视点的位置。
        for x, y in zip(xs, ys):
            # 获取当前位置在分割图上的语义类别（转为整数再转为字符串）
            symbol = str(int(seg_map[int(y), int(x)]))
            string.append(symbol)  # 加入当前点的语义类别标签

        return string  # 返回整个scanpath的类别标签序列

    # ⬇️ 遍历每个预测scanpath
    for scanpath in preds:
        is_fv = scanpath['condition'] == 'freeview'

        # 构造唯一 key 来从 fixations 中找到该图的 ground truth 注视标签序列
        if is_fv:
            key = 'test-{}-{}'.format(scanpath['condition'], scanpath['name'].split('.')[0])
        else:
            key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                         scanpath['name'].split('.')[0])

        # 🔸 获取对应 key 的 ground truth 类别字符串列表
        strings = fixations[key]  # 每个元素是一个字符串列表（多个真实注视路径的语义标签序列）

        # 🔸 从压缩文件中载入当前图像的语义分割图（npy.gz 格式）
        with gzip.GzipFile(
                join(segmentation_map_dir, scanpath['name'][:-3] + 'npy.gz'),
                "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()

        # 🔸 将预测路径坐标转换为语义类别标签序列
        pred = scanpath2categories(segmentation_map, scanpath)

        scores = []  # 保存该预测路径与每个 ground truth 匹配的分数

        # ⬇️ 遍历所有 ground truth 序列，与预测进行匹配比对
        for gt in strings:
            if len(gt) > 0:
                # 截断预测序列（如果太长）
                pred = pred[:truncate] if len(pred) > truncate else pred
                # 截断 ground truth 序列（如果启用 truncate_gt）
                if truncate_gt:
                    gt = gt[:truncate] if len(gt) > truncate else gt

                # 🧮 用 Needleman-Wunsch 算法计算两个序列的匹配得分
                score = nw_matching(pred, gt)
                scores.append(score)

        # ⬇️ 构建当前路径的结果字典
        result = {}
        result['condition'] = scanpath['condition']
        if not is_fv:
            result['task'] = scanpath['task']
        result['name'] = scanpath['name']

        # ⬇️ 汇总多个 ground truth 得分（平均或最大）
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError

        results.append(result)

    return results  # 返回所有预测路径的匹配得分结果

# 1. 编辑距离基础函数
def _Levenshtein_Dmatrix_initializer(len1, len2):
    Dmatrix = []
    for i in range(len1):
        Dmatrix.append([0] * len2)
    for i in range(len1):
        Dmatrix[i][0] = i
    for j in range(len2):
        Dmatrix[0][j] = j
    return Dmatrix

def _Levenshtein_cost_step(Dmatrix, string_1, string_2, i, j, substitution_cost=1):
    char_1 = string_1[i - 1]
    char_2 = string_2[j - 1]
    insertion = Dmatrix[i - 1][j] + 1
    deletion = Dmatrix[i][j - 1] + 1
    substitution = Dmatrix[i - 1][j - 1] + substitution_cost * (char_1 != char_2)
    Dmatrix[i][j] = min(insertion, deletion, substitution)

def _Levenshtein(string_1, string_2, substitution_cost=1):
    len1 = len(string_1)
    len2 = len(string_2)
    Dmatrix = _Levenshtein_Dmatrix_initializer(len1 + 1, len2 + 1)
    
    for i in range(len1):
        for j in range(len2):
            _Levenshtein_cost_step(Dmatrix, string_1, string_2, 
                                   i + 1, j + 1, substitution_cost=substitution_cost)
    
    if substitution_cost == 1:
        max_dist = max(len1, len2)
    elif substitution_cost == 2:
        max_dist = len1 + len2
    
    return Dmatrix[len1][len2]

# 2. ED（编辑距离）计算函数
def compute_ED(preds, clusters, truncate, truncate_gt=False, reduce='mean'):
    results = []
    for scanpath in preds:
        is_fv = scanpath['condition'] == 'freeview'
        if is_fv:
            key = 'test-{}-{}'.format(scanpath['condition'], scanpath['name'].split('.')[0])
        else:
            key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                         scanpath['name'].split('.')[0])
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']
        
        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        if len(strings) == 0:
            continue
        for gt in strings:
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                if truncate_gt:
                    gt = gt[:truncate] if len(gt) > truncate else gt
                score = _Levenshtein(pred, gt)
                scores.append(score)
        
        result = {}
        result['condition'] = scanpath['condition']
        if not is_fv:
            result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results

def get_ed(preds, clusters, max_step, truncate_gt=False, tasks=None):
    results = compute_ED(preds, clusters, max_step, truncate_gt)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))

# 3. SED（语义编辑距离）计算函数
def compute_SED(preds, fixations, truncate, segmentation_map_dir, truncate_gt=False, reduce='mean'):
    results = []
    # ⬇️ 内部函数：将预测的注视路径映射为语义类别标签序列（如 ['3', '7', '7', '10']）
    def scanpath2categories(seg_map, scanpath):
        string = []  # 用来存储每个注视点对应的类别标签（字符串形式）
        xs = scanpath['X']  # X 坐标序列
        ys = scanpath['Y']  # Y 坐标序列

        # 遍历所有注视点 把预测的眼动轨迹中的每个注视点，映射为它落在图像上对应的语义类别标签（通过分割图），从而生成一个“语义类别序列”。
        # zip把两个序列“打包”成一对对坐标 (x, y)，用于一起遍历，比如眼动轨迹中的每一个注视点的位置。
        for x, y in zip(xs, ys):
            # 获取当前位置在分割图上的语义类别（转为整数再转为字符串）
            symbol = str(int(seg_map[int(y), int(x)]))
            string.append(symbol)  # 加入当前点的语义类别标签

        return string  # 返回整个scanpath的类别标签序列
    for scanpath in preds:
        is_fv = scanpath['condition'] == 'freeview'
        if is_fv:
            key = 'test-{}-{}'.format(scanpath['condition'], scanpath['name'].split('.')[0])
        else:
            key = 'test-{}-{}-{}'.format(scanpath['condition'], scanpath['task'],
                                         scanpath['name'].split('.')[0])
        strings = fixations[key]
        
        with gzip.GzipFile(
                join(segmentation_map_dir, scanpath['name'][:-3] + 'npy.gz'), "r") as r:
            segmentation_map = np.load(r, allow_pickle=True)
            r.close()
        
        pred = scanpath2categories(segmentation_map, scanpath)
        scores = []
        for gt in strings:
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                if truncate_gt:
                    gt = gt[:truncate] if len(gt) > truncate else gt
                score = _Levenshtein(pred, gt)
                scores.append(score)
        
        result = {}
        result['condition'] = scanpath['condition']
        if not is_fv:
            result['task'] = scanpath['task']
        result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results

def get_semantic_ed(preds, fixations, max_step, segmentation_map_dir, truncate_gt=False, tasks=None):
    results = compute_SED(preds, fixations, max_step, segmentation_map_dir, truncate_gt)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))


def compute_cAUC(s_map, gt_next_fixs):
    """
    计算基于注视点的AUC-Judd指标，衡量显著图中真实注视点的显著性值在整体显著值中的百分位。
    模型预测的显著区域是否覆盖了真正被人注视的点。它通过 ROC 曲线上的积分（AUC）来表示显著图的“可信度”。
    Args:
       s_map: Tensor，形状为 [B, H, W]，模型预测的显著图
       gt_next_fixs: Tensor，形状为 [B, 2]，真实注视点的(x, y)坐标  2 表示 每个注视点的两个坐标值，即 (x, y)。
    """
    # 对每张图，取出真实注视点对应的显著值作为阈值
    thresholds = s_map[torch.arange(len(gt_next_fixs)),
    gt_next_fixs[:, 1],
    gt_next_fixs[:, 0]]

    bs = len(gt_next_fixs)  # batch size

    area = []
    area.append(torch.zeros(bs, 2))  # AUC曲线起点坐标 (0,0)

    # 只保留显著值大于等于阈值的像素点，构造二值掩码图
    temp = torch.zeros_like(s_map)
    temp[s_map >= thresholds.view(bs, 1, 1)] = 1.0
    temp = temp.view(bs, -1)  # 展平为(batch_size, H*W)

    # 计算True Positive (TP)与False Positive (FP)：
    # 每张图只有一个正样本，TP为1
    tp = torch.ones(bs)
    # FP为剩余被判为正的点数量除以总负样本数
    fp = (temp.sum(-1) - 1) / (temp.size(-1) - 1)

    # 添加当前点 (TP, FP) 坐标到AUC曲线点列表
    area.append(torch.stack([tp, fp.cpu()], dim=1))
    # AUC曲线终点 (1,1)
    area.append(torch.ones(bs, 2))
    # 将起点、阈值点、终点堆叠成一个三维张量，形状为 (batch_size, 3, 2)
    area = torch.stack(area, dim=1)

    # 利用torch.trapz计算AUC面积（梯形积分）
    # 对TP坐标(area[:,:,0])关于FP坐标(area[:,:,1])积分，求和返回总AUC
    return torch.trapz(area[:, :, 0], area[:, :, 1]).sum()

def get_seq_score(preds, clusters, max_step, truncate_gt=False, tasks=None):
    """
    假设你是老师（模型），你让一群学生（preds，模型预测的注视路径）去看一些图片并说出他们看的顺序（scanpath）。
    每个学生给出了自己的“注视顺序”，你手上有每张图的标准答案（真实注视路径的类别序列 = clusters）。
    你的任务是给这些学生打分：看他们说的注视路径和真实路径有多像（用匹配算法 compute_SS 比较）。
    最终你要统计平均得分，看这些预测总体表现好不好。
    """
    # 计算每个预测序列与真实类别的相似度结果（一个列表，每项包含score和task等信息）
    results = compute_SS(preds, clusters, max_step, truncate_gt)

    if tasks is None:
        # 若未指定任务列表，直接返回所有结果的score平均值
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        # 按每个任务筛选结果，计算该任务对应的score均值
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        # 返回一个字典，键为任务名，值为该任务的平均score
        return dict(zip(tasks, scores))


def get_semantic_seq_score(preds,
                           fixations,
                           max_step,
                           segmentation_map_dir,
                           truncate_gt=False,
                           tasks=None):
    # 调用 compute_SSS，传入预测序列、真实注视点、最大步数、语义分割图路径以及是否截断真实序列
    results = compute_SSS(preds, fixations, max_step, segmentation_map_dir,
                          truncate_gt)

    if tasks is None:
        # 若不区分任务，返回所有结果中score的平均值
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        # 如果指定了任务，则分别计算各任务对应的score均值
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        # 返回任务名与平均score组成的字典
        return dict(zip(tasks, scores))
