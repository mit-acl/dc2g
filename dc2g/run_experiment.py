#!/usr/bin/env python
import random
import run_episode

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
import datetime
import glob
import os
import pickle

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 22

np.set_printoptions(suppress=True, precision=4)
# np.warnings.filterwarnings('ignore')
# np.set_printoptions(threshold=np.inf)

# sns.set(style="darkgrid")
planner_color_dict = {"oracle": "#3498db", "dc2g": "#2ecc71", "dc2g_rescale": "#f1c40f", "frontier": "#c0392b"}
# planner_color_dict = {"oracle": "#3498db", "dc2g": "#2ecc71", "dc2g_rescale": "#16a085", "frontier": "#c0392b"}
# planner_color_list = ["#f1c40f", "#2ecc71", "#16a085", "#9b59b6", "#c0392b"]
# planner_color_list = ["#f1c40f", "#2ecc71", "#16a085", "#c0392b"]
# sns.set_palette(planner_color_list)
save_dir = '/home/mfe/code/dc2g/gym-minigrid/results/pandas'

def run_experiment():
    experiement_seed = 1337
    random.seed(experiement_seed)

    env_name = 'MiniGrid-EmptySLAM-32x32-v0'
    env_type = 'MiniGrid'

    numTrials = 100

    results = {}

    # planners = ['dc2g', 'dc2g_rescale', 'frontier', 'oracle']
    planners = ['dc2g', 'frontier', 'oracle']
    # planners = ['dc2g', 'oracle']
    # planners = ['oracle']
    # planners = ['frontier']
    num_planners = len(planners)

    env = run_episode.start_experiment(env_name=env_name, env_type=env_type)
    print('--------')
    for world_difficulty in ['easy', 'medium', 'hard', 'very_hard']:
    # for world_difficulty in ['easy', 'medium']:
    # for world_difficulty in ['hard', 'very_hard']:
        results[world_difficulty] = {}
        for trial_num in range(0, numTrials):
            print("world_difficulty: {}".format(world_difficulty))
            print("trial_num: {}".format(trial_num))
            results[world_difficulty][trial_num] = {}
            seed = random.randint(0, 1e5)
            for planner in planners:
                print("planner: {}".format(planner))
                results[world_difficulty][trial_num][planner] = {}
                success, num_steps, world_id = run_episode.run_episode(planner, seed, env, env_type, world_difficulty)
                print("world_id: {}".format(world_id))
                results[world_difficulty][trial_num][planner]['num_steps'] = num_steps
                results[world_difficulty][trial_num][planner]['success'] = success
                results[world_difficulty][trial_num]['world_id'] = world_id
            print('--------')

    with open('{}/{:%Y-%m-%d_%H_%M_%S}.pkl'.format(save_dir, datetime.datetime.now()), 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("results = {}".format(results))
    print('\n\n\n-----------------------\nDone w/ sim.\n\n')
    return results

# def run_experiment():
#     experiement_seed = 1337
#     random.seed(experiement_seed)

#     env_name = 'MiniGrid-EmptySLAM-32x32-v0'

#     numTrials = 50

#     columns = ['experiment number', 'Planner', 'seed', 'Time to Goal (# steps)', 'success', 'world', 'Sensor Horizon']
#     results = pd.DataFrame(columns=columns)


#     planners = ['oracle', 'dc2g', 'dc2g_rescale', 'frontier']
#     num_planners = len(planners)

#     env = run_episode.start_experiment(env_name=env_name)
#     for trial_num in range(0, numTrials):
#         print("trial_num: {}".format(trial_num))
#         # results[trial_num] = {}
#         seed = random.randint(0, 1e5)
#         for planner in planners:
#             print("planner: {}".format(planner))
#             # results[trial_num][planner] = {}
#             success, num_steps = run_episode.run_episode(planner, seed, env)
#             # results[trial_num][planner]['num_steps'] = num_steps
#             # results[trial_num][planner]['success'] = success
#             df = pd.DataFrame([[trial_num, planner, seed, num_steps, success, 30, 3]], columns=columns)
#             results = results.append(df, ignore_index=True)

#     filename = '{}/{:%Y-%m-%d_%H_%M_%S}.pkl'.format(save_dir, datetime.datetime.now())
#     results.to_pickle(filename)
#     print("results = {}".format(results))
#     # print('\n\n\n-----------------------\nDone w/ sim.')
#     return results


def plot(results=None):
    if results is None:
        # list_of_files = glob.glob('{}/*.pkl'.format(save_dir))
        # latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = "{}/2018-09-13_15_51_57.pkl".format(save_dir)
        with open(latest_file, 'rb') as handle:
            results = pickle.load(handle)

    planners = ['oracle', 'dc2g', 'frontier']
    # planners = ['oracle', 'dc2g', 'frontier']
    # planners = ['oracle', 'dc2g', 'dc2g_rescale', 'frontier']
    # planners = ['oracle', 'dc2g', 'dc2g_rescale']
    # planners = results[0].keys()
    num_planners = len(planners)
    # num_bar_groups = len(results.keys())
    num_bar_groups = 4
    num_trials = len(results[list(results.keys())[0]].keys())

    # colors = ['b', 'g', 'y', 'r']
    width = 0.1       # the width of the bars
    padding_btwn_bar_groups = 0.1
    num_bars = num_planners - 1
    bar_group_width = num_bars*width
    ind = bar_group_width/2.0 + np.arange(num_bar_groups) * (bar_group_width + padding_btwn_bar_groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('Extra Time to Goal (%)')
    ax.set_xticks(ind + width/2.0)
    # ax.set_xticklabels(('Very Similar\n(Houses w/ Typ. Layout)', 'Somewhat Similar\n(Houses w/ Weird Layouts)', 'Very Different\n(Not even houses)'))
    ax.set_xticklabels(('Training\nHouses', 'Same\nNeighborhood', 'Different\nNeighborhood', 'Urban\nArea'))
    # ax.set_xticklabels(('Training', 'Same Neighborhood'))
    ax.set_xlabel('Environment Type')


    extra_steps_vs_oracle = []

    num_steps_arr = np.empty((num_trials, num_planners))
    extra_num_steps_arr = np.empty((num_trials, num_planners))
    pct_incr_num_steps_arr = np.empty((num_trials, num_planners))
    success_arr = np.empty((num_trials, num_planners))
    for group_i, world_difficulty in enumerate(['easy', 'medium', 'hard', 'very_hard']):
    # for group_i, world_difficulty in enumerate(['easy']):
        # for trial_num in range(num_trials):
        #     num_steps_arr[trial_num, num_planners - 1] = results[world_difficulty][trial_num]["frontier"]['num_steps']
        #     success_arr[trial_num, num_planners - 1] = results[world_difficulty][trial_num]["frontier"]['success']
        for i, planner in enumerate(planners):
            if planner == "oracle": continue
            for trial_num in range(num_trials):
                num_steps_arr[trial_num, i] = results[world_difficulty][trial_num][planner]['num_steps']
                extra_num_steps_arr[trial_num, i] = results[world_difficulty][trial_num][planner]['num_steps'] - results[world_difficulty][trial_num]['oracle']['num_steps']
                pct_incr_num_steps_arr[trial_num, i] = 100.*(results[world_difficulty][trial_num][planner]['num_steps'] - results[world_difficulty][trial_num]['oracle']['num_steps']) / float(results[world_difficulty][trial_num]['oracle']['num_steps'])
                success_arr[trial_num, i] = results[world_difficulty][trial_num][planner]['success']
            # print(extra_num_steps_arr)
            # planner_mean = np.mean(extra_num_steps_arr[:, i])
            # planner_std = np.std(extra_num_steps_arr[:, i])
            # planner_mean = np.mean(num_steps_arr[:, i])
            # planner_std = np.std(num_steps_arr[:, i])
            # planner_mean = np.mean(pct_incr_num_steps_arr[:, i])
            # planner_std = np.std(pct_incr_num_steps_arr[:, i])

            if planner == "frontier":
                l = pct_incr_num_steps_arr[:,i].tolist()
                extra_steps_vs_oracle += l

            data = pct_incr_num_steps_arr[:,i]
            median = np.median(data)

            upper_quartile = np.percentile(data, 75)
            lower_quartile = np.percentile(data, 25)
            iqr = upper_quartile - lower_quartile
            upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
            lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
            upper_error = abs(median-upper_whisker)
            lower_error = abs(median-lower_whisker)
            print(median)
            print(upper_whisker)
            print(lower_whisker)

            label = None
            if group_i == 0:
                if planner == 'dc2g': label = 'DC2G'
                elif planner == 'dc2g_rescale': label = 'DC2G-R'
                elif planner == 'frontier': label = 'Frontier'
                elif planner == 'oracle': label = 'Oracle'
                # label = planner
            rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, median, width, color=planner_color_dict[planner], label=label, yerr=[[lower_error], [upper_error]], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, planner_mean, width, color=planner_color_dict[planner], label=label, yerr=planner_std, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # print(world_difficulty, planner, planner_mean, planner_std)
        # l = [num_steps_arr[:,i] for i in range(num_planners)]
        # l = [None]*group_i*num_planners + l
        # ax.boxplot(num_steps_arr)

    print("len:", len(extra_steps_vs_oracle))
    print("extra:", np.mean(extra_steps_vs_oracle))

    ax.legend()
    plt.tight_layout()
    # plt.savefig('{}/{}__{:%Y-%m-%d_%H_%M_%S}.pdf'.format(save_dir, latest_file.split('/')[-1][:-4], datetime.datetime.now()))
    plt.show()

def plot2(results=None):
    list_of_files = glob.glob('{}/*.pkl'.format(save_dir))
    latest_file = max(list_of_files, key=os.path.getctime)
    results = pd.read_pickle(latest_file)
    # columns = ['experiment number', 'planner', 'seed', 'num_steps', 'success', 'world']
    # df = pd.DataFrame([[i, 'RL', 0, 50+100*i, True, 30] for i in range(3)], columns=columns)
    # results = results.append(df, ignore_index=True)
    # print(results)
    ax = sns.barplot(x="Planner", y="Time to Goal (# steps)", data=results, capsize=0.2, order=['oracle', 'dc2g', 'dc2g_rescale', 'frontier'])
    # ax.set(xticklabels=[])
    # ax = sns.barplot(x="planner", y="num_steps", hue="planner", data=results, capsize=0.02, order=['oracle', 'dc2g_rescale', 'dc2g', 'RL', 'frontier'])
    plt.savefig('{}/{}__{:%Y-%m-%d_%H_%M_%S}.pdf'.format(save_dir, latest_file.split('/')[-1][:-4], datetime.datetime.now()))
    plt.show()

def plot3(results=None):
    list_of_files = glob.glob('{}/*.pkl'.format(save_dir))
    latest_file = max(list_of_files, key=os.path.getctime)
    results = pd.read_pickle(latest_file)

    # # columns = ['experiment number', 'planner', 'seed', 'num_steps', 'success', 'world']
    # # df = pd.DataFrame([[i, 'RL', 0, 50+100*i, True, 30] for i in range(3)], columns=columns)
    # # results = results.append(df, ignore_index=True)
    # # print(results)
    # ax = sns.barplot(x="Planner", y="Time to Goal (# steps)", data=results, capsize=0.2, order=['oracle', 'dc2g', 'dc2g_rescale', 'frontier'])
    # # ax.set(xticklabels=[])
    # # ax = sns.barplot(x="planner", y="num_steps", hue="planner", data=results, capsize=0.02, order=['oracle', 'dc2g_rescale', 'dc2g', 'RL', 'frontier'])
    # plt.savefig('{}/{}__{:%Y-%m-%d_%H_%M_%S}.pdf'.format(save_dir, latest_file.split('/')[-1][:-4], datetime.datetime.now()))
    # plt.show()

def plot_per_world(results=None):
    if results is None:
        # list_of_files = glob.glob('{}/*.pkl'.format(save_dir))
        # latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = "{}/2018-09-13_15_51_57.pkl".format(save_dir)
        with open(latest_file, 'rb') as handle:
            results = pickle.load(handle)

    planners = ['oracle', 'dc2g', 'frontier']
    
    results_by_world = {}
    world_ids = []
    for world_difficulty in results:
        for trial in results[world_difficulty]:
            world_id = results[world_difficulty][trial]['world_id']
            if world_id not in results_by_world:
                results_by_world[world_id] = {}
            results_by_world[world_id][trial] = {}
            for planner in planners:
                # world_id = results[world_difficulty][trial]['dc2g']['world_id']
                results_by_world[world_id][trial][planner] = results[world_difficulty][trial][planner]

    # planners = ['oracle', 'dc2g', 'dc2g_rescale', 'frontier']
    # planners = ['oracle', 'dc2g', 'dc2g_rescale']
    # planners = results[0].keys()
    num_planners = len(planners)
    # num_bar_groups = len(results.keys())
    num_bar_groups = len(results_by_world.keys())
    # num_trials = len(results[list(results.keys())[0]].keys())

    # colors = ['b', 'g', 'y', 'r']
    width = 0.1       # the width of the bars
    padding_btwn_bar_groups = 0.1
    num_bars = num_planners - 1
    bar_group_width = num_bars*width
    ind = bar_group_width/2.0 + np.arange(num_bar_groups) * (bar_group_width + padding_btwn_bar_groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('Extra Time to Goal (%)')
    # ax.set_ylabel('Extra Time to Goal (# steps)')
    ax.set_xticks(ind + width/2.0)
    # ax.set_xticklabels(('Very Similar\n(Houses w/ Typ. Layout)', 'Somewhat Similar\n(Houses w/ Weird Layouts)', 'Very Different\n(Not even houses)'))
    # ax.set_xticklabels(('Training', 'Same Neighborhood', 'Different Neighborhood', 'Urban'))
    # ax.set_xticklabels(('Training', 'Same Neighborhood'))
    ax.set_xlabel('World ID')

    # num_steps_arr = np.empty((num_trials, num_planners))
    # extra_num_steps_arr = np.empty((num_trials, num_planners))
    # pct_incr_num_steps_arr = np.empty((num_trials, num_planners))
    # success_arr = np.empty((num_trials, num_planners))
    xticklabels = []

    world_ids = sorted(results_by_world.keys())

    for group_i, world_id in enumerate(world_ids):
        trial_nums = results_by_world[world_id]
        num_trials = len(trial_nums)
        num_steps_arr = np.empty((num_trials, num_planners))
        extra_num_steps_arr = np.empty((num_trials, num_planners))
        pct_incr_num_steps_arr = np.empty((num_trials, num_planners))
        success_arr = np.empty((num_trials, num_planners))
        planner = "oracle"
        for trial_i, trial_num in enumerate(trial_nums):
            num_steps_arr[trial_i, num_planners - 1] = results_by_world[world_id][trial_num][planner]['num_steps']
            success_arr[trial_i, num_planners - 1] = results_by_world[world_id][trial_num][planner]['success']
        for i, planner in enumerate(planners):
            # if planner == "oracle": continue
            for trial_i, trial_num in enumerate(trial_nums):
                num_steps_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['num_steps']
                extra_num_steps_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['num_steps'] - results_by_world[world_id][trial_num]['oracle']['num_steps']
                pct_incr_num_steps_arr[trial_i, i] = (results_by_world[world_id][trial_num][planner]['num_steps'] - results_by_world[world_id][trial_num]['oracle']['num_steps']) / float(results_by_world[world_id][trial_num]['oracle']['num_steps'])
                success_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['success']
            # print(extra_num_steps_arr)
            # planner_mean = np.mean(extra_num_steps_arr[:, i])
            # planner_std = np.std(extra_num_steps_arr[:, i])
            # planner_mean = np.mean(num_steps_arr[:, i])
            # planner_std = np.std(num_steps_arr[:, i])
            # planner_mean = np.mean(pct_incr_num_steps_arr[:, i])
            # planner_std = np.std(pct_incr_num_steps_arr[:, i])

            data = extra_num_steps_arr[:,i]
            median = np.median(data)
            upper_quartile = np.percentile(data, 75)
            lower_quartile = np.percentile(data, 25)
            iqr = upper_quartile - lower_quartile
            upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
            lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
            upper_error = abs(median-upper_whisker)
            lower_error = abs(median-lower_whisker)

            label = None
            if group_i == 0:
                if planner == 'dc2g': label = 'DC2G'
                elif planner == 'dc2g_rescale': label = 'DC2G-R'
                elif planner == 'frontier': label = 'Frontier'
                # elif planner == 'oracle': label = 'Oracle'
            rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, median, width, color=planner_color_dict[planner], label=label, yerr=[[lower_error], [upper_error]], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, planner_mean, width, color=planner_color_dict[planner], label=label, yerr=planner_std, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # print(world_difficulty, planner, planner_mean, planner_std)
        xticklabels.append(world_id)
        # l = [num_steps_arr[:,i] for i in range(num_planners)]
        # l = [None]*group_i*num_planners + l
        # ax.boxplot(num_steps_arr)

    ax.set_xticklabels(xticklabels, rotation=45, ha="right")
    ax.legend()
    # plt.savefig('{}/{}__{:%Y-%m-%d_%H_%M_%S}.pdf'.format(save_dir, latest_file.split('/')[-1][:-4], datetime.datetime.now()))
    plt.show()

def plot_per_world2(results=None):
    if results is None:
        # list_of_files = glob.glob('{}/*.pkl'.format(save_dir))
        # latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = "{}/2018-09-13_15_51_57.pkl".format(save_dir)
        with open(latest_file, 'rb') as handle:
            results = pickle.load(handle)

    with open("image_similarities.p", "rb") as f:
        similarity_dict = pickle.load(f)

    planners = ['oracle', 'dc2g', 'frontier']
    
    results_by_world = {}
    world_ids = []
    for world_difficulty in results:
        for trial in results[world_difficulty]:
            world_id = results[world_difficulty][trial]['world_id']
            if world_id not in results_by_world:
                results_by_world[world_id] = {}
            results_by_world[world_id][trial] = {}
            for planner in planners:
                # world_id = results[world_difficulty][trial]['dc2g']['world_id']
                results_by_world[world_id][trial][planner] = results[world_difficulty][trial][planner]

    # planners = ['oracle', 'dc2g', 'dc2g_rescale', 'frontier']
    # planners = ['oracle', 'dc2g', 'dc2g_rescale']
    # planners = results[0].keys()
    num_planners = len(planners)
    # num_bar_groups = len(results.keys())
    num_bar_groups = len(results_by_world.keys())
    # num_trials = len(results[list(results.keys())[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('Extra Time to Goal (%)')
    ax.set_xlabel('Normalized Difference from Training Images')

    xticklabels = []

    world_ids = sorted(results_by_world.keys())

    means = []
    lower_errs = []
    upper_errs = []
    similarities = []

    dc2g_data = []
    planners_labeled = 0

    for group_i, world_id in enumerate(world_ids):
        trial_nums = results_by_world[world_id]
        num_trials = len(trial_nums)
        num_steps_arr = np.empty((num_trials, num_planners))
        extra_num_steps_arr = np.empty((num_trials, num_planners))
        pct_incr_num_steps_arr = np.empty((num_trials, num_planners))
        success_arr = np.empty((num_trials, num_planners))
        planner = "oracle"
        for trial_i, trial_num in enumerate(trial_nums):
            num_steps_arr[trial_i, num_planners - 1] = results_by_world[world_id][trial_num][planner]['num_steps']
            success_arr[trial_i, num_planners - 1] = results_by_world[world_id][trial_num][planner]['success']
        for i, planner in enumerate(planners):
            if planner == "oracle": continue
            for trial_i, trial_num in enumerate(trial_nums):
                num_steps_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['num_steps']
                extra_num_steps_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['num_steps'] - results_by_world[world_id][trial_num]['oracle']['num_steps']
                pct_incr_num_steps_arr[trial_i, i] = (results_by_world[world_id][trial_num][planner]['num_steps'] - results_by_world[world_id][trial_num]['oracle']['num_steps']) / float(results_by_world[world_id][trial_num]['oracle']['num_steps'])
                success_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['success']
            # print(extra_num_steps_arr)
            # planner_mean = np.mean(extra_num_steps_arr[:, i])
            # planner_std = np.std(extra_num_steps_arr[:, i])
            # planner_mean = np.mean(num_steps_arr[:, i])
            # planner_std = np.std(num_steps_arr[:, i])
            # planner_mean = np.mean(pct_incr_num_steps_arr[:, i])
            # planner_std = np.std(pct_incr_num_steps_arr[:, i])

            data = extra_num_steps_arr[:,i]
            median = np.median(data)
            upper_quartile = np.percentile(data, 75)
            lower_quartile = np.percentile(data, 25)
            iqr = upper_quartile - lower_quartile
            upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
            lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
            upper_error = abs(median-upper_whisker)
            lower_error = abs(median-lower_whisker)

            if similarity_dict[world_id] < 0.01:
                continue
            label = None
            if group_i == 0:
                if planner == 'dc2g': label = 'DC2G'
                elif planner == 'dc2g_rescale': label = 'DC2G-R'
                elif planner == 'frontier': label = 'Frontier'
                # elif planner == 'oracle': label = 'Oracle'

            c = planner_color_dict[planner]
            bp = plt.boxplot([data], positions=[similarity_dict[world_id]], widths=0.005, patch_artist=True, 
                        boxprops=dict(facecolor="None", color=c),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        flierprops=dict(color=c, markeredgecolor=c),
                        medianprops=dict(color=c))
            # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            #     plt.setp(bp[element], color=planner_color_dict[planner])
            # for patch in bp['boxes']:
            #     patch.set(facecolor="None")
            # rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, median, width, color=planner_color_dict[planner], label=label, yerr=[[lower_error], [upper_error]], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, planner_mean, width, color=planner_color_dict[planner], label=label, yerr=planner_std, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # print(world_difficulty, planner, planner_mean, planner_std)
        # xticklabels.append(world_id)
        # l = [num_steps_arr[:,i] for i in range(num_planners)]
        # l = [None]*group_i*num_planners + l
        # ax.boxplot(num_steps_arr)

    xticks = np.round(np.arange(0, 1, step=0.1),1)
    ax.set_xticklabels(xticks)
    ax.set_xticks(xticks)
    ax.set_xlim([0,0.7])
    ax.legend()
    # plt.savefig('{}/{}__{:%Y-%m-%d_%H_%M_%S}.pdf'.format(save_dir, latest_file.split('/')[-1][:-4], datetime.datetime.now()))
    plt.show()

def plot_per_world3(results=None):
    if results is None:
        # list_of_files = glob.glob('{}/*.pkl'.format(save_dir))
        # latest_file = max(list_of_files, key=os.path.getctime)
        latest_file = "{}/2018-09-13_15_51_57.pkl".format(save_dir)
        with open(latest_file, 'rb') as handle:
            results = pickle.load(handle)

    with open("image_similarities.p", "rb") as f:
        similarity_dict = pickle.load(f)

    planners = ['oracle', 'dc2g', 'frontier']
    
    results_by_world = {}
    world_ids = []
    for world_difficulty in results:
        for trial in results[world_difficulty]:
            world_id = results[world_difficulty][trial]['world_id']
            if world_id not in results_by_world:
                results_by_world[world_id] = {}
            results_by_world[world_id][trial] = {}
            for planner in planners:
                # world_id = results[world_difficulty][trial]['dc2g']['world_id']
                results_by_world[world_id][trial][planner] = results[world_difficulty][trial][planner]

    # planners = ['oracle', 'dc2g', 'dc2g_rescale', 'frontier']
    # planners = ['oracle', 'dc2g', 'dc2g_rescale']
    # planners = results[0].keys()
    num_planners = len(planners)
    # num_bar_groups = len(results.keys())
    num_bar_groups = len(results_by_world.keys())
    # num_trials = len(results[list(results.keys())[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('Extra Time to Goal (%)')
    ax.set_xlabel('Normalized Difference from Training Images')

    xticklabels = []

    world_ids = sorted(results_by_world.keys())

    means = [[] for i in range(num_planners)]
    stds = [[] for i in range(num_planners)]
    similarities = [[] for i in range(num_planners)]
    # lower_errs = []
    # upper_errs = []
    # similarities = []

    dc2g_data = []
    planners_labeled = 0

    for group_i, world_id in enumerate(world_ids):
        trial_nums = results_by_world[world_id]
        num_trials = len(trial_nums)
        num_steps_arr = np.empty((num_trials, num_planners))
        extra_num_steps_arr = np.empty((num_trials, num_planners))
        pct_incr_num_steps_arr = np.empty((num_trials, num_planners))
        success_arr = np.empty((num_trials, num_planners))
        planner = "oracle"
        for trial_i, trial_num in enumerate(trial_nums):
            num_steps_arr[trial_i, num_planners - 1] = results_by_world[world_id][trial_num][planner]['num_steps']
            success_arr[trial_i, num_planners - 1] = results_by_world[world_id][trial_num][planner]['success']
        for i, planner in enumerate(planners):
            if planner == "oracle": continue
            for trial_i, trial_num in enumerate(trial_nums):
                num_steps_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['num_steps']
                extra_num_steps_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['num_steps'] - results_by_world[world_id][trial_num]['oracle']['num_steps']
                pct_incr_num_steps_arr[trial_i, i] = (results_by_world[world_id][trial_num][planner]['num_steps'] - results_by_world[world_id][trial_num]['oracle']['num_steps']) / float(results_by_world[world_id][trial_num]['oracle']['num_steps'])
                success_arr[trial_i, i] = results_by_world[world_id][trial_num][planner]['success']
            # print(extra_num_steps_arr)
            # planner_mean = np.mean(extra_num_steps_arr[:, i])
            # planner_std = np.std(extra_num_steps_arr[:, i])
            # planner_mean = np.mean(num_steps_arr[:, i])
            # planner_std = np.std(num_steps_arr[:, i])
            # planner_mean = np.mean(pct_incr_num_steps_arr[:, i])
            # planner_std = np.std(pct_incr_num_steps_arr[:, i])

            data = extra_num_steps_arr[:,i]
            median = np.median(data)
            upper_quartile = np.percentile(data, 75)
            lower_quartile = np.percentile(data, 25)
            iqr = upper_quartile - lower_quartile
            upper_whisker = data[data<=upper_quartile+1.5*iqr].max()
            lower_whisker = data[data>=lower_quartile-1.5*iqr].min()
            upper_error = abs(median-upper_whisker)
            lower_error = abs(median-lower_whisker)

            mean = np.mean(data)
            std = np.std(data)

            if similarity_dict[world_id] < 0.01:
                continue
            label = None
            if group_i == 0:
                if planner == 'dc2g': label = 'DC2G'
                elif planner == 'dc2g_rescale': label = 'DC2G-R'
                elif planner == 'frontier': label = 'Frontier'
                # elif planner == 'oracle': label = 'Oracle'

            means[i].append(mean)
            stds[i].append(std)
            similarities[i].append(similarity_dict[world_id])
            # c = planner_color_dict[planner]
            # bp = plt.boxplot([data], positions=[similarity_dict[world_id]], widths=0.005, patch_artist=True, 
            #             boxprops=dict(facecolor="None", color=c),
            #             capprops=dict(color=c),
            #             whiskerprops=dict(color=c),
            #             flierprops=dict(color=c, markeredgecolor=c),
            #             medianprops=dict(color=c))
            # for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            #     plt.setp(bp[element], color=planner_color_dict[planner])
            # for patch in bp['boxes']:
            #     patch.set(facecolor="None")
            # rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, median, width, color=planner_color_dict[planner], label=label, yerr=[[lower_error], [upper_error]], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # rects = ax.bar(ind[group_i] - bar_group_width/2.0 + i*width, planner_mean, width, color=planner_color_dict[planner], label=label, yerr=planner_std, error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
            # print(world_difficulty, planner, planner_mean, planner_std)
        # xticklabels.append(world_id)
        # l = [num_steps_arr[:,i] for i in range(num_planners)]
        # l = [None]*group_i*num_planners + l
        # ax.boxplot(num_steps_arr)



    for i, planner in enumerate(planners):
        if planner == "oracle": continue
        label = None
        if planner == 'dc2g': label = 'DC2G'
        elif planner == 'dc2g_rescale': label = 'DC2G-R'
        elif planner == 'frontier': label = 'Frontier'
        ranked_similarities = sorted(similarities[i])
        ranked_means = [img for _,img in sorted(zip(similarities[i], means[i]))]
        ranked_stds = [img for _,img in sorted(zip(similarities[i], stds[i]))]
        ax.plot(ranked_similarities, ranked_means, lw=2, label=label, color=planner_color_dict[planner])
        ax.fill_between(ranked_similarities, np.array(ranked_means)+np.array(ranked_stds), np.array(ranked_means)-np.array(ranked_stds), facecolor=planner_color_dict[planner], alpha=0.5)

    xticks = np.round(np.arange(0, 1, step=0.1),1)
    ax.set_xticklabels(xticks)
    ax.set_xticks(xticks)
    ax.set_xlim([0,0.7])
    ax.legend()
    # plt.savefig('{}/{}__{:%Y-%m-%d_%H_%M_%S}.pdf'.format(save_dir, latest_file.split('/')[-1][:-4], datetime.datetime.now()))
    plt.show()


if __name__ == '__main__':
    # results = run_experiment()
    # planner = 'frontier'
    # for category in results2:
    #     for trial_num in results2[category]:
    #         results[category][trial_num][planner] = results2[category][trial_num][planner]
    # print("results = {}".format(results))
    results = {'very_hard': {0: {'oracle': {'num_steps': 23, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 110, 'success': True}}, 1: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 276, 'success': True}}, 2: {'oracle': {'num_steps': 26, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 28, 'success': True}, 'frontier': {'num_steps': 127, 'success': True}}, 3: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 101, 'success': True}}, 4: {'oracle': {'num_steps': 31, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 56, 'success': True}, 'frontier': {'num_steps': 164, 'success': True}}, 5: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 248, 'success': True}}, 6: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 267, 'success': True}}, 7: {'oracle': {'num_steps': 20, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 24, 'success': True}, 'frontier': {'num_steps': 194, 'success': True}}, 8: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 56, 'success': True}, 'frontier': {'num_steps': 121, 'success': True}}, 9: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 68, 'success': True}, 'frontier': {'num_steps': 204, 'success': True}}, 10: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 54, 'success': True}, 'frontier': {'num_steps': 273, 'success': True}}, 11: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 193, 'success': True}, 'frontier': {'num_steps': 102, 'success': True}}, 12: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 179, 'success': True}}, 13: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 83, 'success': True}}, 14: {'oracle': {'num_steps': 21, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 104, 'success': True}}, 15: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 209, 'success': True}}, 16: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 75, 'success': True}}, 17: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 74, 'success': True}, 'frontier': {'num_steps': 68, 'success': True}}, 18: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 120, 'success': True}}, 19: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 56, 'success': True}, 'frontier': {'num_steps': 81, 'success': True}}, 20: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 56, 'success': True}, 'frontier': {'num_steps': 63, 'success': True}}, 21: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 133, 'success': True}, 'frontier': {'num_steps': 219, 'success': True}}, 22: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 79, 'success': True}}, 23: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 239, 'success': True}}, 24: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 30, 'success': True}, 'frontier': {'num_steps': 119, 'success': True}}, 25: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 48, 'success': True}, 'frontier': {'num_steps': 268, 'success': True}}, 26: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 103, 'success': True}, 'frontier': {'num_steps': 70, 'success': True}}, 27: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 279, 'success': True}}, 28: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 99, 'success': True}, 'frontier': {'num_steps': 239, 'success': True}}, 29: {'oracle': {'num_steps': 26, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 38, 'success': True}, 'frontier': {'num_steps': 257, 'success': True}}, 30: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 162, 'success': True}, 'frontier': {'num_steps': 113, 'success': True}}, 31: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 217, 'success': True}}, 32: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 61, 'success': True}, 'frontier': {'num_steps': 118, 'success': True}}, 33: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 69, 'success': True}, 'frontier': {'num_steps': 64, 'success': True}}, 34: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 133, 'success': True}, 'frontier': {'num_steps': 94, 'success': True}}, 35: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 186, 'success': True}, 'frontier': {'num_steps': 276, 'success': True}}, 36: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 67, 'success': True}, 'frontier': {'num_steps': 221, 'success': True}}, 37: {'oracle': {'num_steps': 55, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 122, 'success': True}, 'frontier': {'num_steps': 324, 'success': True}}, 38: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 183, 'success': True}}, 39: {'oracle': {'num_steps': 25, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 89, 'success': True}}, 40: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 100, 'success': True}}, 41: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 34, 'success': True}, 'frontier': {'num_steps': 323, 'success': True}}, 42: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 59, 'success': True}}, 43: {'oracle': {'num_steps': 52, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 189, 'success': True}, 'frontier': {'num_steps': 291, 'success': True}}, 44: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 57, 'success': True}, 'frontier': {'num_steps': 101, 'success': True}}, 45: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 147, 'success': True}}, 46: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 131, 'success': True}}, 47: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 57, 'success': True}, 'frontier': {'num_steps': 61, 'success': True}}, 48: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 171, 'success': True}}, 49: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 106, 'success': True}, 'frontier': {'num_steps': 186, 'success': True}}, 50: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 134, 'success': True}}, 51: {'oracle': {'num_steps': 22, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 22, 'success': True}, 'frontier': {'num_steps': 100, 'success': True}}, 52: {'oracle': {'num_steps': 27, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 216, 'success': True}}, 53: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 103, 'success': True}}, 54: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 261, 'success': True}}, 55: {'oracle': {'num_steps': 50, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 109, 'success': True}, 'frontier': {'num_steps': 113, 'success': True}}, 56: {'oracle': {'num_steps': 21, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 85, 'success': True}, 'frontier': {'num_steps': 223, 'success': True}}, 57: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 52, 'success': True}, 'frontier': {'num_steps': 294, 'success': True}}, 58: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 90, 'success': True}}, 59: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 132, 'success': True}}, 60: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 54, 'success': True}, 'frontier': {'num_steps': 260, 'success': True}}, 61: {'oracle': {'num_steps': 23, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 71, 'success': True}}, 62: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 86, 'success': True}, 'frontier': {'num_steps': 233, 'success': True}}, 63: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 95, 'success': True}, 'frontier': {'num_steps': 208, 'success': True}}, 64: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 52, 'success': True}, 'frontier': {'num_steps': 47, 'success': True}}, 65: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 117, 'success': True}}, 66: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 67, 'success': True}, 'frontier': {'num_steps': 81, 'success': True}}, 67: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 68, 'success': True}, 'frontier': {'num_steps': 108, 'success': True}}, 68: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 47, 'success': True}, 'frontier': {'num_steps': 155, 'success': True}}, 69: {'oracle': {'num_steps': 27, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 199, 'success': True}}, 70: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn004m000h004', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 171, 'success': True}}, 71: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 83, 'success': True}, 'frontier': {'num_steps': 57, 'success': True}}, 72: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 286, 'success': True}}, 73: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 140, 'success': True}}, 74: {'oracle': {'num_steps': 21, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 46, 'success': True}}, 75: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 69, 'success': True}}, 76: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 171, 'success': True}}, 77: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 73, 'success': True}}, 78: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 131, 'success': True}}, 79: {'oracle': {'num_steps': 22, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 26, 'success': True}, 'frontier': {'num_steps': 108, 'success': True}}, 80: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 170, 'success': True}}, 81: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 80, 'success': True}}, 82: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 73, 'success': True}}, 83: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 52, 'success': True}, 'frontier': {'num_steps': 115, 'success': True}}, 84: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 98, 'success': True}, 'frontier': {'num_steps': 206, 'success': True}}, 85: {'oracle': {'num_steps': 31, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 118, 'success': True}}, 86: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 61, 'success': True}, 'frontier': {'num_steps': 66, 'success': True}}, 87: {'oracle': {'num_steps': 26, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 94, 'success': True}}, 88: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 48, 'success': True}, 'frontier': {'num_steps': 84, 'success': True}}, 89: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 72, 'success': True}, 'frontier': {'num_steps': 85, 'success': True}}, 90: {'oracle': {'num_steps': 20, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 28, 'success': True}, 'frontier': {'num_steps': 68, 'success': True}}, 91: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn004m000h003', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 181, 'success': True}}, 92: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn004m000h002', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 146, 'success': True}}, 93: {'oracle': {'num_steps': 61, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 138, 'success': True}, 'frontier': {'num_steps': 164, 'success': True}}, 94: {'oracle': {'num_steps': 19, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 19, 'success': True}, 'frontier': {'num_steps': 76, 'success': True}}, 95: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 86, 'success': True}}, 96: {'oracle': {'num_steps': 31, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 441, 'success': True}}, 97: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 329, 'success': True}}, 98: {'oracle': {'num_steps': 23, 'success': True}, 'world_id': 'worldn004m000h001', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 68, 'success': True}}, 99: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn004m000h000', 'dc2g': {'num_steps': 38, 'success': True}, 'frontier': {'num_steps': 255, 'success': True}}}, 'medium': {0: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 466, 'success': True}}, 1: {'oracle': {'num_steps': 67, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 83, 'success': True}, 'frontier': {'num_steps': 175, 'success': True}}, 2: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 212, 'success': True}}, 3: {'oracle': {'num_steps': 68, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 143, 'success': True}}, 4: {'oracle': {'num_steps': 89, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 113, 'success': True}, 'frontier': {'num_steps': 237, 'success': True}}, 5: {'oracle': {'num_steps': 85, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 93, 'success': True}, 'frontier': {'num_steps': 265, 'success': True}}, 6: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 142, 'success': True}, 'frontier': {'num_steps': 500, 'success': True}}, 7: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 475, 'success': True}}, 8: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 104, 'success': True}, 'frontier': {'num_steps': 492, 'success': True}}, 9: {'oracle': {'num_steps': 88, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 92, 'success': True}, 'frontier': {'num_steps': 297, 'success': True}}, 10: {'oracle': {'num_steps': 57, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 78, 'success': True}, 'frontier': {'num_steps': 468, 'success': True}}, 11: {'oracle': {'num_steps': 79, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 81, 'success': True}, 'frontier': {'num_steps': 172, 'success': True}}, 12: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 172, 'success': True}, 'frontier': {'num_steps': 473, 'success': True}}, 13: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 61, 'success': True}, 'frontier': {'num_steps': 258, 'success': True}}, 14: {'oracle': {'num_steps': 76, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 209, 'success': True}, 'frontier': {'num_steps': 457, 'success': True}}, 15: {'oracle': {'num_steps': 69, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 259, 'success': True}}, 16: {'oracle': {'num_steps': 83, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 161, 'success': True}, 'frontier': {'num_steps': 150, 'success': True}}, 17: {'oracle': {'num_steps': 76, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 86, 'success': True}, 'frontier': {'num_steps': 228, 'success': True}}, 18: {'oracle': {'num_steps': 67, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 77, 'success': True}, 'frontier': {'num_steps': 169, 'success': True}}, 19: {'oracle': {'num_steps': 79, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 152, 'success': True}, 'frontier': {'num_steps': 406, 'success': True}}, 20: {'oracle': {'num_steps': 89, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 119, 'success': True}, 'frontier': {'num_steps': 222, 'success': True}}, 21: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 90, 'success': True}, 'frontier': {'num_steps': 75, 'success': True}}, 22: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 72, 'success': True}, 'frontier': {'num_steps': 167, 'success': True}}, 23: {'oracle': {'num_steps': 72, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 77, 'success': True}, 'frontier': {'num_steps': 138, 'success': True}}, 24: {'oracle': {'num_steps': 64, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 66, 'success': True}, 'frontier': {'num_steps': 362, 'success': True}}, 25: {'oracle': {'num_steps': 68, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 346, 'success': True}}, 26: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 132, 'success': True}}, 27: {'oracle': {'num_steps': 101, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 111, 'success': True}, 'frontier': {'num_steps': 177, 'success': True}}, 28: {'oracle': {'num_steps': 81, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 129, 'success': True}, 'frontier': {'num_steps': 163, 'success': True}}, 29: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 190, 'success': True}}, 30: {'oracle': {'num_steps': 71, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 85, 'success': True}, 'frontier': {'num_steps': 237, 'success': True}}, 31: {'oracle': {'num_steps': 71, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 101, 'success': True}, 'frontier': {'num_steps': 181, 'success': True}}, 32: {'oracle': {'num_steps': 89, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 113, 'success': True}, 'frontier': {'num_steps': 324, 'success': True}}, 33: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 81, 'success': True}, 'frontier': {'num_steps': 337, 'success': True}}, 34: {'oracle': {'num_steps': 76, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 85, 'success': True}, 'frontier': {'num_steps': 142, 'success': True}}, 35: {'oracle': {'num_steps': 81, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 83, 'success': True}, 'frontier': {'num_steps': 261, 'success': True}}, 36: {'oracle': {'num_steps': 60, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 253, 'success': True}, 'frontier': {'num_steps': 131, 'success': True}}, 37: {'oracle': {'num_steps': 81, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 87, 'success': True}, 'frontier': {'num_steps': 289, 'success': True}}, 38: {'oracle': {'num_steps': 72, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 83, 'success': True}, 'frontier': {'num_steps': 262, 'success': True}}, 39: {'oracle': {'num_steps': 57, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 216, 'success': True}, 'frontier': {'num_steps': 236, 'success': True}}, 40: {'oracle': {'num_steps': 83, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 177, 'success': True}, 'frontier': {'num_steps': 142, 'success': True}}, 41: {'oracle': {'num_steps': 88, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 96, 'success': True}, 'frontier': {'num_steps': 270, 'success': True}}, 42: {'oracle': {'num_steps': 68, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 346, 'success': True}}, 43: {'oracle': {'num_steps': 74, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 100, 'success': True}, 'frontier': {'num_steps': 152, 'success': True}}, 44: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 500, 'success': True}, 'frontier': {'num_steps': 258, 'success': True}}, 45: {'oracle': {'num_steps': 66, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 72, 'success': True}, 'frontier': {'num_steps': 127, 'success': True}}, 46: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 87, 'success': True}, 'frontier': {'num_steps': 232, 'success': True}}, 47: {'oracle': {'num_steps': 70, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 133, 'success': True}, 'frontier': {'num_steps': 153, 'success': True}}, 48: {'oracle': {'num_steps': 80, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 327, 'success': True}}, 49: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 139, 'success': True}}, 50: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 47, 'success': True}, 'frontier': {'num_steps': 167, 'success': True}}, 51: {'oracle': {'num_steps': 63, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 214, 'success': True}, 'frontier': {'num_steps': 138, 'success': True}}, 52: {'oracle': {'num_steps': 81, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 91, 'success': True}, 'frontier': {'num_steps': 161, 'success': True}}, 53: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 226, 'success': True}}, 54: {'oracle': {'num_steps': 71, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 206, 'success': True}, 'frontier': {'num_steps': 150, 'success': True}}, 55: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 121, 'success': True}, 'frontier': {'num_steps': 476, 'success': True}}, 56: {'oracle': {'num_steps': 107, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 111, 'success': True}, 'frontier': {'num_steps': 176, 'success': True}}, 57: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 223, 'success': True}}, 58: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 174, 'success': True}}, 59: {'oracle': {'num_steps': 61, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 114, 'success': True}, 'frontier': {'num_steps': 194, 'success': True}}, 60: {'oracle': {'num_steps': 73, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 78, 'success': True}, 'frontier': {'num_steps': 143, 'success': True}}, 61: {'oracle': {'num_steps': 71, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 270, 'success': True}}, 62: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 217, 'success': True}}, 63: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 98, 'success': True}, 'frontier': {'num_steps': 59, 'success': True}}, 64: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 246, 'success': True}}, 65: {'oracle': {'num_steps': 66, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 105, 'success': True}, 'frontier': {'num_steps': 301, 'success': True}}, 66: {'oracle': {'num_steps': 71, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 77, 'success': True}, 'frontier': {'num_steps': 257, 'success': True}}, 67: {'oracle': {'num_steps': 52, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 130, 'success': True}}, 68: {'oracle': {'num_steps': 88, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 106, 'success': True}, 'frontier': {'num_steps': 272, 'success': True}}, 69: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 221, 'success': True}}, 70: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 48, 'success': True}, 'frontier': {'num_steps': 169, 'success': True}}, 71: {'oracle': {'num_steps': 57, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 156, 'success': True}}, 72: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 275, 'success': True}}, 73: {'oracle': {'num_steps': 72, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 90, 'success': True}, 'frontier': {'num_steps': 139, 'success': True}}, 74: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn001m003h000', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 265, 'success': True}}, 75: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 99, 'success': True}, 'frontier': {'num_steps': 472, 'success': True}}, 76: {'oracle': {'num_steps': 75, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 78, 'success': True}, 'frontier': {'num_steps': 153, 'success': True}}, 77: {'oracle': {'num_steps': 70, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 80, 'success': True}, 'frontier': {'num_steps': 242, 'success': True}}, 78: {'oracle': {'num_steps': 70, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 160, 'success': True}}, 79: {'oracle': {'num_steps': 110, 'success': True}, 'world_id': 'worldn001m004h000', 'dc2g': {'num_steps': 128, 'success': True}, 'frontier': {'num_steps': 335, 'success': True}}, 80: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 245, 'success': True}}, 81: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 69, 'success': True}, 'frontier': {'num_steps': 111, 'success': True}}, 82: {'oracle': {'num_steps': 74, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 83, 'success': True}, 'frontier': {'num_steps': 280, 'success': True}}, 83: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 52, 'success': True}, 'frontier': {'num_steps': 138, 'success': True}}, 84: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 190, 'success': True}}, 85: {'oracle': {'num_steps': 93, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 109, 'success': True}, 'frontier': {'num_steps': 259, 'success': True}}, 86: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 65, 'success': True}, 'frontier': {'num_steps': 179, 'success': True}}, 87: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 111, 'success': True}, 'frontier': {'num_steps': 431, 'success': True}}, 88: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 57, 'success': True}, 'frontier': {'num_steps': 277, 'success': True}}, 89: {'oracle': {'num_steps': 77, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 82, 'success': True}, 'frontier': {'num_steps': 153, 'success': True}}, 90: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn001m003h001', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 485, 'success': True}}, 91: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 216, 'success': True}}, 92: {'oracle': {'num_steps': 79, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 105, 'success': True}, 'frontier': {'num_steps': 315, 'success': True}}, 93: {'oracle': {'num_steps': 79, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 95, 'success': True}, 'frontier': {'num_steps': 355, 'success': True}}, 94: {'oracle': {'num_steps': 75, 'success': True}, 'world_id': 'worldn001m004h002', 'dc2g': {'num_steps': 142, 'success': True}, 'frontier': {'num_steps': 152, 'success': True}}, 95: {'oracle': {'num_steps': 78, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 88, 'success': True}, 'frontier': {'num_steps': 152, 'success': True}}, 96: {'oracle': {'num_steps': 72, 'success': True}, 'world_id': 'worldn001m004h001', 'dc2g': {'num_steps': 77, 'success': True}, 'frontier': {'num_steps': 140, 'success': True}}, 97: {'oracle': {'num_steps': 69, 'success': True}, 'world_id': 'worldn001m003h002', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 337, 'success': True}}, 98: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn001m004h003', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 134, 'success': True}}, 99: {'oracle': {'num_steps': 66, 'success': True}, 'world_id': 'worldn001m004h004', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 200, 'success': True}}}, 'hard': {0: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn002m000h002', 'dc2g': {'num_steps': 36, 'success': True}, 'frontier': {'num_steps': 156, 'success': True}}, 1: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn002m001h002', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 323, 'success': True}}, 2: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn002m002h010', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 127, 'success': True}}, 3: {'oracle': {'num_steps': 31, 'success': True}, 'world_id': 'worldn002m002h012', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 57, 'success': True}}, 4: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn002m001h006', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 160, 'success': True}}, 5: {'oracle': {'num_steps': 31, 'success': True}, 'world_id': 'worldn002m002h013', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 48, 'success': True}}, 6: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn002m000h002', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 193, 'success': True}}, 7: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn002m000h000', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 151, 'success': True}}, 8: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn002m002h010', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 281, 'success': True}}, 9: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn002m002h005', 'dc2g': {'num_steps': 186, 'success': True}, 'frontier': {'num_steps': 80, 'success': True}}, 10: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn002m002h001', 'dc2g': {'num_steps': 48, 'success': True}, 'frontier': {'num_steps': 136, 'success': True}}, 11: {'oracle': {'num_steps': 61, 'success': True}, 'world_id': 'worldn002m000h004', 'dc2g': {'num_steps': 67, 'success': True}, 'frontier': {'num_steps': 125, 'success': True}}, 12: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn002m001h006', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 186, 'success': True}}, 13: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn002m002h004', 'dc2g': {'num_steps': 48, 'success': True}, 'frontier': {'num_steps': 87, 'success': True}}, 14: {'oracle': {'num_steps': 60, 'success': True}, 'world_id': 'worldn002m001h000', 'dc2g': {'num_steps': 101, 'success': True}, 'frontier': {'num_steps': 240, 'success': True}}, 15: {'oracle': {'num_steps': 26, 'success': True}, 'world_id': 'worldn002m002h004', 'dc2g': {'num_steps': 26, 'success': True}, 'frontier': {'num_steps': 94, 'success': True}}, 16: {'oracle': {'num_steps': 31, 'success': True}, 'world_id': 'worldn002m001h005', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 197, 'success': True}}, 17: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn002m002h002', 'dc2g': {'num_steps': 41, 'success': True}, 'frontier': {'num_steps': 72, 'success': True}}, 18: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn002m000h001', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 166, 'success': True}}, 19: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn002m002h006', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 100, 'success': True}}, 20: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn002m000h003', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 204, 'success': True}}, 21: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 69, 'success': True}, 'frontier': {'num_steps': 221, 'success': True}}, 22: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 92, 'success': True}}, 23: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn002m000h003', 'dc2g': {'num_steps': 306, 'success': True}, 'frontier': {'num_steps': 303, 'success': True}}, 24: {'oracle': {'num_steps': 60, 'success': True}, 'world_id': 'worldn002m001h006', 'dc2g': {'num_steps': 74, 'success': True}, 'frontier': {'num_steps': 156, 'success': True}}, 25: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn002m000h001', 'dc2g': {'num_steps': 72, 'success': True}, 'frontier': {'num_steps': 96, 'success': True}}, 26: {'oracle': {'num_steps': 24, 'success': True}, 'world_id': 'worldn002m001h005', 'dc2g': {'num_steps': 36, 'success': True}, 'frontier': {'num_steps': 48, 'success': True}}, 27: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn002m002h009', 'dc2g': {'num_steps': 65, 'success': True}, 'frontier': {'num_steps': 329, 'success': True}}, 28: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn002m002h011', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 281, 'success': True}}, 29: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn002m000h005', 'dc2g': {'num_steps': 69, 'success': True}, 'frontier': {'num_steps': 175, 'success': True}}, 30: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn002m002h014', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 266, 'success': True}}, 31: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn002m001h006', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 57, 'success': True}}, 32: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn002m002h012', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 137, 'success': True}}, 33: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn002m002h005', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 329, 'success': True}}, 34: {'oracle': {'num_steps': 63, 'success': True}, 'world_id': 'worldn002m001h007', 'dc2g': {'num_steps': 177, 'success': True}, 'frontier': {'num_steps': 130, 'success': True}}, 35: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn002m002h014', 'dc2g': {'num_steps': 37, 'success': True}, 'frontier': {'num_steps': 250, 'success': True}}, 36: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn002m002h007', 'dc2g': {'num_steps': 57, 'success': True}, 'frontier': {'num_steps': 108, 'success': True}}, 37: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn002m001h002', 'dc2g': {'num_steps': 41, 'success': True}, 'frontier': {'num_steps': 267, 'success': True}}, 38: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn002m002h012', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 112, 'success': True}}, 39: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn002m000h005', 'dc2g': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 153, 'success': True}}, 40: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn002m000h004', 'dc2g': {'num_steps': 61, 'success': True}, 'frontier': {'num_steps': 162, 'success': True}}, 41: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn002m002h009', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 237, 'success': True}}, 42: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn002m001h002', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 166, 'success': True}}, 43: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 293, 'success': True}}, 44: {'oracle': {'num_steps': 27, 'success': True}, 'world_id': 'worldn002m002h013', 'dc2g': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 190, 'success': True}}, 45: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn002m001h004', 'dc2g': {'num_steps': 68, 'success': True}, 'frontier': {'num_steps': 101, 'success': True}}, 46: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn002m002h000', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 390, 'success': True}}, 47: {'oracle': {'num_steps': 55, 'success': True}, 'world_id': 'worldn002m000h004', 'dc2g': {'num_steps': 69, 'success': True}, 'frontier': {'num_steps': 123, 'success': True}}, 48: {'oracle': {'num_steps': 27, 'success': True}, 'world_id': 'worldn002m002h015', 'dc2g': {'num_steps': 29, 'success': True}, 'frontier': {'num_steps': 379, 'success': True}}, 49: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 278, 'success': True}}, 50: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn002m002h002', 'dc2g': {'num_steps': 41, 'success': True}, 'frontier': {'num_steps': 114, 'success': True}}, 51: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn002m001h004', 'dc2g': {'num_steps': 57, 'success': True}, 'frontier': {'num_steps': 163, 'success': True}}, 52: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn002m001h001', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 119, 'success': True}}, 53: {'oracle': {'num_steps': 25, 'success': True}, 'world_id': 'worldn002m002h012', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 39, 'success': True}}, 54: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn002m001h005', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 187, 'success': True}}, 55: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn002m000h004', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 86, 'success': True}}, 56: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn002m000h003', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 91, 'success': True}}, 57: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn002m002h009', 'dc2g': {'num_steps': 61, 'success': True}, 'frontier': {'num_steps': 193, 'success': True}}, 58: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn002m000h006', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 148, 'success': True}}, 59: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn002m000h002', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 141, 'success': True}}, 60: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn002m000h001', 'dc2g': {'num_steps': 54, 'success': True}, 'frontier': {'num_steps': 224, 'success': True}}, 61: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn002m002h007', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 95, 'success': True}}, 62: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn002m002h000', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 58, 'success': True}}, 63: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 61, 'success': True}, 'frontier': {'num_steps': 136, 'success': True}}, 64: {'oracle': {'num_steps': 23, 'success': True}, 'world_id': 'worldn002m002h002', 'dc2g': {'num_steps': 33, 'success': True}, 'frontier': {'num_steps': 43, 'success': True}}, 65: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn002m002h012', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 47, 'success': True}}, 66: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn002m000h001', 'dc2g': {'num_steps': 78, 'success': True}, 'frontier': {'num_steps': 90, 'success': True}}, 67: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn002m000h003', 'dc2g': {'num_steps': 36, 'success': True}, 'frontier': {'num_steps': 317, 'success': True}}, 68: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn002m002h011', 'dc2g': {'num_steps': 72, 'success': True}, 'frontier': {'num_steps': 208, 'success': True}}, 69: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn002m002h015', 'dc2g': {'num_steps': 212, 'success': True}, 'frontier': {'num_steps': 372, 'success': True}}, 70: {'oracle': {'num_steps': 27, 'success': True}, 'world_id': 'worldn002m001h001', 'dc2g': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 61, 'success': True}}, 71: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 366, 'success': True}}, 72: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn002m002h009', 'dc2g': {'num_steps': 40, 'success': True}, 'frontier': {'num_steps': 378, 'success': True}}, 73: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn002m002h006', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 117, 'success': True}}, 74: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn002m000h005', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 213, 'success': True}}, 75: {'oracle': {'num_steps': 87, 'success': True}, 'world_id': 'worldn002m002h015', 'dc2g': {'num_steps': 257, 'success': True}, 'frontier': {'num_steps': 407, 'success': True}}, 76: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn002m002h001', 'dc2g': {'num_steps': 86, 'success': True}, 'frontier': {'num_steps': 74, 'success': True}}, 77: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn002m002h007', 'dc2g': {'num_steps': 54, 'success': True}, 'frontier': {'num_steps': 104, 'success': True}}, 78: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn002m000h006', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 158, 'success': True}}, 79: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn002m001h002', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 235, 'success': True}}, 80: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn002m000h002', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 163, 'success': True}}, 81: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn002m002h002', 'dc2g': {'num_steps': 35, 'success': True}, 'frontier': {'num_steps': 85, 'success': True}}, 82: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn002m002h006', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 337, 'success': True}}, 83: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn002m002h001', 'dc2g': {'num_steps': 66, 'success': True}, 'frontier': {'num_steps': 89, 'success': True}}, 84: {'oracle': {'num_steps': 38, 'success': True}, 'world_id': 'worldn002m001h004', 'dc2g': {'num_steps': 38, 'success': True}, 'frontier': {'num_steps': 146, 'success': True}}, 85: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn002m001h001', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 86, 'success': True}}, 86: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn002m002h008', 'dc2g': {'num_steps': 53, 'success': True}, 'frontier': {'num_steps': 176, 'success': True}}, 87: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn002m002h014', 'dc2g': {'num_steps': 138, 'success': True}, 'frontier': {'num_steps': 258, 'success': True}}, 88: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn002m002h005', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 87, 'success': True}}, 89: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn002m002h005', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 235, 'success': True}}, 90: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn002m000h002', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 145, 'success': True}}, 91: {'oracle': {'num_steps': 81, 'success': True}, 'world_id': 'worldn002m002h015', 'dc2g': {'num_steps': 113, 'success': True}, 'frontier': {'num_steps': 179, 'success': True}}, 92: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn002m001h005', 'dc2g': {'num_steps': 54, 'success': True}, 'frontier': {'num_steps': 100, 'success': True}}, 93: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn002m001h001', 'dc2g': {'num_steps': 41, 'success': True}, 'frontier': {'num_steps': 97, 'success': True}}, 94: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn002m000h004', 'dc2g': {'num_steps': 63, 'success': True}, 'frontier': {'num_steps': 81, 'success': True}}, 95: {'oracle': {'num_steps': 36, 'success': True}, 'world_id': 'worldn002m002h002', 'dc2g': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 150, 'success': True}}, 96: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn002m002h012', 'dc2g': {'num_steps': 38, 'success': True}, 'frontier': {'num_steps': 102, 'success': True}}, 97: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn002m002h006', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 115, 'success': True}}, 98: {'oracle': {'num_steps': 26, 'success': True}, 'world_id': 'worldn002m002h003', 'dc2g': {'num_steps': 28, 'success': True}, 'frontier': {'num_steps': 128, 'success': True}}, 99: {'oracle': {'num_steps': 47, 'success': True}, 'world_id': 'worldn002m002h006', 'dc2g': {'num_steps': 103, 'success': True}, 'frontier': {'num_steps': 119, 'success': True}}}, 'easy': {0: {'oracle': {'num_steps': 66, 'success': True}, 'world_id': 'worldn000m000h002', 'dc2g': {'num_steps': 82, 'success': True}, 'frontier': {'num_steps': 131, 'success': True}}, 1: {'oracle': {'num_steps': 89, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 104, 'success': True}, 'frontier': {'num_steps': 300, 'success': True}}, 2: {'oracle': {'num_steps': 50, 'success': True}, 'world_id': 'worldn000m000h003', 'dc2g': {'num_steps': 54, 'success': True}, 'frontier': {'num_steps': 96, 'success': True}}, 3: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn000m001h005', 'dc2g': {'num_steps': 74, 'success': True}, 'frontier': {'num_steps': 142, 'success': True}}, 4: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn000m000h000', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 83, 'success': True}}, 5: {'oracle': {'num_steps': 66, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 85, 'success': True}, 'frontier': {'num_steps': 123, 'success': True}}, 6: {'oracle': {'num_steps': 53, 'success': True}, 'world_id': 'worldn001m002h000', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 221, 'success': True}}, 7: {'oracle': {'num_steps': 37, 'success': True}, 'world_id': 'worldn000m001h002', 'dc2g': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 85, 'success': True}}, 8: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn000m001h006', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 94, 'success': True}}, 9: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn000m001h001', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 75, 'success': True}}, 10: {'oracle': {'num_steps': 57, 'success': True}, 'world_id': 'worldn001m000h004', 'dc2g': {'num_steps': 81, 'success': True}, 'frontier': {'num_steps': 126, 'success': True}}, 11: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn000m000h001', 'dc2g': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 97, 'success': True}}, 12: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn000m000h003', 'dc2g': {'num_steps': 47, 'success': True}, 'frontier': {'num_steps': 105, 'success': True}}, 13: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 298, 'success': True}}, 14: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn000m000h009', 'dc2g': {'num_steps': 90, 'success': True}, 'frontier': {'num_steps': 78, 'success': True}}, 15: {'oracle': {'num_steps': 53, 'success': True}, 'world_id': 'worldn001m000h001', 'dc2g': {'num_steps': 431, 'success': True}, 'frontier': {'num_steps': 325, 'success': True}}, 16: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn000m000h008', 'dc2g': {'num_steps': 83, 'success': True}, 'frontier': {'num_steps': 152, 'success': True}}, 17: {'oracle': {'num_steps': 68, 'success': True}, 'world_id': 'worldn000m000h010', 'dc2g': {'num_steps': 94, 'success': True}, 'frontier': {'num_steps': 118, 'success': True}}, 18: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn000m001h008', 'dc2g': {'num_steps': 34, 'success': True}, 'frontier': {'num_steps': 228, 'success': True}}, 19: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn000m001h007', 'dc2g': {'num_steps': 51, 'success': True}, 'frontier': {'num_steps': 166, 'success': True}}, 20: {'oracle': {'num_steps': 63, 'success': True}, 'world_id': 'worldn001m000h002', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 237, 'success': True}}, 21: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn000m000h008', 'dc2g': {'num_steps': 74, 'success': True}, 'frontier': {'num_steps': 84, 'success': True}}, 22: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn001m002h000', 'dc2g': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 220, 'success': True}}, 23: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn000m000h001', 'dc2g': {'num_steps': 81, 'success': True}, 'frontier': {'num_steps': 220, 'success': True}}, 24: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn001m000h003', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 179, 'success': True}}, 25: {'oracle': {'num_steps': 72, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 97, 'success': True}, 'frontier': {'num_steps': 221, 'success': True}}, 26: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn000m001h006', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 132, 'success': True}}, 27: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn001m001h005', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 207, 'success': True}}, 28: {'oracle': {'num_steps': 41, 'success': True}, 'world_id': 'worldn000m001h006', 'dc2g': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 121, 'success': True}}, 29: {'oracle': {'num_steps': 66, 'success': True}, 'world_id': 'worldn000m000h006', 'dc2g': {'num_steps': 76, 'success': True}, 'frontier': {'num_steps': 122, 'success': True}}, 30: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn001m000h004', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 69, 'success': True}}, 31: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn000m000h000', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 255, 'success': True}}, 32: {'oracle': {'num_steps': 80, 'success': True}, 'world_id': 'worldn001m001h001', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 296, 'success': True}}, 33: {'oracle': {'num_steps': 64, 'success': True}, 'world_id': 'worldn001m000h001', 'dc2g': {'num_steps': 500, 'success': True}, 'frontier': {'num_steps': 346, 'success': True}}, 34: {'oracle': {'num_steps': 53, 'success': True}, 'world_id': 'worldn000m000h007', 'dc2g': {'num_steps': 63, 'success': True}, 'frontier': {'num_steps': 163, 'success': True}}, 35: {'oracle': {'num_steps': 75, 'success': True}, 'world_id': 'worldn001m001h003', 'dc2g': {'num_steps': 102, 'success': True}, 'frontier': {'num_steps': 377, 'success': True}}, 36: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn001m001h001', 'dc2g': {'num_steps': 66, 'success': True}, 'frontier': {'num_steps': 212, 'success': True}}, 37: {'oracle': {'num_steps': 51, 'success': True}, 'world_id': 'worldn000m001h004', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 206, 'success': True}}, 38: {'oracle': {'num_steps': 74, 'success': True}, 'world_id': 'worldn000m000h010', 'dc2g': {'num_steps': 98, 'success': True}, 'frontier': {'num_steps': 179, 'success': True}}, 39: {'oracle': {'num_steps': 53, 'success': True}, 'world_id': 'worldn000m000h009', 'dc2g': {'num_steps': 93, 'success': True}, 'frontier': {'num_steps': 143, 'success': True}}, 40: {'oracle': {'num_steps': 97, 'success': True}, 'world_id': 'worldn001m001h003', 'dc2g': {'num_steps': 102, 'success': True}, 'frontier': {'num_steps': 395, 'success': True}}, 41: {'oracle': {'num_steps': 40, 'success': True}, 'world_id': 'worldn000m001h006', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 228, 'success': True}}, 42: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn000m001h002', 'dc2g': {'num_steps': 43, 'success': True}, 'frontier': {'num_steps': 221, 'success': True}}, 43: {'oracle': {'num_steps': 71, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 246, 'success': True}}, 44: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn001m001h005', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 244, 'success': True}}, 45: {'oracle': {'num_steps': 60, 'success': True}, 'world_id': 'worldn001m000h001', 'dc2g': {'num_steps': 320, 'success': True}, 'frontier': {'num_steps': 318, 'success': True}}, 46: {'oracle': {'num_steps': 64, 'success': True}, 'world_id': 'worldn001m000h001', 'dc2g': {'num_steps': 253, 'success': True}, 'frontier': {'num_steps': 336, 'success': True}}, 47: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn000m001h004', 'dc2g': {'num_steps': 40, 'success': True}, 'frontier': {'num_steps': 268, 'success': True}}, 48: {'oracle': {'num_steps': 30, 'success': True}, 'world_id': 'worldn000m000h007', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 76, 'success': True}}, 49: {'oracle': {'num_steps': 64, 'success': True}, 'world_id': 'worldn000m000h006', 'dc2g': {'num_steps': 84, 'success': True}, 'frontier': {'num_steps': 118, 'success': True}}, 50: {'oracle': {'num_steps': 53, 'success': True}, 'world_id': 'worldn000m000h003', 'dc2g': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 195, 'success': True}}, 51: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn000m000h006', 'dc2g': {'num_steps': 105, 'success': True}, 'frontier': {'num_steps': 439, 'success': True}}, 52: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn000m001h006', 'dc2g': {'num_steps': 65, 'success': True}, 'frontier': {'num_steps': 272, 'success': True}}, 53: {'oracle': {'num_steps': 74, 'success': True}, 'world_id': 'worldn001m001h001', 'dc2g': {'num_steps': 80, 'success': True}, 'frontier': {'num_steps': 224, 'success': True}}, 54: {'oracle': {'num_steps': 55, 'success': True}, 'world_id': 'worldn000m000h009', 'dc2g': {'num_steps': 65, 'success': True}, 'frontier': {'num_steps': 75, 'success': True}}, 55: {'oracle': {'num_steps': 68, 'success': True}, 'world_id': 'worldn001m000h000', 'dc2g': {'num_steps': 91, 'success': True}, 'frontier': {'num_steps': 229, 'success': True}}, 56: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn000m001h001', 'dc2g': {'num_steps': 40, 'success': True}, 'frontier': {'num_steps': 75, 'success': True}}, 57: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn000m000h004', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 72, 'success': True}}, 58: {'oracle': {'num_steps': 72, 'success': True}, 'world_id': 'worldn001m001h003', 'dc2g': {'num_steps': 101, 'success': True}, 'frontier': {'num_steps': 382, 'success': True}}, 59: {'oracle': {'num_steps': 57, 'success': True}, 'world_id': 'worldn000m000h006', 'dc2g': {'num_steps': 67, 'success': True}, 'frontier': {'num_steps': 181, 'success': True}}, 60: {'oracle': {'num_steps': 58, 'success': True}, 'world_id': 'worldn001m002h000', 'dc2g': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 196, 'success': True}}, 61: {'oracle': {'num_steps': 74, 'success': True}, 'world_id': 'worldn001m001h005', 'dc2g': {'num_steps': 80, 'success': True}, 'frontier': {'num_steps': 258, 'success': True}}, 62: {'oracle': {'num_steps': 45, 'success': True}, 'world_id': 'worldn000m001h004', 'dc2g': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 133, 'success': True}}, 63: {'oracle': {'num_steps': 75, 'success': True}, 'world_id': 'worldn001m001h005', 'dc2g': {'num_steps': 81, 'success': True}, 'frontier': {'num_steps': 223, 'success': True}}, 64: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn000m001h009', 'dc2g': {'num_steps': 57, 'success': True}, 'frontier': {'num_steps': 91, 'success': True}}, 65: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn000m000h001', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 244, 'success': True}}, 66: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn001m000h003', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 220, 'success': True}}, 67: {'oracle': {'num_steps': 74, 'success': True}, 'world_id': 'worldn001m000h002', 'dc2g': {'num_steps': 79, 'success': True}, 'frontier': {'num_steps': 196, 'success': True}}, 68: {'oracle': {'num_steps': 58, 'success': True}, 'world_id': 'worldn001m001h001', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 260, 'success': True}}, 69: {'oracle': {'num_steps': 46, 'success': True}, 'world_id': 'worldn000m000h001', 'dc2g': {'num_steps': 95, 'success': True}, 'frontier': {'num_steps': 194, 'success': True}}, 70: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn001m001h000', 'dc2g': {'num_steps': 121, 'success': True}, 'frontier': {'num_steps': 326, 'success': True}}, 71: {'oracle': {'num_steps': 63, 'success': True}, 'world_id': 'worldn000m000h007', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 141, 'success': True}}, 72: {'oracle': {'num_steps': 33, 'success': True}, 'world_id': 'worldn000m000h003', 'dc2g': {'num_steps': 41, 'success': True}, 'frontier': {'num_steps': 385, 'success': True}}, 73: {'oracle': {'num_steps': 63, 'success': True}, 'world_id': 'worldn000m000h009', 'dc2g': {'num_steps': 69, 'success': True}, 'frontier': {'num_steps': 177, 'success': True}}, 74: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn000m000h010', 'dc2g': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 296, 'success': True}}, 75: {'oracle': {'num_steps': 29, 'success': True}, 'world_id': 'worldn000m001h003', 'dc2g': {'num_steps': 30, 'success': True}, 'frontier': {'num_steps': 87, 'success': True}}, 76: {'oracle': {'num_steps': 32, 'success': True}, 'world_id': 'worldn000m001h008', 'dc2g': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 192, 'success': True}}, 77: {'oracle': {'num_steps': 44, 'success': True}, 'world_id': 'worldn000m001h007', 'dc2g': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 88, 'success': True}}, 78: {'oracle': {'num_steps': 34, 'success': True}, 'world_id': 'worldn000m001h001', 'dc2g': {'num_steps': 66, 'success': True}, 'frontier': {'num_steps': 49, 'success': True}}, 79: {'oracle': {'num_steps': 87, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 96, 'success': True}, 'frontier': {'num_steps': 230, 'success': True}}, 80: {'oracle': {'num_steps': 43, 'success': True}, 'world_id': 'worldn000m000h010', 'dc2g': {'num_steps': 103, 'success': True}, 'frontier': {'num_steps': 84, 'success': True}}, 81: {'oracle': {'num_steps': 95, 'success': True}, 'world_id': 'worldn001m000h000', 'dc2g': {'num_steps': 110, 'success': True}, 'frontier': {'num_steps': 190, 'success': True}}, 82: {'oracle': {'num_steps': 27, 'success': True}, 'world_id': 'worldn000m001h001', 'dc2g': {'num_steps': 48, 'success': True}, 'frontier': {'num_steps': 46, 'success': True}}, 83: {'oracle': {'num_steps': 57, 'success': True}, 'world_id': 'worldn000m000h001', 'dc2g': {'num_steps': 105, 'success': True}, 'frontier': {'num_steps': 267, 'success': True}}, 84: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn001m000h002', 'dc2g': {'num_steps': 62, 'success': True}, 'frontier': {'num_steps': 214, 'success': True}}, 85: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn000m001h004', 'dc2g': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 68, 'success': True}}, 86: {'oracle': {'num_steps': 39, 'success': True}, 'world_id': 'worldn000m000h003', 'dc2g': {'num_steps': 65, 'success': True}, 'frontier': {'num_steps': 241, 'success': True}}, 87: {'oracle': {'num_steps': 65, 'success': True}, 'world_id': 'worldn001m001h003', 'dc2g': {'num_steps': 78, 'success': True}, 'frontier': {'num_steps': 291, 'success': True}}, 88: {'oracle': {'num_steps': 49, 'success': True}, 'world_id': 'worldn000m001h005', 'dc2g': {'num_steps': 99, 'success': True}, 'frontier': {'num_steps': 239, 'success': True}}, 89: {'oracle': {'num_steps': 35, 'success': True}, 'world_id': 'worldn000m000h000', 'dc2g': {'num_steps': 65, 'success': True}, 'frontier': {'num_steps': 248, 'success': True}}, 90: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn000m001h008', 'dc2g': {'num_steps': 184, 'success': True}, 'frontier': {'num_steps': 74, 'success': True}}, 91: {'oracle': {'num_steps': 56, 'success': True}, 'world_id': 'worldn001m000h003', 'dc2g': {'num_steps': 80, 'success': True}, 'frontier': {'num_steps': 134, 'success': True}}, 92: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn001m000h002', 'dc2g': {'num_steps': 73, 'success': True}, 'frontier': {'num_steps': 181, 'success': True}}, 93: {'oracle': {'num_steps': 65, 'success': True}, 'world_id': 'worldn001m001h004', 'dc2g': {'num_steps': 80, 'success': True}, 'frontier': {'num_steps': 270, 'success': True}}, 94: {'oracle': {'num_steps': 28, 'success': True}, 'world_id': 'worldn000m001h003', 'dc2g': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 40, 'success': True}}, 95: {'oracle': {'num_steps': 42, 'success': True}, 'world_id': 'worldn000m001h008', 'dc2g': {'num_steps': 46, 'success': True}, 'frontier': {'num_steps': 174, 'success': True}}, 96: {'oracle': {'num_steps': 48, 'success': True}, 'world_id': 'worldn000m000h002', 'dc2g': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 122, 'success': True}}, 97: {'oracle': {'num_steps': 59, 'success': True}, 'world_id': 'worldn001m001h000', 'dc2g': {'num_steps': 123, 'success': True}, 'frontier': {'num_steps': 195, 'success': True}}, 98: {'oracle': {'num_steps': 54, 'success': True}, 'world_id': 'worldn001m000h004', 'dc2g': {'num_steps': 138, 'success': True}, 'frontier': {'num_steps': 133, 'success': True}}, 99: {'oracle': {'num_steps': 61, 'success': True}, 'world_id': 'worldn001m000h002', 'dc2g': {'num_steps': 78, 'success': True}, 'frontier': {'num_steps': 233, 'success': True}}}}


    plot(results)
    # plot_per_world2(results)
    # plot_per_world3(results)

    # plot3()
    # small FOV
    # results = {0: {'dc2g': {'num_steps': 61, 'success': True}, 'dc2g_rescale': {'num_steps': 119, 'success': True}, 'frontier': {'num_steps': 232, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 1: {'dc2g': {'num_steps': 40, 'success': True}, 'dc2g_rescale': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 151, 'success': True}, 'oracle': {'num_steps': 25, 'success': True}}, 2: {'dc2g': {'num_steps': 33, 'success': True}, 'dc2g_rescale': {'num_steps': 89, 'success': True}, 'frontier': {'num_steps': 300, 'success': True}, 'oracle': {'num_steps': 26, 'success': True}}, 3: {'dc2g': {'num_steps': 66, 'success': True}, 'dc2g_rescale': {'num_steps': 343, 'success': True}, 'frontier': {'num_steps': 184, 'success': True}, 'oracle': {'num_steps': 38, 'success': True}}, 4: {'dc2g': {'num_steps': 71, 'success': True}, 'dc2g_rescale': {'num_steps': 135, 'success': True}, 'frontier': {'num_steps': 180, 'success': True}, 'oracle': {'num_steps': 42, 'success': True}}, 5: {'dc2g': {'num_steps': 50, 'success': True}, 'dc2g_rescale': {'num_steps': 226, 'success': True}, 'frontier': {'num_steps': 185, 'success': True}, 'oracle': {'num_steps': 35, 'success': True}}, 6: {'dc2g': {'num_steps': 63, 'success': True}, 'dc2g_rescale': {'num_steps': 55, 'success': True}, 'frontier': {'num_steps': 158, 'success': True}, 'oracle': {'num_steps': 34, 'success': True}}, 7: {'dc2g': {'num_steps': 60, 'success': True}, 'dc2g_rescale': {'num_steps': 235, 'success': True}, 'frontier': {'num_steps': 157, 'success': True}, 'oracle': {'num_steps': 39, 'success': True}}, 8: {'dc2g': {'num_steps': 53, 'success': True}, 'dc2g_rescale': {'num_steps': 357, 'success': True}, 'frontier': {'num_steps': 212, 'success': True}, 'oracle': {'num_steps': 34, 'success': True}}, 9: {'dc2g': {'num_steps': 63, 'success': True}, 'dc2g_rescale': {'num_steps': 297, 'success': True}, 'frontier': {'num_steps': 166, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 10: {'dc2g': {'num_steps': 53, 'success': True}, 'dc2g_rescale': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 268, 'success': True}, 'oracle': {'num_steps': 24, 'success': True}}, 11: {'dc2g': {'num_steps': 56, 'success': True}, 'dc2g_rescale': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 235, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 12: {'dc2g': {'num_steps': 32, 'success': True}, 'dc2g_rescale': {'num_steps': 39, 'success': True}, 'frontier': {'num_steps': 294, 'success': True}, 'oracle': {'num_steps': 22, 'success': True}}, 13: {'dc2g': {'num_steps': 69, 'success': True}, 'dc2g_rescale': {'num_steps': 299, 'success': True}, 'frontier': {'num_steps': 204, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 14: {'dc2g': {'num_steps': 62, 'success': True}, 'dc2g_rescale': {'num_steps': 240, 'success': True}, 'frontier': {'num_steps': 117, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 15: {'dc2g': {'num_steps': 67, 'success': True}, 'dc2g_rescale': {'num_steps': 368, 'success': True}, 'frontier': {'num_steps': 229, 'success': True}, 'oracle': {'num_steps': 41, 'success': True}}, 16: {'dc2g': {'num_steps': 58, 'success': True}, 'dc2g_rescale': {'num_steps': 284, 'success': True}, 'frontier': {'num_steps': 197, 'success': True}, 'oracle': {'num_steps': 29, 'success': True}}, 17: {'dc2g': {'num_steps': 61, 'success': True}, 'dc2g_rescale': {'num_steps': 383, 'success': True}, 'frontier': {'num_steps': 272, 'success': True}, 'oracle': {'num_steps': 40, 'success': True}}, 18: {'dc2g': {'num_steps': 50, 'success': True}, 'dc2g_rescale': {'num_steps': 484, 'success': True}, 'frontier': {'num_steps': 143, 'success': True}, 'oracle': {'num_steps': 35, 'success': True}}, 19: {'dc2g': {'num_steps': 52, 'success': True}, 'dc2g_rescale': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 231, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 20: {'dc2g': {'num_steps': 69, 'success': True}, 'dc2g_rescale': {'num_steps': 89, 'success': True}, 'frontier': {'num_steps': 312, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 21: {'dc2g': {'num_steps': 80, 'success': True}, 'dc2g_rescale': {'num_steps': 202, 'success': True}, 'frontier': {'num_steps': 271, 'success': True}, 'oracle': {'num_steps': 53, 'success': True}}, 22: {'dc2g': {'num_steps': 61, 'success': True}, 'dc2g_rescale': {'num_steps': 383, 'success': True}, 'frontier': {'num_steps': 272, 'success': True}, 'oracle': {'num_steps': 40, 'success': True}}, 23: {'dc2g': {'num_steps': 68, 'success': True}, 'dc2g_rescale': {'num_steps': 356, 'success': True}, 'frontier': {'num_steps': 241, 'success': True}, 'oracle': {'num_steps': 33, 'success': True}}, 24: {'dc2g': {'num_steps': 64, 'success': True}, 'dc2g_rescale': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 299, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 25: {'dc2g': {'num_steps': 31, 'success': True}, 'dc2g_rescale': {'num_steps': 40, 'success': True}, 'frontier': {'num_steps': 154, 'success': True}, 'oracle': {'num_steps': 26, 'success': True}}, 26: {'dc2g': {'num_steps': 19, 'success': True}, 'dc2g_rescale': {'num_steps': 9, 'success': True}, 'frontier': {'num_steps': 330, 'success': True}, 'oracle': {'num_steps': 9, 'success': True}}, 27: {'dc2g': {'num_steps': 77, 'success': True}, 'dc2g_rescale': {'num_steps': 199, 'success': True}, 'frontier': {'num_steps': 256, 'success': True}, 'oracle': {'num_steps': 42, 'success': True}}, 28: {'dc2g': {'num_steps': 62, 'success': True}, 'dc2g_rescale': {'num_steps': 376, 'success': True}, 'frontier': {'num_steps': 281, 'success': True}, 'oracle': {'num_steps': 41, 'success': True}}, 29: {'dc2g': {'num_steps': 76, 'success': True}, 'dc2g_rescale': {'num_steps': 176, 'success': True}, 'frontier': {'num_steps': 153, 'success': True}, 'oracle': {'num_steps': 43, 'success': True}}}
    # big FOV
    # results = {0: {'dc2g': {'num_steps': 43, 'success': True}, 'dc2g_rescale': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 138, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 1: {'dc2g': {'num_steps': 29, 'success': True}, 'dc2g_rescale': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 133, 'success': True}, 'oracle': {'num_steps': 25, 'success': True}}, 2: {'dc2g': {'num_steps': 28, 'success': True}, 'dc2g_rescale': {'num_steps': 32, 'success': True}, 'frontier': {'num_steps': 148, 'success': True}, 'oracle': {'num_steps': 26, 'success': True}}, 3: {'dc2g': {'num_steps': 38, 'success': True}, 'dc2g_rescale': {'num_steps': 44, 'success': True}, 'frontier': {'num_steps': 78, 'success': True}, 'oracle': {'num_steps': 38, 'success': True}}, 4: {'dc2g': {'num_steps': 42, 'success': True}, 'dc2g_rescale': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 74, 'success': True}, 'oracle': {'num_steps': 42, 'success': True}}, 5: {'dc2g': {'num_steps': 47, 'success': True}, 'dc2g_rescale': {'num_steps': 35, 'success': True}, 'frontier': {'num_steps': 95, 'success': True}, 'oracle': {'num_steps': 35, 'success': True}}, 6: {'dc2g': {'num_steps': 58, 'success': True}, 'dc2g_rescale': {'num_steps': 41, 'success': True}, 'frontier': {'num_steps': 132, 'success': True}, 'oracle': {'num_steps': 34, 'success': True}}, 7: {'dc2g': {'num_steps': 68, 'success': True}, 'dc2g_rescale': {'num_steps': 60, 'success': True}, 'frontier': {'num_steps': 77, 'success': True}, 'oracle': {'num_steps': 39, 'success': True}}, 8: {'dc2g': {'num_steps': 34, 'success': True}, 'dc2g_rescale': {'num_steps': 70, 'success': True}, 'frontier': {'num_steps': 130, 'success': True}, 'oracle': {'num_steps': 34, 'success': True}}, 9: {'dc2g': {'num_steps': 46, 'success': True}, 'dc2g_rescale': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 114, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 10: {'dc2g': {'num_steps': 26, 'success': True}, 'dc2g_rescale': {'num_steps': 40, 'success': True}, 'frontier': {'num_steps': 48, 'success': True}, 'oracle': {'num_steps': 24, 'success': True}}, 11: {'dc2g': {'num_steps': 31, 'success': True}, 'dc2g_rescale': {'num_steps': 49, 'success': True}, 'frontier': {'num_steps': 115, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 12: {'dc2g': {'num_steps': 25, 'success': True}, 'dc2g_rescale': {'num_steps': 38, 'success': True}, 'frontier': {'num_steps': 110, 'success': True}, 'oracle': {'num_steps': 22, 'success': True}}, 13: {'dc2g': {'num_steps': 42, 'success': True}, 'dc2g_rescale': {'num_steps': 42, 'success': True}, 'frontier': {'num_steps': 140, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 14: {'dc2g': {'num_steps': 41, 'success': True}, 'dc2g_rescale': {'num_steps': 38, 'success': True}, 'frontier': {'num_steps': 91, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 15: {'dc2g': {'num_steps': 70, 'success': True}, 'dc2g_rescale': {'num_steps': 64, 'success': True}, 'frontier': {'num_steps': 89, 'success': True}, 'oracle': {'num_steps': 41, 'success': True}}, 16: {'dc2g': {'num_steps': 43, 'success': True}, 'dc2g_rescale': {'num_steps': 37, 'success': True}, 'frontier': {'num_steps': 53, 'success': True}, 'oracle': {'num_steps': 29, 'success': True}}, 17: {'dc2g': {'num_steps': 42, 'success': True}, 'dc2g_rescale': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 122, 'success': True}, 'oracle': {'num_steps': 40, 'success': True}}, 18: {'dc2g': {'num_steps': 53, 'success': True}, 'dc2g_rescale': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 63, 'success': True}, 'oracle': {'num_steps': 35, 'success': True}}, 19: {'dc2g': {'num_steps': 31, 'success': True}, 'dc2g_rescale': {'num_steps': 33, 'success': True}, 'frontier': {'num_steps': 121, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 20: {'dc2g': {'num_steps': 38, 'success': True}, 'dc2g_rescale': {'num_steps': 50, 'success': True}, 'frontier': {'num_steps': 114, 'success': True}, 'oracle': {'num_steps': 36, 'success': True}}, 21: {'dc2g': {'num_steps': 93, 'success': True}, 'dc2g_rescale': {'num_steps': 81, 'success': True}, 'frontier': {'num_steps': 65, 'success': True}, 'oracle': {'num_steps': 53, 'success': True}}, 22: {'dc2g': {'num_steps': 42, 'success': True}, 'dc2g_rescale': {'num_steps': 75, 'success': True}, 'frontier': {'num_steps': 122, 'success': True}, 'oracle': {'num_steps': 40, 'success': True}}, 23: {'dc2g': {'num_steps': 42, 'success': True}, 'dc2g_rescale': {'num_steps': 59, 'success': True}, 'frontier': {'num_steps': 139, 'success': True}, 'oracle': {'num_steps': 33, 'success': True}}, 24: {'dc2g': {'num_steps': 50, 'success': True}, 'dc2g_rescale': {'num_steps': 35, 'success': True}, 'frontier': {'num_steps': 125, 'success': True}, 'oracle': {'num_steps': 31, 'success': True}}, 25: {'dc2g': {'num_steps': 29, 'success': True}, 'dc2g_rescale': {'num_steps': 30, 'success': True}, 'frontier': {'num_steps': 50, 'success': True}, 'oracle': {'num_steps': 26, 'success': True}}, 26: {'dc2g': {'num_steps': 9, 'success': True}, 'dc2g_rescale': {'num_steps': 9, 'success': True}, 'frontier': {'num_steps': 9, 'success': True}, 'oracle': {'num_steps': 9, 'success': True}}, 27: {'dc2g': {'num_steps': 72, 'success': True}, 'dc2g_rescale': {'num_steps': 58, 'success': True}, 'frontier': {'num_steps': 58, 'success': True}, 'oracle': {'num_steps': 42, 'success': True}}, 28: {'dc2g': {'num_steps': 60, 'success': True}, 'dc2g_rescale': {'num_steps': 71, 'success': True}, 'frontier': {'num_steps': 95, 'success': True}, 'oracle': {'num_steps': 41, 'success': True}}, 29: {'dc2g': {'num_steps': 45, 'success': True}, 'dc2g_rescale': {'num_steps': 45, 'success': True}, 'frontier': {'num_steps': 85, 'success': True}, 'oracle': {'num_steps': 43, 'success': True}}}
    # plot(results)
