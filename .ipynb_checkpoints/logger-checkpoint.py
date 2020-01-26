import time
import pickle
import joblib
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from statistics import mean, stdev


EXP_BASE_PATH = "/tmp/experiments/"


def logger_exp_setup(exp_name):
    date = time.strftime('%Y-%m-%d')
    output_dir = os.path.join('.', date, exp_name)

    if os.path.exists(output_dir):
        print('{} already exists. Storing new data in this location.'.format(output_dir))
    else:
        os.makedirs(output_dir)

    return output_dir


class Logger:
    """
    Maintains record and calculates statistics of experment. Dumps data into pickle file.

    """
    def __init__(self, output_dir=None, seed=None, keep_raw=True):
        """
        Instantiate Logger object.

        ::args::
            output_dir: directory to store experiment data. if None, defaults to EXP_BASE_PATH
            seed: seed for experiment
            keep_raw: if True, pickle all data collected. if False, only store calculated statistics
            alg: name of algorithm used
        """
        output_dir = output_dir or EXP_BASE_PATH + time.strftime('%Y-%m-%d')
        self.output_dir = os.path.join(output_dir, 'seed={}'.format(seed))

        # create directory path if it does not exist
        if os.path.exists(self.output_dir):
            print('{} already exists. Storing new data in this location.'.format(self.output_dir))
        else:
            os.makedirs(self.output_dir)

        self.log_headers = list()
        self.log_data = dict()

        self.keep_raw = keep_raw


    def save_config(self, config):
        """
        Saves all hyperparameters for current experiment.

        ::args::
            config: hyperparameters used in current experiment, passed by calling locals() in save_config call
        """
        self.run_config = str(config)
        self.output_file = os.path.join(self.output_dir, self.run_config) + '.pkl'

        cfg = {'config': config}
        with open(self.output_file, 'wb') as out_file:
            pickle.dump(cfg, out_file, protocol=pickle.HIGHEST_PROTOCOL)
        

    def log(self, data):
        """
        Record data from current iteration under key in self.log_data.

        ::args::
            data: dictionary where keys are headers, values are data to add under given header
        """
        for key, value in data.items():
            # add header to log if not present
            if key not in self.log_headers:
                self.log_headers.append(key)
                self.log_data[key] = []

            self.log_data[key].append(value)

    
    def dump(self, get_stats={}):
        """
        Dump logged data to pickle file defined by self.output_file.

        ::args::
            get_stats: dictionary -> keys are header names, values are lists of stats to calculate for given header
        """
        logged_data = self.get_data()

        # if first dump: key 'raw' gets added with dictionary as value where keys are logged headers and values are lists 
        #                key 'stats' gets added with dictionary as value where keys are headers to have statistics calculated 
        #                   and the values are dictionaries where the keys will be the stat name (eg. mean, stdev, ...) and 
        #                   values are lists
        if 'raw' not in logged_data.keys():
            logged_data.update(raw={header: [] for header in self.log_headers},
                               stats={header: {} for header in get_stats})

        # iterate over headers to extract data stored from previous iteration
        for key in self.log_headers:
            # if current header has stats to be calculated
            if key in get_stats:
                stats = get_stats[key]

                # iterates over and executes all stats for header, adding calculation to st_dict
                for st in stats:
                    if st not in logged_data['stats'][key]:
                        logged_data['stats'][key][str(st)] = []

                    if isinstance(st, str):
                        logged_data['stats'][key][st].append(eval(st)(self.log_data[key]))
                    else:           # allow for passing of lambda function
                        logged_data['stats'][key][str(st)].append(st(self.log_data[key]))

            if self.keep_raw:
                logged_data['raw'][key].append(self.log_data[key])

        with open(self.output_file, 'wb') as out_file:
            pickle.dump(logged_data, out_file, protocol=pickle.HIGHEST_PROTOCOL)

        self.log_data = {header:[] for header in self.log_headers}


    def make_plots(self):
        stats = self.get_data()['stats']
        figs = []

        # return stats
        return_stats = stats['return']

        fig0, ax0 = plt.subplots(2, 1, figsize=(10,8))

        ax0[0].set_xlabel('Episode')
        ax0[0].set_ylabel('Returns')

        mean = np.array(return_stats['mean'])
        std = np.array(return_stats['stdev'])

        ax0[0].plot(mean, label='Average Returns')
        ax0[0].fill_between(mean-std, 
                            mean+std, 
                            facecolor='blue', 
                            alpha=0.1)
        ax0[0].legend()

        ax0[1].plot(return_stats['min'], label='Min Reward')
        ax0[1].plot(return_stats['max'], label='Max Reward')
        ax0[1].legend()

        # ax0.set_title('Return Stats')
        fig0.tight_layout() 

        figs.append(fig0)

        # ...

        plots_dir = self.output_dir + '/' + self.run_config + '_plots'

        if os.path.exists(plots_dir):
            print('{} already exists. Storing new data in this location.'.format(plots_dir))
        else:
            os.makedirs(plots_dir)

        fig0.savefig(plots_dir + '/return_stats.png')


    def get_data(self, open_as='rb'):
        if os.path.exists(self.output_file):
            with open(self.output_file, open_as) as out_file: 
                logged_data = pickle.load(out_file)

            return logged_data
        else:
            print('File {} never created'.format(self.output_file))
            exit(0)