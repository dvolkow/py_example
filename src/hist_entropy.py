#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import ipaddress
import math
import logging
import numpy as np
import pandas as pd
import progressbar
import scipy.stats as stats

from hist_groups import get_groups_size
from hist_time import date_range_from_sample
from hist_io import IO
from hist_io import iocfg
from plotter import Plotter

class Entropy:
    def harley_entropy(self, p):
        s = 0
        for i in p:
            s += i ** self._alpha
    
        if s != 0:
            return 1. / (1 - self._alpha) * math.log(s, 2) 
    
        return None
    

    def __init__(self, args):
        self._map = {"sci": stats.entropy,
                "harley": self.harley_entropy}

        self._alpha = int(args.state['math']['entropyDeg'])
        self._type = args.state['math']['entropyType']

    def __call__(self, seq):
        return self._map[self._type](seq)





def data_to_prob(counts):
    s = float(np.sum(counts))
    return counts / s



def packet_distribution(data, args):
    if args.verbose:
        print(date_range_from_sample(data, args))
    s = date_range_from_sample(data, args)
    res = pd.DataFrame(columns = [args.times, 'entropy'])

    bar = progressbar.ProgressBar(prefix = '(time {variables.time})',
            variables = {'time': '--'}).start()

    for i in range(len(s) - 1):
        bar.update(float(i)/(len(s) - 1) * 100,
                time = s[i])

        d = get_groups_size(data.between_time(s[i].time(), s[i + 1].time()), args)
        if d.empty:
            continue

        e = Entropy(args)(data_to_prob(d['counts']))

        if e is not None:
            res = res.append({args.times: s[i], 'entropy': e}, ignore_index = True)

    bar.finish()
    return res.set_index(args.times)
        


def dist(data):
    res = pd.DataFrame(columns = ['dist', 'time'])
    for i in range(1, len(data.index)):
        res = res.append({'dist': math.fabs(data['entropy'][i] - data['entropy'][i - 1]),
            'time': data.index[i]},
                ignore_index = True)
    return res.set_index('time')


'''
@data: Manager.dict 
'''
def entropy_reduce(data, args):
    ds = pd.Series()
    for i in range(len(data.keys())):
        ds = ds.append(data[i])

    Plotter(args).plot_entropy_distance(ds, dist(ds))

def entropy_distance(data, args):
    ds = packet_distribution(data, args)
    return ds
