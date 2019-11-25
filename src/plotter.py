#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
import pandas as pd

#my modules:
from hist_io import iocfg
from hist_io import IO

class Plotter:
    def __init__(self, args):
        self._args = args

    def plot_entropy_distance(self, ds, dist):
        self._args.output = "{}_".format(ds.index[0].time())
        etype = self._args.state['math']['entropyType']

        if self._args.state['io']['epsOut'] == 'yes':
            col = ['blue', 'red']
            IO.plot_general_ts(iocfg(x = [ds, dist],
                grid = True,
                title = "{} entropy for {}".format(etype, self._args.field),
                figsize = (20, 9),
                color = col), 
                self._args)
        
            IO.savefig(self._args, "{}_entropy.eps".format(etype))

        IO.write(ds, self._args, "{}_entropy.csv".format(etype))
        IO.write(dist, self._args, "{}_entropy_dist.csv".format(etype))




'''
@data: [ds1, ds2 ... dsN]

Return: ds1 + ... + dsN
'''
def cat_datasets(data):
    res = pd.DataFrame(columns = data[0].columns)
    for i in range(len(data)):
        res = res.append(data[i])

    return res

'''
@data: [ds1, ds2 ... dsN]

Cat ds[1 ... N] -> ds, and apply any plotting ops. 
'''
def generic_plot(data, args):
    cdata = data.set_index(args.times)
    Plotter(args).plot_entropy_distance(cdata, None)
    return 


