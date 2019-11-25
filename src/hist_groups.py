#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ipaddress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging

from collections import Counter
from hist_io import IO
from hist_io import iocfg

def color_assign(value):
    if value == 1:
        return 'red'
    if value < 5:
        return 'yellow'
    if value < 20:
        return 'green'
    if value < 100:
        return 'blue'
    return 'black'


def groupby_ops(data, args):
    c = Counter(data[args.category])
    data = data.groupby(args.category)[args.field].median().to_frame()
    data = data.reset_index()
    
    res = len(data.index) * [None]
    res_c = len(data.index) * [None]
    col = len(data.index) * [None]
    for i in range(len(data.index)):
        res[i] = int(ipaddress.IPv4Address(data[args.category][i]))
        res_c[i] = c[data[args.category][i]]
        col = color_assign(res_c[i])

    data = data.join(pd.DataFrame({'IP': res, 'Count': res_c}))

    IO.plot(iocfg(x=data['IP'], y=data[args.field], markersize=1.5, color=col, grid=True, figsize=(13,13)))
    IO.write(data.sort_values(by=[args.field], ascending=False), args, "_groupby.csv")



def get_groups_size(data, args):
    return data.groupby([args.field]).size().reset_index(name='counts').sort_values(by='counts', ascending=False).reset_index(drop=True)



def counter_req(data, args):
    c = Counter(data[args.field])
    with open("".join([args.output, "_counter_", args.field, ".txt"]), 'w') as f:
        for key in c.keys():
            f.write(str(key) + " " + str(c[key]) + '\n')

    
class IP:
    def convert(data, args):
        for i in range(len(data)):
            try:
                data['IP'][i] = int(ipaddress.IPv4Address(data['IP'][i]))
            except ipaddress.AddressValueError:
                data['IP'][i] = None
        return data

    def dump(data, args):
        with open("".join([args.output, "_ip.txt"]), 'w') as f:
            for i in range(len(data)):
                try:
                    data['IP'][i] = int(ipaddress.IPv4Address(data['IP'][i]))
                except ipaddress.AddressValueError:
                    continue
                f.write("".join([str(data['IP'][i]), " ", str(data['Count'][i]), "\n"]))
        return data


def ips_distribution(data, args):
    default_ip_field_name = "clientip"
    if args.verbose:
        print(data)

    df = pd.DataFrame(list(Counter(data[default_ip_field_name]).items()),
            columns=['IP', 'Count']).sort_values(by='Count', ascending=False, na_position='last')

    IO.write(df.dropna(axis='rows'), args, "_ips.csv")
    IO.write(pd.DataFrame({default_ip_field_name: df['IP'].dropna()}), args, "_ipflist.csv")

    logging.debug("Df size is %d items.", len(df.index))
    if args.verbose:
        print(df)
    df = IP.convert(df, args).dropna(axis='rows')

    IO.plot(iocfg(x=df['IP'], y=df['Count'], markersize=0.1, color='b', grid=True, figsize=(20,9),
        title="IPs"))

    IO.savefig(args, "_ips.eps")


'''
Split data to k samples. Usually k = args.state['pipeline']['processCnt'].
Return sample #i.
'''
def ds_split(data, k, i):
    ls = np.linspace(0, len(data.index), k + 1, dtype = int)
    return data.iloc[ls[i]:ls[i+1]]
