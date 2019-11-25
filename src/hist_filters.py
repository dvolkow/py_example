#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys

import re

# my modules:
from hist_io import IO
from hist_setup import Statement
from hist_setup import Action
from hist_groups import get_groups_size


def filter_data_by_time(data, args):
    data = data.set_index(args.times)
    data[args.times] = data.index
    
    if args.tb:
        data = data.between_time(args.tb[0], args.tb[1])
        assert not data.empty, 'No data for selected period! Exit'

    logging.debug("Time series have %s items. Continue", len(data.index))
    return data


def filter_data_by_values(data, args):
    if args.filter:
        args.filter = dict([arg.split(':') for arg in args.filter])
        for key in args.filter.keys():
            data = data[data[key] == args.filter[key]]
            if data.empty:
                logging.error("Selected data is empty (%s = %s). Exit", key, args.filter[key])
                sys.exit(1)
            else:
                logging.debug("Selected data have %d items (%s = %s). Continue", len(data.index),
                        key, args.filter[key])

    return data


def filter_data_by_list(data, args):
    if args.filterlist:
        _data = IO.read_flist(args.filterlist).drop(columns=['Unnamed: 0']).dropna()
        key = _data.columns[0]
        minListLen = int(Statement(Action().get_actions()).get_config()['filter']['minListLen'])
        logging.debug("filter_data_by_list: have %d|%d length list, key %s",
                len(_data.index), minListLen, key)

        return data[data[key].isin(_data[key][:min(minListLen, len(_data.index))])]
    return data



def filter_ip_by_regex(data, args):
    field = args.field
    args.field = args.state['time']['ipField']

    iplist = get_groups_size(data, args)

    # filter for local IPs:
    ipregex = str(args.state['filter']['localIPRegex'])
    term = re.compile(ipregex)
    filtered = [i for i in iplist[args.field] if not term.match(i)]
    iplist = iplist[iplist[args.field].isin(filtered)]
    assert not iplist.empty, 'filter_ip_by_regex fault'

    res = data[data[args.field].isin(iplist[args.field])]
    args.field = field

    return res

    



class Filter:
    __args = None

    def __init__(self, args):
        self.__args = args

    def __call__(self, data):
        data = filter_data_by_values(data, self.__args)
        data = filter_data_by_list(data, self.__args)
        if self.__args.times != 'no':
            data = filter_data_by_time(data, self.__args)
        if self.__args.state['filter']['ipFilterEnabled'] != 'no':
            data = filter_ip_by_regex(data, self.__args)
        return data

    def cut(self, data):
        x = [float(item) for item in data[self.__args.field] if not np.isnan(item)]
        if not np.isnan(self.__args.less):
            x = list(filter(lambda k: k <= self.__args.less, x))
        if not np.isnan(args.great):
            x = list(filter(lambda k: k >= self.__args.great, x))
    
        return x


