#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import configparser
import logging
import numpy as np
import sys

from pathlib import Path
# TODO immutable configuration (maybe partial): from collections import namedtuple

#my modules:
from hist_time import time_calls
from hist_time import finding_spectral_anomaly
from hist_time import compare_2ts
from hist_time import average_freq_interval
from hist_groups import counter_req
from hist_groups import ips_distribution
from hist_groups import groupby_ops
from hist_io import hist_calls
from hist_io import math_calls
from hist_entropy import entropy_distance, entropy_reduce
from plotter import generic_plot


def path_to_samples(args):
    if args.state['io']['usePrefix'] == 'yes':
        return args.state['io']['samplesPath']
    return ""



class Action:
    def __init__(self):
        # type: [handling function, reduce function]
        self.__functions_map = {
            "hist": [hist_calls, self._default_reduce],
            "math": [math_calls, self._default_reduce],
            "groups": [counter_req, self._default_reduce],
            "time": [time_calls, self._default_reduce],
            "ips": [ips_distribution, self._default_reduce],
            "empty": [self.empty_call, self._default_reduce],
            "gby": [groupby_ops, self._default_reduce],
            "compare_2ts": [compare_2ts, self._default_reduce],
            "find": [finding_spectral_anomaly, self._default_reduce],
            "aver": [average_freq_interval, self._default_reduce],
            "entropy": [entropy_distance, entropy_reduce],
            "plot": [generic_plot, self._default_reduce]
            }

    def __call__(self, data, args):
        if args.type in self.__functions_map:
            return self.__functions_map[args.type][0](data, args)
        else:
            logging.error("Type %s is not available. Exit", args.type)
            sys.exit(1)

    # This function is not return any value! Make all IO into reduce logic
    def reduce(self, data, args):
        self.__functions_map[args.type][1](data, args) 
        
    def get_actions(self):
        return self.__functions_map

    def _default_reduce(self, data, args):
        return data

    def empty_call(self, data, args):
        print(data)
        return


class Statement:
    __config = configparser.ConfigParser()
    __args = None
    __path = '/home/dvolkow/wd/l7_attacks/data/src/hist.ini'


    def setup_parser(self, fmap):
        parser = argparse.ArgumentParser(
                description="Histograms, test tool. Path to configuration file: {}".format(self.__path)
        )
    
        parser.add_argument(
            "-i", "--input", nargs='+', default = [],
            dest="infile", help="Input file (with header if CSV). Set it as 'click' if you want get data from ClickHouse. Also, see hist.ini to more options for IO"
        )
    
        parser.add_argument(
            "-f", "--field",
            default=self.__config['DEFAULT']['field'],
            dest="field", help="Field for plotting histogram, and statistics for groupby ops"
        )
    
        parser.add_argument(
            "-t", "--type",  
            default=self.__config['DEFAULT']['type'],
            dest="type", help="Type of operation: {}".format(fmap.keys())
        )
    
        parser.add_argument(
            "-s", "--resampling",
            default=self.__config['DEFAULT']['resampling'],
            dest="resampling", help="Type of resampling of time series. Default is 'counter'. Available: 'mean', 'median'"
        )
    
        parser.add_argument(
            "-o", "--output", default="", 
            dest="output", help="Output files prefix"
        )
    
        parser.add_argument(
            "-r", "--round",
            default=self.__config['DEFAULT']['rounding'],
            dest="round", help="Rounding for time data, '1s' for example to round by 1 second. Disabled by default"
        )
    
        parser.add_argument(
            "-v", "--verbose", dest="verbose", action="store_true", help="Increases logging"
        )
    
        parser.add_argument(
            "--time-between", nargs='+', default = [],
            dest="tb", help="Cutting for data by times."
        )
    
        parser.add_argument(
            "--category",
            default=self.__config['DEFAULT']['category'],
            dest="category", help="Category for groupby ops"
        )
    
        parser.add_argument(
            "--times",
            default=self.__config['DEFAULT']['times'],
            dest="times", help="Name for dataset's timestamps. If your data have not timestamps, set it as 'no'"
        )
    
        parser.add_argument(
            "--dump", action="store_true",
            dest="dump", help="Dump hist info to *.dat file"
        )
    
        parser.add_argument(
            "--to-pickle", action="store_true",
            dest="topickle", help="Pickle DataFrame files"
        )
    
        parser.add_argument(
            "--noshow", action="store_true",
            dest="noshow", help="No show the plotted dias, only save to *.eps file"
        )
    
        parser.add_argument(
            "--normalization", dest="normalization", action="store_true", help="Normalize compared dataframes"
        )
    
        parser.add_argument(
            "--great", default=np.NAN, type=float,
            dest="great", help="Cutting for data by upper bound, float"
        )
       
        parser.add_argument(
            "--less", default=np.NAN, type=float,
            dest="less", help="Cutting for data by low bound, float"
        )
    
        parser.add_argument(
            "--filter", nargs='*', default=[], type=str,
            dest="filter", help="Filter your data by values. Example: 'parameter:value'"
        )
    
        parser.add_argument(
            "--filter-list", default="", type=str,
            dest="filterlist", help="Filter list values into filename as parameter."
        )
    
        parser.add_argument(
            "--bins", type=int,
            default=self.__config['DEFAULT']['bins'],
            dest="bins", help="Count of bins for hist, integer"
        )
    
        parser.add_argument(
            "--config-no-use", default=None,
            dest="state", help=""
        )
    
        args = parser.parse_args()
        if len(sys.argv) <= 1:
            parser.print_help()
            sys.exit(1)
            
        logging.basicConfig(
            format="[%(asctime)s] %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
            filename=self.__config['logging']['fileName']
        )
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        args.state = self.__config
        if not args.noshow and args.state['io']['showPick'] == 'no':
            args.noshow = False

        return args


    def __init__(self, fmap):
        # TODO: find ini file
        self.__config.read(self.__path)
        self.__args = self.setup_parser(fmap)
        self.check_args()

    def get_args(self):
        return (self.__args)

    def get_config(self):
        return (self.__config)

    def check_args(self):
        if "click" not in self.__args.infile:
            for i in range(len(self.__args.infile)):
                self.__args.infile[i] = "".join([path_to_samples(self.__args), self.__args.infile[i]])
    
            for i in range(len(self.__args.infile)):
                f = Path(self.__args.infile[i])
                if f.is_file():
                    continue
                else:
                    logging.error("No such file: %s. Exit.", self.__args.infile[i])
                    sys.exit(1)
            
            if len(self.__args.output) == 0:
                self.__args.output = self.__args.infile[0].split(".")[-2] # no extending
                self.__args.output = self.__args.output.split("/")[-1] # no path
                logging.debug("Output file set as %s", self.__args.output)



