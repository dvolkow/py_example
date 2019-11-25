#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import csv
import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd

from clickhouse_driver import Client
from collections import namedtuple
from prettytable import PrettyTable

import numpy as np

def csvreader_times(args, i):
    return pd.read_csv(args.infile[i], parse_dates = [args.times])

def csvreader(fname):
    return pd.read_csv(fname)

def pickreader(args, i):
    return pd.read_pickle(args.infile[i])


'''
Sample:
    SELECT
        date,
        clientip,
        request_length,
        request_time
    FROM nginx_access_distr
    WHERE date BETWEEN '2019-11-20 10:30:00' AND '2019-11-20 12:30:00'
    AND type = 'nginx_access' AND match(clientip, '^10?[0-9]*')
    ORDER BY date
'''
def clickread(args):
    query, fields_list = gen_request(args)
    if args.verbose:
        print(query)
    return pd.DataFrame(Client(args.state['io']['clickIP']).execute(query),
            columns = fields_list)


def gen_request(args):
    if not args.tb:
        args.tb = args.state['DEFAULT']['tb'].split(",")

    fields_list = args.state['qrequest']['fields'].split(":")
    fields = ", ".join(fields_list)
    req = "SELECT {f} FROM nginx_access_distr WHERE date BETWEEN '{d} {t1}' AND '{d} {t2}' AND type = 'nginx_access' ORDER BY date".format(f = fields,
            d = args.state['qrequest']['date'],
            t1 = args.tb[0],
            t2 = args.tb[1])
    return req, fields_list



'''
Attention! 
See about another DataFrame hadling and keeping:
* HDF: https://habr.com/ru/company/otus/blog/416309/

Elasticsearch more preferable -- instead of read csv
will requests to index
'''

fields = ['x', 'y', 'title', 'markersize', 'color', 'grid', 'figsize',
        'label', 'addition_x', 'addition_y', 'compared']
iocfg = namedtuple('iocfg', fields,
        defaults=(None,) * len(fields))

class IO:
    def single_read(args, i):
        if "csv" in args.infile[i]:
            if args.times != "no":
                data = csvreader_times(args, i)
            else:
                data = csvreader(args.infile[i])
        elif "pickle" in args.infile[i]:
            data = pickreader(args, i)
        elif "click" in args.infile[i]:
            data = clickread(args) 
        else:
            logging.error("Unknown input file format. Exit")
            sys.exit(1)
        return data

    def read(args):
        data = pd.DataFrame(columns = IO.single_read(args, 0).columns)
        for i in range(len(args.infile)):
            data = data.append(IO.single_read(args, i))

        return data

    def read_flist(fname):
        return csvreader(fname)

    def dump(data, args):
        if args.topickle:
            data.to_pickle("".join([args.output, ".pickle"]))
            logging.debug("Success dump DataFrame to %s.pickle", args.output)


    def write(data, args, postfix):
        if args.state['io']['csvOut'] == 'yes':
            data.to_csv("".join([args.output, postfix]), header = True)


    def set_fig(figsize):
        plt.figure(figsize=cfg.figsize)

    def plot_ts(cfg):
        plt.figure(figsize=cfg.figsize)
        plt.plot(cfg.x)
        plt.grid(cfg.grid)
        plt.title(cfg.title)

    def plot_line(cfg):
        plt.figure(figsize=cfg.figsize)
        plt.plot(cfg.x, cfg.y)
        if cfg.addition_x is not None:
            plt.plot(cfg.addition_x, cfg.addition_y, "x")
        plt.title(cfg.title)

    def plot_line_err(cfg):
        plt.figure(figsize=cfg.figsize)
        plt.title(cfg.title)
        plt.plot(cfg.x, cfg.y, color='blue')
        plt.plot(cfg.x, cfg.y + cfg.addition_y, color='red')
        plt.plot(cfg.x, cfg.y - cfg.addition_y, color='red')
        plt.grid(cfg.grid)
        if cfg.compared is not None:
            plt.plot(cfg.x, cfg.compared, color = 'black')

    def plot(cfg):
        plt.figure(figsize=cfg.figsize)
        plt.plot(cfg.x, cfg.y, '*', markersize=cfg.markersize, color=cfg.color)
        plt.grid(cfg.grid)

    def savefig(args, postfix):
        plt.savefig("".join([args.output, postfix]), format='eps')


    def plot_2ts(cfg):
        plt.figure(figsize=cfg.figsize)
        plt.grid(cfg.grid)

        for i in range(len(cfg.x)):
            plt.plot(cfg.x[i][cfg.label[0]], cfg.x[i][cfg.label[1]], color=cfg.color[i])

        plt.show()

    def plot_general_ts(cfg, args):
        plt.figure(figsize=cfg.figsize)
        plt.grid(cfg.grid)
        plt.title(cfg.title)

        for i in range(len(cfg.x)):
            if cfg.x[i] is not None:
                plt.plot(cfg.x[i], color=cfg.color[i])




    # Time
    class Time:
        '''
        @f_t: fourier_t 
        '''
        def plot_fourier_spectre(f_t, args):
            if args.state['io']['epsOut'] == 'yes':
                if args.state['time']['fftNoZero'] == 'yes':
                    IO.plot_line(iocfg(x=f_t.f[1:], y=f_t.ps[1:], figsize=(20,9),
                        title="Power spectrum by FFT (without zero)"))
                    IO.savefig(args, "_fft_nozero.eps")
        
        
                IO.plot_line(iocfg(x=f_t.f, y=f_t.ps, figsize=(20,9),
                    title="Power spectrum by FFT (with zero)",
                    addition_x = f_t.f[f_t.peaks], addition_y = f_t.ps[f_t.peaks]))
                IO.savefig(args, "_fft.eps")
                if not args.noshow:
                    plt.show()
            
            if args.state['io']['csvOut'] == 'yes':
                IO.write(pd.DataFrame({"freq": f_t.f, "power": f_t.ps}), args, "_fft.csv")
    
    
        def plot_timesquash_count(data, args):
            if args.state['io']['epsOut'] == 'yes':
                IO.plot_ts(iocfg(x=data, figsize=(20,9), grid=True, title="Time Series"))
                IO.savefig(args, "_series.eps")
            if args.state['io']['csvOut'] == 'yes':
                IO.write(data, args, "_series.csv")
            if not args.noshow:
                plt.show()

        def write_simple_txt(data, args):
            if args.state['io']['csvOut'] == 'yes':
                data.to_csv(args.output, sep = ' ', header = True, index = False)

        def write_pretty_txt(data, args):
            table = PrettyTable()
            table.field_names = data.columns
            for i in range(len(data.index)):
                table.add_row(data.iloc[i])
            if args.verbose:
                print(table)
            with open(args.output, 'w') as f:
                f.write(table.get_string())
                f.write('\n')




class Hist:
    def dump_hist(n, bins, args):
        with open("{}.dat".format(args.output), 'w') as f:
            for i in range(len(n)):
                f.write(str(bins[i]) + " " + str(n[i]) + '\n')
    
    
    def plot_hist(data, args):
        f = Filter(args)
        x = f.cut(data)
        n, bins, patches = plt.hist(x, args.bins, facecolor='blue', alpha=0.5)
        if args.dump:
            self.dump_hist(n, bins, args)
    
        plt.title("Histogram for {}".format(args.field))
        plt.savefig("{}.pdf".format(args.output))


def print_stat(data, args):
    x = Filter(args).cut(data)
    table = PrettyTable()
    table.field_names = ["Statistic", "Value"]
    table.align["Value"] = "r"
    table.add_row(["Size", len(x)])
    table.add_row(["Min", float_format(np.min(x))])
    table.add_row(["Max", float_format(np.max(x))])
    table.add_row(["Mean", float_format(np.mean(x))])
    table.add_row(["Median", float_format(np.median(x))])
    table.add_row(["Var", float_format(np.var(x))])
    print("Result for parameter '{}'".format(args.field))
    print(table)


def hist_calls(data, args):
    Hist.plot_hist(data, args)
    print_stat(data, args)


def float_format(value):
    return float("{0:.2}".format(value))

def how_much_req(data, args):
    data = Filter(args).cut(data)
    table = PrettyTable()
    table.field_names = ["Desc", "Value", "Percentage"]
    table.align["Value"] = "r"
    table.add_row(["Total size", len(data), 100])
    table.add_row(["Subset size", len(x), float_format(float(len(x)) / len(data) * 100)])
    table.add_row(["Addition size", len(data) - len(x), float_format(float(len(data) - len(x)) / len(data) * 100)])
    print(table)


def math_calls(data, args):
    how_much_req(data, args)
    return


'''
Usually intersection is empty.
'''
def show_intersection_ips(data):
    clients = set(data['clientip']) 
    hosts = set(data['host'])
    print(clients.intersection(hosts))



