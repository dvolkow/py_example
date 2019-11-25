#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import progressbar
import multiprocessing as mp

from collections import Counter
from collections import namedtuple
from sklearn import preprocessing
from scipy.signal import find_peaks

import dateutil.parser as dtpars

# my modules:
from hist_io import IO
from hist_io import iocfg
from hist_groups import get_groups_size
from hist_groups import ds_split

import re

def filter_ip_by_regex(iplist, regex, args):
    term = re.compile(regex)
    filtered = [i for i in iplist[args.field] if not term.match(i)]
    res = iplist[iplist[args.field].isin(filtered)].reset_index()
    assert not res.empty, 'filter_ip_by_regex fault'
    return res
    


fourier_t = namedtuple('fourier_t', 'f ps peaks')

'''
Discrete Fourier Transform: plot real values as power spectre.
@data: uniform time series
'''
def fourier_with_peaks(data, args):
    ps = np.abs(np.fft.rfft(data, n = int(args.state['DEFAULT']['fftInpLen'])))
    f = np.linspace(0, 1. / 2, len(ps))
    logging.debug('fourier_with_peaks: len(ps) = {}'.format(len(ps)))
    maxp = np.max(ps)
    ps /= maxp # normalization? 1 is max value for power
    meanp = np.median(ps)
    varp = np.sqrt(np.var(ps) / (len(ps) - 1))
    level = int(args.state['time']['peakLevel']) * meanp    
    peaks, _ = find_peaks(ps, height = level, 
            distance = int(args.state['time']['peakDistance'])
            )
    logging.debug("Maxp = %d, mean = %d, var = %d, level = %d", maxp, meanp, varp, level)

    return f, ps, peaks


def get_ip_list(data, args):
    source = str(args.state['time']['ipFromFile'])
    if source == 'yes':
        iplistfile = str(args.state['time']['ipListFile'])
        iplist = IO.read_flist(iplistfile).drop(columns=['Unnamed: 0']).dropna()
        key = iplist.columns[0]
    else:
        args.field = args.state['time']['ipField']
        iplist = get_groups_size(data, args)
        key = args.field

    ipLen = int(args.state['time']['ipLen'])

    # filter for local IPs:
    ipregex = str(args.state['filter']['localIPRegex'])
    iplist = filter_ip_by_regex(iplist, ipregex, args)

    return iplist, key, ipLen



prefix_p = namedtuple('prefix_p', 'ip start_time fin_time freq')

def prefix_ip_plot(prefix_arg):
    return "_".join([prefix_arg.ip,
        prefix_arg.start_time,
        prefix_arg.fin_time,
        prefix_arg.freq])


def average_freq_interval(ts, args):
    #tvalues = ts.index
    #delta = (tvalues[len(tvalues) - 1] - tvalues[0]).total_seconds()
    #print('delta: {}, len: {}'.format(delta, len(tvalues)))
    #return float(len(tvalues))/delta
    if args.state['time']['averagedType'] == 'mean':
        means = ts.mean(axis=1, skipna = True)
    elif args.state['time']['averagedType'] == 'median':
        means = ts.median(axis=1, skipna = True)
    else:
        logging.error("time: averagedType %s is not valid. Exit",
                args.state['time']['averagedType'])
        sys.exit(1)
    sd = ts.var(axis = 1, skipna = True)
    return means, sd


def get_len_resampled_subset(data, key, iplist, args):
    logging.debug('get_len_resampled_subset called with {} key, {}'.format(key, iplist.columns))
    logging.debug('iplist[key][0] = {}'.format(iplist[key][0]))
    subset = data[data[key] == iplist[key][0]]
    ts = resampling(subset, args)
    assert len(subset.index) > 1, 'get_len_resampled_subset failed.'
    args.state['DEFAULT']['fftInpLen'] = str(len(ts))
    f, ps, peaks = fourier_with_peaks(ts, args)

    return len(ps)


def counts_ip_filter(iplist, args):
    lb = str(args.state['time']['averagedFreqInt']).split(':')
    if args.verbose:
        print(iplist.columns)
        print(lb)
    if 'counts' in iplist.columns:
        if len(lb) > 1:
            low_b = int(lb[1])
            high_b = int(lb[0])
            iplist = iplist[iplist['counts'] >= high_b]
            iplist = iplist[iplist['counts'] <= low_b]
        iplist = iplist[iplist['counts'] >= int(args.state['time']['threshFreq'])]
    return iplist


def spectre_deviations(ps, means, sd):
    sd_i, ik = 0, 0
    
    n = min(len(ps), len(means))
    for i in range(n):
        #assert sd[i] != 0, 'sd[{}] = 0'.format(i) 
        if (sd[i] != 0 and not math.isnan(ps[i]) and not math.isnan(means[i])):
            sd_i += abs(ps[i] - means[i]) 
        else:
            ik += 1

    assert n != ik, 'Bad spectre_deviations result'
    return sd_i


def find_spectre_deviation(ts, means, sd, args):
    thr = float(args.state['time']['SDValue'])
    res = pd.DataFrame(columns = ['ip', 'ssq'])
    for cname in ts.columns:
        res = res.append({'ip': cname, 'ssq': spectre_deviations(ts[cname], means, sd)}, ignore_index = True)

    return res.sort_values(by = 'ssq', ascending = False).reset_index(drop = True)



def search(data, args, iplist, ipLen, bar, key, avgStatus, rdict, n):
    th = int(args.state['time']['threshFreq'])
    low_b = int(str(args.state['time']['averagedFreqInt']).split(':')[1])
    high_b = int(str(args.state['time']['averagedFreqInt']).split(':')[0])

    ip = pd.DataFrame(columns = ['ip', 'reqs', 'peaks'])
    iplist = iplist.reset_index(drop = True)
    print(iplist)

    for i in range(min(len(iplist.index), ipLen)):
        logging.debug("check: IP %s", iplist[key][i])
        subset = data[data[key] == iplist[key][i]]
        bar.update(float(i)/min(len(iplist.index), ipLen) * 100,
                freq = len(subset),
                ip = iplist[key][i])
        if subset.empty:
            logging.debug("Empty data for IP %s, continue",
                    iplist[key][i])
            continue

        if len(subset) > low_b:
            logging.debug("No handling data for IP %s (len %d > %d), continue",
                    iplist[key][i], len(subset), low_b)
            continue

        if len(subset) < th or len(subset) < high_b:
            logging.info("finding_spectral_anomaly: stop by threshFreq %d",
                    max(th, high_b))
            bar.update(100, ip = "End", freq = "None")
            break

        ts = resampling(subset, args)
        f, ps, peaks = fourier_with_peaks(ts, args)
        f_t = fourier_t(f=f, ps=ps, peaks=peaks)
        if len(peaks) > 0 or args.state['time']['allPlot'] == 'yes':
            prefix_arg = prefix_p(ip=str(iplist[key][i]),
                    start_time=str(args.tb[0]),
                    fin_time=str(args.tb[1]),
                    freq=str(len(subset))
                    )
            args.output = prefix_ip_plot(prefix_arg)
            IO.Time.plot_fourier_spectre(f_t, args)
            logging.info("Found peaks in %d addr %s (freq %s)",
                    len(peaks), iplist[key][i], prefix_arg.freq)
            IO.Time.plot_timesquash_count(ts, args)

            ip = ip.append({'ip': iplist[key][i], 'reqs': len(subset), 'peaks': len(peaks)}, ignore_index = True)
            continue

        logging.debug("IP %s (#%d) passed", 
                iplist[key][i], i)

        if avgStatus == 'yes':
            _ip_avg[iplist[key][i]] = pd.Series(ps)

    rdict[n] = ip



'''
Walk by ip time series subsets and count peaks. If len(peaks) > 2, dump result.
'''
def finding_spectral_anomaly(data, args):
    field = args.field
    iplist, key, ipLen = get_ip_list(data, args)
    logging.debug("finding_spectral_anomaly: key %s, ipLen %d, len iplist %d",
            key, ipLen, len(iplist[key]))
    if args.verbose:
        print(iplist)

    avgStatus = args.state['time']['avgPlot']

    iplist = counts_ip_filter(iplist, args).reset_index()
    if avgStatus == 'yes':
        _ip_avg = pd.DataFrame.from_dict({'noname':[None] * get_len_resampled_subset(data, key, iplist, args)})

    bar = progressbar.ProgressBar(prefix = '(freq {variables.freq})  Check {variables.ip}',
            variables = {'ip': '--', 'freq': '--'}).start()
    args.field = field

    procs = []
    rdict = mp.Manager().dict()
    k = int(args.state['pipeline']['processCnt'])
    for i in range(k):
        ipl = ds_split(iplist, k, i)
        proc = mp.Process(target = search,
            args = (data, args, ipl, ipLen, bar, key, avgStatus, rdict, i))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    bar.finish()
    if avgStatus == 'yes':
        averages_check(_ip_avg, args)
    res = pd.DataFrame(columns = ['ip', 'reqs', 'peaks'])
    for i in range(k):
        res = res.append(rdict[i], ignore_index = True)

    IO.Time.write_pretty_txt(res, args)
    return res



def averages_check(_ip_avg, args):
    logging.debug("Plot average_freq_interval")
    # plot average spectre:
    _ip_avg = _ip_avg.drop(['noname'], axis=1)
    means, sd = average_freq_interval(_ip_avg, args)

    spd = find_spectre_deviation(_ip_avg, means, sd, args)

    for i in range(int(args.state['time']['avgTop'])):
        if args.state['time']['avgType'] == 'reverse':
            idx = len(spd.index) - i - 1
        else:
            idx = i

        args.output = "_".join(["ms", spd['ip'][idx], args.state['time']['avgType'],
            args.tb[0], args.tb[1]])
        par = iocfg(
            x = means.index, y=means, addition_y=sd,
            figsize = (20, 9), grid=True,
            title="Mean spectre from {} to {} freq (by {} points, {} sample)".format(high_b, low_b, 
                len(_ip_avg.columns), args.round),
            compared = _ip_avg[spd['ip'][idx]]
            )
    
        IO.plot_line_err(par)
        IO.savefig(args, ".eps")



    args.output = "_".join(["ip", args.tb[0], args.tb[1]])
    IO.Time.write_pretty_txt(pd.DataFrame.from_dict({"IP": ip, "Freq": freqs}), args)
    par = iocfg(x = iplist[iplist.columns[1]][:i],
            y = iplist[iplist.columns[2]][:i],
            title = "IP freq distribution",
            figsize = (12, 12),
            addition_x = idx,
            addition_y = iplist[iplist.columns[2]][idx]
            )
    IO.plot_line(par)
    IO.savefig(args, "_ip_distr.eps")



'''
Probably this fuctions is not needs: use 'resample' from Pandas?
'''
def times_rounding(data, args):
    '''
    Note: this cycling is not effective opportunity do it
    for ts in data[args.times]:
        print(ts.floor('1s'))
    '''
    return pd.Series(data[args.times]).dt.round(args.round)


'''
Non-uniform time series -> uniform time series
'''
def resampling(data, args):
    assert args.round != "no", 'You must run this with --round parameter! Exit'
    if args.resampling == "counter":
        c = Counter(data[args.times])
        uts = pd.Series(c).resample(args.round).sum()
    elif args.resampling == "mean":
        c = data[args.field]
        uts = pd.Series(c).resample(args.round).mean().fillna(0)
    elif args.resampling == "median":
        c = data[args.field]
        uts = pd.Series(c).resample(args.round).median().fillna(0)
    elif args.resampling == "var":
        c = data[args.field]
        uts = pd.Series(c).resample(args.round).var().fillna(0)
    else:
        logging.error("Unknown resampling type %s. Exit", args.resampling)
        sys.exit(1)

    IO.write(uts, args, "_uts.csv")
    return uts



def date_range_from_sample(data, args):
    return pd.date_range(start=data[args.times][0],
            end=data[args.times][len(data.index) - 1],
            periods=int(args.state['time']['samplesCount']))


def time_calls(data, args):
    ts = resampling(data, args)
       
    IO.Time.plot_timesquash_count(ts, args)
    f, ps, peaks = fourier_with_peaks(ts, args)
    f_t = fourier_t(f=f, ps=ps, peaks=peaks)
    IO.Time.plot_fourier_spectre(f_t, args)
    return f_t


def standartization_ts(data):
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(data)



def compare_2ts(data, args):
    col = ['blue', 'red', 'black', 'green']
    lab = data[0].columns[-2:]
    for i in range(len(data)):
        data[i][lab] = standartization_ts(data[i][lab])
    IO.plot_2ts(iocfg(x=data, label=lab, grid=True, figsize=(20,9), color=col))
    print(data[0][lab[1]].corr(data[1][lab[1]]))


