#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import logging

import warnings
warnings.filterwarnings("ignore")

import logging

from multiprocessing import Process
from multiprocessing import Manager

#my modules:
from hist_io import IO
from hist_filters import Filter
from hist_setup import Statement
from hist_setup import Action
from hist_groups import ds_split

def stage(data, args, rdict, i):
    logging.info("Reading is complete: %d items", len(data.index))
    if args.state['pipeline']['dump'] == 'yes':
        IO.dump(data, args)
    
    if args.state['pipeline']['filter'] == 'yes':
        data = Filter(args)(data)
    
    # Action must return handling data result
    rdict[i] = Action()(data, args)


def work(args, i, rdict):
    data = IO.single_read(args, i)
    stage(data, args, rdict, i)


def main():
    logging.debug("start")

    args = Statement(Action().get_actions()).get_args()

    pipeline = args.state['pipeline']['type']

    rdict = Manager().dict()
    procs = []
    if pipeline == 'linear':
        for i in range(len(args.infile)):
            proc = Process(target = work,
                    args = (args, i, rdict))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()
        Action().reduce(rdict, args)

    elif pipeline == 'agg':
        data = IO.read(args)
        stage(data, args)

    elif pipeline == 'parallel':
        data = IO.read(args)
        k = int(args.state['pipeline']['processCnt'])
        for i in range(k):
            proc = Process(target = stage,
                    args = (ds_split(data, k, i), args, rdict, i))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()
        Action().reduce(rdict, args)

    else:
        logging.error('Bad pipeline type {}. Exit'.format(pipeline))
        return


    logging.debug("Success finish.")


if __name__ == "__main__":
    main()
