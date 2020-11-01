# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import

import sys
import os
import time


def get_eta(time_from, pct=None):
    if time_from is None:
        return ''

    if pct is None:
        seconds = time.time() - time_from
    elif pct <= 0:
        return ''
    else:
        if pct > 1.0:
            pct = 1.0
        seconds = (time.time() - time_from) * (1 - pct) / pct

    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    if days >= 1:
        days = int(days)
        hours = int(hours - days * 24)
        return str(days) + 'd' + str(hours) + 'h'
    elif hours >= 1:
        hours = int(hours)
        minutes = int(minutes - hours * 60)
        return str(hours) + 'h' + str(minutes) + 'm'
    elif minutes >= 1:
        minutes = int(minutes)
        seconds = int(seconds - minutes * 60)
        return str(minutes) + 'm' + str(seconds) + 's'
    else:
        if seconds >= 1:
            seconds = int(seconds)
            return str(seconds) + 's'
        else:
            milliseconds = int(seconds * 1000)
            return str(milliseconds) + 'ms'


def show_bar(time_from, pct, s=""):
    if pct > 1.0:
        pct = 1.0

    sys.stdout.write("\r%6.2f%% |%-21s| %-6s %s" % (
        pct * 100,
        '='*int(20 * pct) + '>',
        get_eta(time_from, pct),
        s)
    )
    sys.stdout.flush()


class Progress(object):
    def __init__(self, num=1):
        self.num = num
        self.start_num = 0

        self.has_bar = False

    def __iter__(self):
        self.time_from = time.time()
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        self.start_num += 1
        if self.start_num >= self.num:
            if self.has_bar:
                sys.stdout.write("\n")
            sys.stdout.write("time elapsed: %s\n" %
                             get_eta(self.time_from))
            sys.stdout.flush()
            raise StopIteration

        if self.num > 1:
            self.bar((self.start_num + 1) / self.num)
        return self.start_num

    def bar(self, pct):
        self.has_bar = True

        show_bar(self.time_from, pct)

    def __enter__(self):
        self.time_from = time.time()
        return self

    def __exit__(self, type, value, trace):
        if self.has_bar:
            sys.stdout.write("\n")
        sys.stdout.write("time elapsed: %s\n" %
                         get_eta(self.time_from))
        sys.stdout.flush()


if __name__ == '__main__':
    with Progress() as p:
        for i in range(10):
            for j in range(10000000):
                pass
            p.bar((i + 1) / 10)

    for i in Progress(10):
        for j in range(10000000):
            pass

    print("end")
