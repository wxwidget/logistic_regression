#!/bin/python

import sys
import random

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        flist = f.readlines()
        random.shuffle(flist)
        for line in flist:
            print line.strip()
