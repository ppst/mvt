#!/usr/bin/python3

import train
import sys


__author__     = "Philip Stegmaier"
__contact__    = "https://github.com/ppst/mvt/issues"
__copyright__  = "Copyright (c) 2021, Philip Stegmaier"
__license__    = "https://en.wikipedia.org/wiki/MIT_License"
__maintainer__ = "Philip Stegmaier"
__version__    = "0.1.0"


apps = {'train': train.MultiVectorTrainer().main }

def main():
    cmd = sys.argv[0]
    if not cmd in apps:
        raise ValueError("Error: invalid application " + cmd)
    apps[cmd]()
    

if __name__ == "__main__":
    sys.argv = sys.argv[1:]
    main()
