#! /bin/sh

# http_proxy=
https_proxy=

VE_NODE_NUMBER=0 python multi.py >log1.txt 2>&1 &
sleep 2
VE_NODE_NUMBER=2 python multi.py
