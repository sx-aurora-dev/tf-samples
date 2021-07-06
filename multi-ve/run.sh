#! /bin/sh

https_proxy=

VE_NODE_NUMBER=0 TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 0}}' python test01.py >log0 2>&1 &
sleep 1

VE_NODE_NUMBER=2 TF_CONFIG='{"cluster": {"worker": ["localhost:12345", "localhost:23456"]}, "task": {"type": "worker", "index": 1}}' python test01.py
