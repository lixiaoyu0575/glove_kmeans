# coding=utf8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import json
import numpy as np


with open("./../data/GZDMZYYHXXZB_words.txt", "r") as f:
    words = eval(f.readline())
print "read CP words dic"

jsonF = open("./../data/res_GZDMZYYHXXZB.json")
jsonD = json.load(jsonF)
sequences = []
sequence = []
for i in range(len(jsonD)):
    if i > 0 and jsonD[i]["id"] != jsonD[i-1]["id"]:
        sequences.append(sequence)
        sequence = []
    events = []
    if jsonD[i]["ops"] != " ":
        ops = jsonD[i]["ops"].split(":")
        for a in ops:
            events.append(a)
    if jsonD[i]["drug"] != " ":
        drug = jsonD[i]["drug"].split(":")
        for a in drug:
            events.append(a)
    if jsonD[i]["undrug"] != " ":
        undrug = jsonD[i]["undrug"].split(":")
        for a in undrug:
            events.append(a)
    sequence.append(events)
    if i == len(jsonD) - 1:
        sequences.append(sequence)



fileOut = file("./../data/glove_input", "a+")
# fileOut.write(str(len(words)) + "\t" + str(embSize) + "\n")
for seq in sequences:
    emd_str = ""
    for a in seq:
        for b in a:
            emd_str += "\t" + str(words.index(b))
    emd_str += "\n"
    print emd_str
    fileOut.write(emd_str)
fileOut.close()