import networkx as nx
import numpy as np
import pickle


nums = ['04','09','14','19','24','29','34','39','44','49','54','59','64','69','74','79','84','89','94','99','104','109']
node_to_label = {}
node_to_label_1 = {}
compare_node_to_label = {}
compare_node_to_label_1 = {}

for ia in nums:
    if ia == '04':
        graph_id = '4'
        labels = open("labels_"+ia+".txt","r")
        print(len(node_to_label))
        for f in labels.readlines():
            node_num1 = f.split("|")[0]
            node_num = int(node_num1.split("_")[0])
            node_label = int(f.split("|")[2])
            node_to_label[node_num] = node_label
            node_to_label_1[node_num1] = node_label
        print(len(node_to_label))
        print("Done "+ia+" time step")
        continue
    else:
        if ia == '09':
            graph_id = '9'
        else:
            graph_id = ia
        labels = open("labels_"+ia+".txt","r")
        for f in labels.readlines():
            node_num1 = f.split("|")[0]
            node_num = int(node_num1.split("_")[0])
            node_label = int(f.split("|")[2])
            compare_node_to_label[node_num] = node_label
            compare_node_to_label_1[node_num1] = node_label
        x = node_to_label.copy()
        print(len(node_to_label))
        for i in x.keys():
            if i not in compare_node_to_label.keys():
                del node_to_label[i]
        print(len(node_to_label))
        print("Done "+ia+" time step")
print(list(node_to_label.keys()))