import networkx as nx
import numpy as np
import pickle
import pandas as pd 

#change accordingly
path='/home/divya/Desktop/general_64_5.txt'
data = pd.read_csv(path, header = None, delimiter='|')

nums = ['04','09','14','19','24','29','34','39','44','49','54','59','64','69','74','79','84','89','94','99','104','109']
nums_graphs = ['4','9','14','19','24','29','34','39','44','49','54','59','64','69','74','79','84','89','94','99','104','109',]


node_to_label = {}
node_to_label_1 = {}



'''
Collecting common nodes from all timesteps
'''










for ia in nums:
    if ia == '04':
        graph_id = '4'
        labels = open("general_64_5.txt","r")
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
        compare_node_to_label = {}
        compare_node_to_label_1 = {}
        if ia == '09':
            graph_id = '9'
        else:
            graph_id = ia
        labels = open("general_64_5.txt","r")
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
common_labels = node_to_label.copy()
print("Len: ", str(len(common_labels)))

'''
Creating actual batches of graphs to be used as training examples
'''

for ia in nums:
    G_list = []
    label_list = []
    if ia == '04':
        graph_id = '4'
    elif ia == '09':
        graph_id = '9'
    else:
        graph_id = ia

    labels = open("general_64_5.txt","r")
    node_to_labelling = {}
    node_to_labelling_1 = {}
    for f in labels.readlines():
        node_num1 = f.split("|")[0]
        node_num = int(node_num1.split("_")[0])
        node_label = int(f.split("|")[2])
        la = [0,0]
        la[node_label] = 1
        node_to_labelling[node_num] = node_label
        node_to_labelling_1[node_num1] = node_label

    for j in range(int(graph_id),int(graph_id)-5,-1):
        G = nx.read_graphml("../../STWalk/ciao/input_graphs/graph_"+str(j)+".graphml")
        node_list = list(G.nodes())
        # print(nx.to_numpy_matrix(G))
        x = common_labels.copy()
        y = node_to_labelling.copy()
        # print(len(x))
        n1 = {}
        for i in node_list:
            gh = int(i.split("_")[0])
            if gh not in common_labels.keys():
                # print("Not here", gh)
                G.remove_node(i)
            else:
                # print("Here", gh)
                n1[i.split("_")[0]+"_"+str(j)] = y[gh]
                del x[gh]
        print("Len before: ", str(len(n1)))
        for i in x.keys():
            n1[str(i)+"_"+str(j)] = y[i]
            G.add_node(str(i)+"_"+str(j))
        print("Len after: ", str(len(n1)))
        # print(nx.to_numpy_matrix(G))
        pos = nx.spring_layout(G)
        # nx.draw_networkx_labels(G,pos,n1)
        D = nx.to_numpy_matrix(G)
        # print(D)
        G_list.append(D)
        label_list.append(n1)
        print("Done "+str(j)+" time step each")
    print("Done "+ia+" labeled time step")
# print("Done finally")
    G_list = np.array(G_list)
    filename = open("result_ciao" + graph_id + ".dat","wb")
    pickle.dump(G_list,filename)
    filename.close()
    filename = open("./labels_3dgcn_ciao" + graph_id + ".dat","wb")
    pickle.dump(label_list,filename)
    filename.close()
