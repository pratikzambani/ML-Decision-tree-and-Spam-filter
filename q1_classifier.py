import random
import pickle as pkl
import argparse
import csv
import numpy as np
import copy
from scipy import stats
from math import log

'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node:
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take

    - A leaf node:
        - data: 'T' or 'F'
        - children[0]-children[4]: Doesn't matter, you can leave them the same or cast to None.

'''
tree_size = 0
# DO NOT CHANGE THIS CLASS
class TreeNode():
    def __init__(self, data='T',children=[-1]*5):
        self.nodes = list(children)
        self.data = data

    def save_tree(self,filename):
        obj = open(filename,'w')
        pkl.dump(self,obj)

# loads Train and Test data
def load_data(ftrain, ftest):
	Xtrain, Ytrain, Xtest = [],[],[]
	with open(ftrain, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtrain.append(rw)

	with open(ftest, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = map(int,row[0].split())
	        Xtest.append(rw)

	ftrain_label = ftrain.split('.')[0] + '_label.csv'
	with open(ftrain_label, 'rb') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        rw = int(row[0])
	        Ytrain.append(rw)

	print('Data Loading: done')
	return Xtrain, Ytrain, Xtest


num_feats = 274

parser = argparse.ArgumentParser()
parser.add_argument('-p', required=True)
parser.add_argument('-f1', help='training file in csv format', required=True)
parser.add_argument('-f2', help='test file in csv format', required=True)
parser.add_argument('-o', help='output labels for the test dataset', required=True)
parser.add_argument('-t', help='output tree filename', required=True)

args = vars(parser.parse_args())

pval = args['p']
Xtrain_name = args['f1']
Ytrain_name = args['f1'].split('.')[0]+ '_labels.csv' #labels filename will be the same as training file name but with _label at the end

Xtest_name = args['f2']
Ytest_predict_name = args['o']

tree_name = args['t']

Xtrain, Ytrain, Xtest = load_data(Xtrain_name, Xtest_name)


def create_decision_tree(Xtrain, Ytrain, used_feats):

    new_node = TreeNode(-1)

    vis, nvis = 0.0,0.0
    for y in Ytrain:
        if not y:
            nvis += 1.0
            continue
        vis += 1.0

  # leaf node
    if not vis or not nvis or all(used_feats):
        #print 'me leaf bro'
        new_node.data = 'T' if vis else 'F'
        return new_node
    # internal node
    else:
        #print vis, nvis, len(Ytrain)
        lenytrain = len(Ytrain)*1.0

        visentropy, nvisentropy = 0.0, 0.0
        if vis:
            visentropy = -(vis/lenytrain)*log(vis/lenytrain, 2)
        if nvis:
            nvisentropy = -(nvis/lenytrain)*log(nvis/lenytrain, 2)
        entropy = visentropy + nvisentropy

        # checking for feature to split on
        for feat in range(num_feats):
            if used_feats[feat]:
                #print 'feat', feat, ' already used'
                continue

            feat_val_count = dict()
            for x in Xtrain:
                feat_val_count.setdefault(x[feat], 0)
                feat_val_count[x[feat]] += 1

            # entropy calculation
            f_entropy = 0.0
            feat_entropy, ifgain = 0.0, 0.0
            for feat_val in feat_val_count:
                w = (feat_val_count[feat_val]*1.0)/(len(Xtrain)*1.0)
                sublabel = []
                fvis,fnvis = 0.0,0.0
                for i in range(len(Xtrain)):
                    if Xtrain[i][feat] == feat_val:
                        sublabel.append(Ytrain[i])
                        if Ytrain[i]:
                            fvis += 1.0
                        else:
                            fnvis += 1.0

                fvisentropy, fnvisentropy = 0.0, 0.0
                if fvis:
                    fvisentropy = -(fvis/(fvis+fnvis))*log(fvis/(fvis+fnvis), 2)
                if fnvis:
                    fnvisentropy = -(fnvis/(fvis+fnvis))*log(fnvis/(fvis+fnvis), 2)

                feat_entropy += w*(fvisentropy + fnvisentropy)
            # check for max information gain
            if entropy - feat_entropy > ifgain:
                ifgain = entropy - feat_entropy
                new_node.data = feat

        #print 'yo assigned new_node.data as', new_node.data
        if new_node.data == -1:
            new_node.data = 'T' if vis > nvis else 'F'
            return new_node

        summation = 0.0
        uniq_vals = set()
        child_recurse_lists = dict()
        for x in Xtrain:
            uniq_vals.add(x[new_node.data])
        #print 'uniq vals', uniq_vals
        # chi squared criteria
        for uval in uniq_vals:
            child_xtrain, child_ytrain = [], []
            for i in range(len(Xtrain)):
                if Xtrain[i][new_node.data] == uval:
                    child_xtrain.append(Xtrain[i])
                    child_ytrain.append(Ytrain[i])
            child_used_feats = copy.copy(used_feats)
            child_used_feats[new_node.data] = True
            child_recurse_lists[uval] = [child_xtrain, child_ytrain, child_used_feats]

            pidash = (vis*len(child_xtrain)*1.0)/(len(Xtrain)*1.0)
            nidash = (nvis*len(child_xtrain)*1.0)/(len(Xtrain)*1.0)
            child_vis, child_nvis = 0.0, 0.0
            for y in child_ytrain:
                if y:
                    child_vis += 1.0
                else:
                    child_nvis += 1.0
            if child_vis:
                summation += ((child_vis - pidash)*(child_vis - pidash)*1.0)/(child_vis*1.0)
            if child_nvis:
                summation += ((child_nvis - nidash)*(child_nvis - nidash)*1.0)/(child_nvis*1.0)

        p_val = 1.0 - stats.chi2.cdf(summation, len(uniq_vals))

        if p_val > pval:
            new_node.data = 'T' if vis else 'F'
            return new_node
        else:
            for uval, child_data in child_recurse_lists.iteritems():
                #print 'recursing..'
                child = create_decision_tree(child_data[0], child_data[1], child_data[2])
                new_node.nodes[uval-1] = child

    gr = False
    for i in range(0, 5):
        if new_node.nodes[i] != -1:
            gr = True
            break

    if not gr:
        new_node.data = 'T' if vis > nvis else 'F'
        return new_node

    for i in range(0, 5):
        if new_node.nodes[i] == -1:
            ch = 'T' if vis > nvis else 'F'
            new_node.nodes[i] = TreeNode(ch)

    return new_node
def get_label(root, test_data):

    if root.data == 'T':
        return 1
    elif root.data == 'F':
        return 0
    return get_label(root.nodes[test_data[int(root.data)-1]-1], test_data)

print("Training...")
used_feats = [False]*num_feats

s = create_decision_tree(Xtrain, Ytrain, used_feats)
s.save_tree(tree_name)
print("Testing...")
Ypredict = []
#generate random labels
for i in range(0,len(Xtest)):
    #Ypredict.append([np.random.randint(0,2)])
	Ypredict.append([get_label(s, Xtest[i])])

with open(Ytest_predict_name, "wb") as f:
    writer = csv.writer(f)
    writer.writerows(Ypredict)

print("Output files generated")
