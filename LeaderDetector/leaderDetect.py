# -*- coding: utf-8 -*-
import networkx as nx
import Queue
import numpy as np

MAX_ITER = 10

class CommunityDetector(object):
    def __init__(self, igraph):
        self.igrah = igraph
        self.Detectcommunity()

    #initialize the leader
    def initLeader(self):
        #ranking node id by nerborhoods volume
        self.nDegree = nx.degree(self.igrah, weight = 'similarity')        
        nDegree = Queue.Queue()
        for nd in sorted(self.nDegree, key=lambda (k,v):(v,k),reverse=True):
            nDegree.put(nd[0])
        #select top N node which (neiborhood volume) > N
        Hnode = []
        while not nDegree.empty():
            item = nDegree.get()
            if self.nDegree[item] > 0:
                isHnode = True
                for s in Hnode:
                    if item in nx.all_neighbors(self.igrah, s):
                        isHnode = False
                        break
                if isHnode:
                    Hnode.append(item)
        #initialize community
        self.leader = [nd for nd in Hnode]
        self.community = []
        self.outlier = []
        
    #detect community, i.e. clustering sentences into community
    def Detectcommunity(self):
        self.initLeader()
        #centr_score = nx.degree_centrality(self.igrah)
        itect = 0
        while True:
            itect += 1
            print(itect)
            #get non-leader nodes, i.e. followers
            other_nodes = [k for k in self.igrah.nodes() if k not in self.leader]            
            #sort followers by distance to leaders
            dist = []
            for nd in other_nodes:
                d = []
                for lnd in self.leader:
                    try:
                        d.append(nx.shortest_path_length(self.igrah, nd, lnd, weight='distance'))
                    except:
                        d.append(1000000)
                if len(d):
                    dist.append((nd, min(d)))
            #assign follower node to leader
            dist = sorted(dist, key=lambda (nd, v):(v, nd))
            community_local = [[] for i in range(len(self.leader))]
            for nd, _ in dist:
                #update community
                score = []
                for k in range(len(self.leader)):
                    lmember = [self.leader[k]] + community_local[k]
                    cnb = []
                    for lnd in lmember:
                        cnb += [i for i in nx.common_neighbors(self.igrah, nd, lnd)]
                        #cnb.append(len([i for i in nx.common_neighbors(self.igrah, nd, lnd)]))
                    d= len(set(cnb))
                    #d = max(cnb)
                    score.append((k,d))
                community_id, sc = max(score, key=lambda (mber, d): (d, mber))
                if sc > 0.: 
                    community_local[community_id].append(nd)
                else:
                    self.outlier.append(nd)
            #update leader
            new_leader = []
            for k in range(len(self.leader)):
                lmember = community_local[k] + [self.leader[k]]
                if len(lmember) > 1:
                    centr_score = nx.degree_centrality(self.igrah.subgraph(lmember)) #
                    #comm_centr = [centr_score[knd] for knd in lmember]
                    #mxid = np.argmax(comm_centr)
                    #new_leader.append(lmember[mxid])
                    mxid,_ = max(centr_score.items(), key=lambda (ky,v):(v,ky))
                    new_leader.append(mxid)
                    lmember.remove(mxid)
                    community_local[k] = lmember
                else:
                    new_leader.append(self.leader[k])
                #new_leader.append(lmember[mxid])
            #check leader is changing
            #print('%d %d'%(len(set(new_leader).intersection(self.leader)),len(new_leader)))
            if set(new_leader) == set(self.leader) or itect > MAX_ITER:
                #sort community in terms of community size
                comm_size = np.array([len(nn)+1 for nn in community_local])
                comm_sid = np.argsort(comm_size)[::-1]
                self.leader = [new_leader[nn] for nn in comm_sid]
                self.community = [community_local[nn] for nn in comm_sid]
                break
            #reset community
            self.leader = new_leader
            self.outlier = []
