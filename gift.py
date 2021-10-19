import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
import random
from collections import Counter
import networkx as nx
from networkx.algorithms.community.centrality import girvan_newman
from itertools import islice
from scipy.optimize import curve_fit
import networkx.algorithms.community as nx_comm


if int(sys.argv[1])==0:
    if not os.path.exists("./figs"):
        os.mkdir("./figs")
        os.mkdir("./figs/graph")
    if not os.path.exists("./res"):
        os.mkdir("./res")

def linear_fit(x, a, b):
    return a * x + b

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g

class State:
    def __init__(self, families, connection):
        self.families = families
        self.connection = connection
        # self.df = pd.DataFrame()

class Family:
    def __init__(self, family_id, wealth, status, debt, subordinates, independent_duration, subordinate_duration, rich_duration):
        self.family_id = family_id
        self.wealth = wealth
        self.status = status
        self.debt = debt
        self.subordinates = subordinates
        self.earned = 0
        self.give = 0
        self.given = 0
        self.independent_duration = independent_duration
        self.subordinate_duration = subordinate_duration
        self.rich_duration = rich_duration

def community_size_analysis(cur_connection):
    df = pd.DataFrame(cur_connection)
    G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())

    k = num_families
    comp = girvan_newman(G)
    all_communities = []
    for communities in islice(comp, k):
        all_communities.append(list(sorted(c) for c in communities))
        if len(all_communities[-1]) == num_families:
            break

    unique_communities = []
    for community in all_communities:
        for com in community:
            if com not in unique_communities:
                unique_communities.append(com)
    community_sizes = np.array([len(community) for community in unique_communities])

    return community_sizes


def generation(families, connection):
    if exchange == 0:
        for family in families:
            family.wealth += 1 + math.log(1 + family.wealth * feedback)
    else:
        for e in range(exchange):
            independents = [family.family_id for family in families if family.status == 1]
            random.shuffle(independents)
            if len(independents) > 1:
                for doner_id in independents:
                    doner = families[doner_id]
                    if random.random() > exploration:
                        weights = connection[independents, doner_id]
                        if np.sum(weights) > 0:
                            recipient_id = random.choices(independents, weights = weights, k= 1)[0]
                        else:
                            recipient_id = random.choice(list(set(independents) - set([doner_id])))
                    else:
                        recipient_id = random.choice(list(set(independents) - set([doner_id])))
                    if doner_id != recipient_id and doner.wealth > epsilon:
                        recipient = families[recipient_id]
                        doner.wealth -= epsilon
                        recipient.wealth +=  doner.wealth
                        recipient.debt.append([doner, doner.wealth * (interest - 1)])
                        doner.give = doner.wealth
                        recipient.given += doner.wealth
                        doner.wealth = 0
                        connection[recipient_id, doner_id] += eta

            for family in families:
                if birth == "linear":
                    family.earned = (1 + math.log(1 + family.wealth * feedback)) / exchange
                else:
                    family.earned = (1 + family.wealth * feedback) / exchange
                # production = 1 + family.wealth * feedback
            for family in families:
                production = family.earned
                if family.status == 1:
                    family.wealth += production
                else:
                    num_boss = len(family.debt)
                    family.wealth += production / 2
                    for j in range(num_boss):
                        family.debt[j][0].wealth += production / 2 / num_boss

            for family in families:
                family.wealth += family.give - family.given
                family.give = 0
                family.given = 0

            family_ids = list(range(num_families))
            random.shuffle(family_ids)
            for family_id in family_ids:
                family = families[family_id]
                if len(family.debt) == 0:
                    family.status = 1
                    continue
                else:
                    count = 0
                    for debt in family.debt:
                        owner = debt[0]
                        debt_wealth = debt[1]
                        if family.wealth >= debt_wealth:
                            owner.wealth += debt_wealth
                            connection[owner.family_id, family_id] += eta
                            if family in owner.subordinates:
                                owner.subordinates.remove(family)
                            family.wealth -= debt_wealth
                            count += 1
                        else:
                            owner.wealth += family.wealth
                            connection[owner.family_id, family_id] += eta
                            debt[1] -= family.wealth
                            family.wealth = 0
                            break
                    family.debt = family.debt[count:]
                    if len(family.debt) == 0:
                        family.status = 1
                    else:
                        family.status = 0
                        for debt in family.debt:
                            owner = debt[0]
                            if not family in owner.subordinates:
                                owner.subordinates.append(family)
            connection = connection / np.sum(connection, axis = 0)

    # for i in range(len(families)):
    #     families[i].family_id = i

    statuses = []
    num_subordinates =  []
    wealths = []
    connection_to = np.sum(connection, axis = 1)
    for family in families:
        wealths.append(family.wealth)
        statuses.append(family.status)
        num_subordinates.append(len(family.subordinates))

    wealths.sort()
    top_5pc = wealths[round(num_families * 0.95)]

    independent_duration, subordinate_duration, rich_duration = [], [], []
    for family in families:
        if family.independent_duration > 0:
            if family.status == 1:
                family.independent_duration += 1
            else:
                family.subordinate_duration = 1
                independent_duration.append(family.independent_duration)
                family.independent_duration = 0
        else:
            if family.status == 1:
                family.independent_duration = 1
                subordinate_duration.append(family.subordinate_duration)
                family.subordinate_duration = 0
            else:
                family.subordinate_duration += 1
        if family.wealth >= top_5pc:
            if family.rich_duration > 0:
                family.rich_duration += 1
            else:
                family.rich_duration = 1
        elif family.rich_duration > 0:
            rich_duration.append(family.rich_duration)
            family.rich_duration = 0

    return families, connection, statuses, num_subordinates, wealths, connection_to, independent_duration, subordinate_duration, rich_duration


def reproduction(families, connection):
    # wealths = [1 + family.wealth * feedback for family in families]
    if birth == "linear":
        wealths = [1 + family.wealth * feedback for family in families]
    else:
        wealths = [1 + math.log(1 + family.wealth) for family in families]
    births = Counter(random.choices(list(range(num_families)), weights = wealths, k = num_families))
    count = num_families
    next_families = []
    for family_id in births:
        family = families[family_id]
        num_children = births[family_id]
        if distribution == "normal":
            weights = np.random.rand(num_children)
        elif distribution == "exp":
            weights = np.exp(-np.array(range(num_children)))
        weights = weights / np.sum(weights)
        weights_ori = np.copy(weights)
        if len(weights) > 1:
            for i in range(len(weights)):
                if i > 0:
                    weights[i] += weights[i - 1]
        wealth = family.wealth * (1 - decay)
        subordinates = family.subordinates
        debts = family.debt
        for debt in debts:
            debt[1] = debt[1] * (1 - decay)
        for i in range(len(weights)):
            weight = weights[i]
            cur_wealth = wealth * weight
            wealth -= cur_wealth
            cur_subordinates, cur_debts = [], []
            if family.status == 1:
                if len(subordinates) > 0:
                    cur_subordinates = random.sample(subordinates, k = int(round(len(subordinates) * weight)))
                    subordinates = list(set(subordinates) - set(cur_subordinates))
            else:
                if len(debts) > 0:
                    cur_debts = random.sample(debts, k = int(round(len(debts) * weight)))
                    for debt in cur_debts:
                        debts.remove(debt)

            next_families.append(Family(count - num_families, cur_wealth, 1 * (len(cur_debts) == 0), cur_debts, cur_subordinates, family.independent_duration, family.subordinate_duration, family.rich_duration))
            for subordinate in cur_subordinates:
                for debt in subordinate.debt:
                    if debt[0] == family:
                        debt[0] = families[-1]
            for debt in cur_debts:
                boss = debt[0]
                boss.subordinates = list(set(boss.subordinates) - set([family]) | set([families[-1]]))

            arr = connection[family_id] * np.random.normal(1.0, mutation, count) * weights_ori[i]
            connection = np.insert(connection, count, arr, axis = 0)
            arr = connection[:, family_id] * np.random.normal(1.0, mutation, count + 1)
            connection = np.insert(connection, count, arr, axis = 1)
            for j in range(i):
                connection[count, count - j - 1] = 1 / num_families
                connection[count - j - 1, count] = 1 / num_families
            count += 1

    for family_id in range(num_families):
        if births[family_id] == 0:
            family == families[family_id]
            for subordinate in family.subordinates:
                my_debt = []
                for debt in subordinate.debt:
                    if debt[0] == family:
                        subordinate.debt.remove(debt)
                        if len(subordinate.debt) == 0:
                            subordinate.status = 1
                        break
            for debt in family.debt:
                boss = debt[0]
                boss.subordinates.remove(family)

    # connection = np.delete(connection,list(range(num_families)), 0)
    # connection = np.delete(connection,list(range(num_families)), 1)
    connection = connection[num_families:, num_families:]
    connection = connection / np.sum(connection, axis = 0)
    families = next_families[:]


    return families, connection


def main():
    families = [Family(i, 1.0, 1, [], [], 1, 0, 0) for i in range(num_families)]
    # connection[i, j] represents the weaight of the pass from j to i.
    connection = np.random.rand(num_families, num_families)
    for i in range(num_families):
        connection[i, i] = 0
    connection = connection / np.sum(connection, axis = 0)
    state = State(families, connection)

    for iter in range(iteration):
        state.families, state.connection, statuses, num_subordinates, wealths, connection_to, independent_duration, subordinate_duration, rich_duration = generation(state.families, state.connection)
        # state.df[iter] = [statuses, num_subordinates, wealths, connection_to]
        state.families, state.connection = reproduction(state.families, state.connection)
        # print(iter)

    degree_assortativity_res, average_path_res2, average_cluster_res, core_res, modularity_res, num_comms_res = [], [], [], [], [], []
    statuses_res, num_subordinates_res, wealths_res, connection_to_res, connection_to_res2, flow_hierarchy_res, gini_res, population_ratio_res = [], [], [], [], [], [], [], []
    independent_duration_res, subordinate_duration_res, rich_duration_res = [], [], []
    for iter in range(100):
    # for iter in range(10):
        state.families, state.connection, statuses, num_subordinates, wealths, connection_to, independent_duration, subordinate_duration, rich_duration = generation(state.families, state.connection)
        # state.df[iteration - 100 + iter] = [statuses, num_subordinates, wealths, connection_to]
        statuses_res.extend(statuses)
        num_subordinates_res.extend(num_subordinates)
        wealths_res.extend(wealths)
        connection_to_res.extend(connection_to)
        wealths = np.array(wealths)
        gini_res.append(gini(wealths[wealths > 0]))
        independent_duration_res.extend(independent_duration)
        subordinate_duration_res.extend(subordinate_duration)
        rich_duration_res.extend(rich_duration)

        cur_connection = 1 * (state.connection > eta)
        df = pd.DataFrame(cur_connection)
        G2 = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())

        G = nx.from_numpy_matrix(1 * (state.connection > eta), create_using=nx.Graph())

        # G = nx.from_numpy_matrix(1 / state.connection, create_using=nx.DiGraph())

        if len(G2.edges()) < num_families:
            cur_connection = 1 * (state.connection > 1 / num_families)
            df = pd.DataFrame(cur_connection)
            G2 = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())
        try:
            # average_path_res1.append(nx.average_shortest_path_length(G, weight = "weight"))
            average_path_res2.append(nx.average_shortest_path_length(G2))
            core_res.append(max(nx.core_number(G2).values()))
        except:
            pass
        try:
            communities = nx_comm.greedy_modularity_communities(G)
            modularity = nx_comm.modularity(G, communities)
            num_comms = len(communities)
            modularity_res.append(modularity)
            num_comms_res.append(num_comms)
        except:
            pass
        average_cluster_res.append(nx.average_clustering(G2))
        flow_hierarchy_res.append(nx.flow_hierarchy(G2))
        independents, subordinates  = [], []
        for id in range(num_families):
            if families[id].status == 1:
                independents.append(id)
            else:
                subordinates.append(id)
        population_ratio = int(round(sum(statuses_res) / len(statuses_res), 2) * 100)
        population_ratio_res.append(population_ratio)
        connection_to_res2.extend(np.sum(cur_connection, axis = 1))
        degree_assortativity_res.append(nx.degree_pearson_correlation_coefficient(G2))

        # try:
        #     # community_sizes_1time.extend(community_size_analysis(1 * (state.connection > 1 / num_families)))
        #     community_sizes_3times.extend(community_size_analysis(1 * (state.connection > 3 / num_families)))
        # except:
        #     pass

        state.families, state.connection = reproduction(state.families, state.connection)

    # communities = nx_comm.greedy_modularity_communities(G)
    # modularity = nx_comm.modularity(G, communities)
    # print(modularity)
    # for i in range(len(communities)):
    #     print(nx.flow_hierarchy(nx.subgraph(G2, communities[i])))

    if True:
        community_sizes = np.array(connection_to_res2)
        community_sizes.sort()
        sizes = community_sizes[::-1][num_families // 10 : num_families * 10]
        ranks = np.arange(community_sizes.size)[num_families // 10 : num_families * 10]
        param, cov = curve_fit(linear_fit, sizes, np.log(ranks))
        exp2 = - param[0]

        community_sizes = np.array(connection_to_res)
        community_sizes.sort()
        sizes = community_sizes[::-1][num_families // 10 : num_families * 10]
        ranks = np.arange(community_sizes.size)[num_families // 10 : num_families * 10]
        param, cov = curve_fit(linear_fit, sizes, np.log(ranks))
        exp1 = - param[0]

        community_sizes = np.array(wealths_res)
        community_sizes.sort()
        sizes = community_sizes[::-1][num_families // 10 : num_families * 10]
        ranks = np.arange(community_sizes.size)[num_families // 10 : num_families * 10]
        param, cov = curve_fit(linear_fit, sizes, np.log(ranks))
        exp3 = - param[0]



    # if False:
    #     community_sizes = np.array(community_sizes_3times)
    #     community_sizes.sort()
    #     sizes = community_sizes[::-1][num_families // 10 : num_families * 10]
    #     ranks = np.arange(community_sizes.size)[num_families // 10 : num_families * 10]
    #     param, cov = curve_fit(linear_fit, np.log(sizes), np.log(ranks))
    #     power = - param[0]
    # power = 0

    if trial == 0:
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(population_ratio_res)
        # # ax.set_yscale('log')
        # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # # ax.set_ylim(-0.1,1.5)
        # ax.tick_params(labelsize=14)
        # fig.tight_layout()
        # fig.savefig(f"figs/{path}_{trial}_population_ratio.pdf")
        # plt.close('all')
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(flow_hierarchy_res)
        # # ax.set_yscale('log')
        # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # # ax.set_ylim(-0.1,1.5)
        # ax.tick_params(labelsize=14)
        # fig.tight_layout()
        # fig.savefig(f"figs/{path}_{trial}_flow_hierarchy.pdf")
        # plt.close('all')
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # ax.plot(average_cluster_res)
        # # ax.set_yscale('log')
        # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # # ax.set_ylim(-0.1,1.5)
        # ax.tick_params(labelsize=14)
        # fig.tight_layout()
        # fig.savefig(f"figs/{path}_{trial}_average_cluster.pdf")
        # plt.close('all')

        wealths = np.array(wealths)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(connection_to[independents], wealths[independents], c = "b")
        ax.scatter(connection_to[subordinates], wealths[subordinates], c = "m")
        ax.set_xlabel("connection_to",fontsize=18)
        ax.set_ylabel(r"$w$",fontsize=18)
        # ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_wealth_connection_to.pdf")
        plt.close('all')


        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(np.ravel(connection[independents]), bins = 50, alpha = 0.5, color = "b", density = 1)
        ax.hist(np.ravel(connection[subordinates]), bins = 50, alpha = 0.5, color = "m", density = 1)
        ax.set_xlabel("connection_from",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_from.pdf")
        plt.close('all')

        statuses = []
        for family in state.families:
            if family.status == 1:
                statuses.append("b")
            else:
                statuses.append("m")

        wealths = []
        for family in state.families:
            wealths.append(family.wealth)

        wealths = np.array(wealths)
        # wealths = np.log(wealths)
        wealths = wealths / np.max(wealths)


        # cur_connection = 1 * (state.connection > 10 / num_families)
        # df = pd.DataFrame(cur_connection)
        # G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1,1,1)
        # nx.draw_networkx(G, node_size = 10, with_labels=False, ax = ax, node_color = statuses)
        # # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # # ax.set_ylim(-0.1,1.5)
        # ax.tick_params(labelsize=14)
        # fig.tight_layout()
        # fig.savefig(f"figs/{path}_{trial}_connection_10times_{population_ratio}pc.pdf")
        # plt.close('all')

        cur_connection = 1 * (state.connection > 3 / num_families)
        df = pd.DataFrame(cur_connection)
        G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        nx.draw_networkx(G, node_size = 10, with_labels=False, ax = ax, node_color = statuses)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_3times_{population_ratio}pc.pdf")
        fig.savefig(f"figs/graph/{path}_{trial}_connection_3times_{population_ratio}pc.pdf")
        plt.close('all')


        if len(G.edges()) < num_families:
            cur_connection = 1 * (state.connection > 1 / num_families)
            df = pd.DataFrame(cur_connection)
            G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            nx.draw_networkx(G, node_size = 10, with_labels=False, ax = ax, node_color = statuses)
            # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
            # ax.set_ylim(-0.1,1.5)
            ax.tick_params(labelsize=14)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_connection_1time_{population_ratio}pc.pdf")
            plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_res, bins = 50)
        ax.set_xlabel("connection_to",fontsize=18)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_to_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_res, bins = 50)
        ax.set_xlabel("connection_to",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_to_log_{population_ratio}pc.pdf")
        plt.close('all')


        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_res2, bins = 50)
        ax.set_xlabel("connection_to",fontsize=18)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_to2_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_res2, bins = 50)
        ax.set_xlabel("connection_to",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_to2_log_{population_ratio}pc.pdf")
        plt.close('all')


        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealths_res, bins = 50)
        ax.set_xlabel("wealth",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_wealth_log_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(num_subordinates_res, bins = 10)
        ax.set_xlabel("subordinates",fontsize=18)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_per_subordinates_{population_ratio}pc.pdf")
        plt.close('all')

    res = [np.median(np.array(population_ratio_res)), np.median(np.array(flow_hierarchy_res)), np.median(np.array(average_cluster_res)), np.corrcoef(wealths_res, connection_to_res)[0,1], np.median(np.array(gini_res)), np.median(np.array(core_res)), np.median(np.array(independent_duration_res)), np.median(np.array(subordinate_duration_res)), np.median(np.array(rich_duration_res)), np.median(np.array(degree_assortativity_res)), np.median(np.array(modularity_res)), np.median(np.array(num_comms_res)), exp1, exp2, exp3]

    return res

num_families = 100
iteration = 1000
mutation = 0.01
trial = 0
interest = 1.2
decay = 0.5
death = 1.0
feedback  = 1.0
eta = 0.03
ratio = 0.7
epsilon = 0.01
exploration = 0.1
exchange = 30
trial = 0
birth = "linear"
distribution = "exp"
eta_ = 3


# for birth in [0.8, 1.0, 1.2]:
for birth in ["log", "linear"]:
    for num_families in [30, 100, 300]:
        for exchange in [1, 3, 10, 30]:
            for decay in [[0.0, 0.1, 0.3][int(sys.argv[1]) % 3]]:
                for interest in [[1.1, 1.2, 1.5, 2.0, 3.0][int(sys.argv[1]) // 4 % 5]]:
                    df = pd.DataFrame(index = ["exchange", "decay", "exploration",  "eta", "feedback", "epsilon", "mutation", "interest", "num_families", "distribution", "birth", "population_ratio", "hierarchy", "cluster_index", "corrcoef", "gini", "core", "independent_duration", "subordinate_duration", "rich_duration", "assortativity", "modularity", "num community", "exp1", "exp2", "exp3"])
                    eta = eta_ / num_families
                    path = f"dist{distribution}_birth{birth}_{num_families}fam_{exchange}exchange_d{round(decay * 100)}pc_ex{round(exploration * 100)}pc_eta{round(eta_ * 100)}pc_f{round(feedback * 100)}pc_epsilon{round(epsilon * 100000)}pm_mu{round(mutation * 1000)}pm_interest{round(interest * 10)}pd"
                    for trial in range(30):
                        try:
                            res = main()
                            params = [exchange, decay, exploration, eta, feedback, epsilon, mutation, interest, num_families, distribution, birth]
                            params.extend(res)
                            df[len(df.columns)] = params
                        except:
                            pass
                    df.to_csv(f"res/res_{path}.csv")
