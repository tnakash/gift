import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
import random
import networkx as nx
from collections import Counter
from scipy.optimize import curve_fit
import networkx.algorithms.community as nx_comm



if int(sys.argv[1])==0:
    if not os.path.exists("./res"):
        os.mkdir("./res")
        os.mkdir("./figs")


def linear_fit(x, a, b):
    return a * x + b

def gini2(y):
    y.sort()
    n = len(y)
    nume = 0
    for i in range(n):
        nume = nume + (i+1)*y[i]

    deno = n*sum(y)
    return ((2*nume)/deno - (n+1)/(n))*(n/(n-1))

class State:
    def __init__(self, families, connection):
        self.families = families
        self.connection = connection
        # self.df = pd.DataFrame()

class Family:
    def __init__(self, family_id, lifetime, wealth, inheritance, status, debt, subordinates, independent_duration, subordinate_duration, rich_duration):
        self.family_id = family_id
        self.lifetime = lifetime
        self.wealth = wealth
        self.inheritance = inheritance
        self.status = status
        self.debt = debt
        self.subordinates = subordinates
        self.counter = 0
        self.give = 0
        self.given = 0
        self.independent_duration = independent_duration
        self.subordinate_duration = subordinate_duration
        self.rich_duration = rich_duration


def separation(families, connection):
    for family in families:
        for subordinate in family.subordinates:
            family.subordinates = list(set(family.subordinates) & set(families))
        remove_ls = []
        for debt in family.debt:
            if debt[0] not in families:
                remove_ls.append(debt)
        for debt in remove_ls:
            family.debt.remove(debt)
        if len(family.debt) == 0:
            family.status = 1
    for i in range(len(families)):
        families[i].family_id = i

    connection = connection / np.sum(connection, axis = 0)

    return families, connection

# state = states[-2]
# families, connection = state.families, state.connection

def generation(families, connection):
    for i in range(len(families)):
        families[i].family_id = i

    family_ids = list(range(len(families)))
    random.shuffle(family_ids)
    donation_ls = []
    for doner_id in family_ids:
        doner = families[doner_id]
        doner.counter += 1
        if doner.wealth  > 0:
            recipient_id = random.choices(family_ids, weights = connection[:, doner_id], k= 1)[0]
            if doner_id != recipient_id:
                recipient = families[recipient_id]
                recipient.wealth +=  doner.wealth
                recipient.debt.append([doner, doner.wealth * (interest - 1)])
                doner.give = doner.wealth
                recipient.given += doner.wealth
                doner.wealth = 0
                connection[recipient_id, doner_id] += eta
                donation_ls.append([recipient_id, doner_id])


    for family in families:
        family.wealth += (1 + math.log(1 + family.wealth)) / family.lifetime
        family.wealth += family.give - family.given
        family.give = 0
        family.given = 0

    unpayback_ls = donation_ls[:]
    random.shuffle(family_ids)
    for family_id in family_ids:
        family = families[family_id]
        if len(family.debt) == 0:
            family.status = 1
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
                    if [family_id, owner.family_id] in unpayback_ls:
                        unpayback_ls.remove([family_id, owner.family_id])
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

    # while True:
    #     counter = 0
    #     for family in families:
    #         if family.status == 0 and family.wealth > 0:
    #             count = 0
    #             for debt in family.debt:
    #                 owner = debt[0]
    #                 debt_wealth = debt[1]
    #                 if family.wealth >= debt_wealth:
    #                     owner.wealth += debt_wealth
    #                     connection[owner.family_id, family.family_id] += eta
    #                     if family in owner.subordinates:
    #                         owner.subordinates.remove(family)
    #                     family.wealth -= debt_wealth
    #                     count += 1
    #                 else:
    #                     owner.wealth += family.wealth
    #                     connection[owner.family_id, family.family_id] += eta
    #                     debt[1] -= family.wealth
    #                     family.wealth = 0
    #                     break
    #             family.debt = family.debt[count:]
    #             if len(family.debt) == 0:
    #                 family.status = 1
    #             counter += 1
    #     if counter == 0:
    #         break
    connection = connection / np.sum(connection, axis = 0)
    if len(donation_ls) > 0:
        payback = 1  - len(unpayback_ls) / len(donation_ls)
    else:
        payback = 0


    statuses = [family.status for family in families]
    wealths = [family.wealth for family in families]
    inheritances = [family.inheritance for family in families]

    return families, connection, statuses, wealths, inheritances, payback

# families, connection = state.families, state.connection

def reproduction(families, connection):
    wealths = [family.wealth for family in families]
    wealths_ls = wealths[:]
    wealths_ls.sort()
    top_5pc = wealths_ls[round(len(families) * 0.94)]

    independent_duration, subordinate_duration, rich_duration = [], [], []

    remove_ls = []
    count = len(families)
    for family in families:
        if family.lifetime == family.counter:
            remove_ls.append(family)
            num_children = np.random.poisson(lam = 1 + math.log10(1 + family.wealth))
            # num_children = 1 + 1 * (random.random() < 0.5)
            # num_children = np.random.poisson(lam = 1 + family.wealth)
            # print(num_children)
            family_id = family.family_id
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
            if family.wealth > top_5pc:
                if family.rich_duration > 0:
                    family.rich_duration += 1
                else:
                    family.rich_duration = 1
            elif family.rich_duration > 0:
                rich_duration.append(family.rich_duration)
                family.rich_duration = 0
            if num_children > 0:
                # print(num_children)
                weights = np.exp(- family.inheritance * np.array(range(num_children)))
                weights = weights / np.sum(weights)
                weights_ori = weights[:]
                if len(weights) > 1:
                    for i in range(len(weights)):
                        if i > 0:
                            weights[i] += weights[i - 1]
                wealth = family.wealth
                subordinates = family.subordinates
                debts = family.debt
                for debt in debts:
                    debt[1] = debt[1]
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

                    # inheritance = family.inheritance + random.gauss(0, mutation)
                    # if inheritance < 0:
                    #     inheritance = 0.01
                    # lifetime = exchange + 1 * (random.random() < 0.1)
                    lifetime = max(1, np.random.poisson(lam = exchange))
                    families.append(Family(count, lifetime, cur_wealth, inheritance, 1 * (len(cur_debts) == 0), cur_debts, cur_subordinates, family.independent_duration, family.subordinate_duration, family.rich_duration))
                    for subordinate in cur_subordinates:
                        for debt in subordinate.debt:
                            if debt[0] == family:
                                debt[0] = families[-1]
                    for debt in cur_debts:
                        boss = debt[0]
                        boss.subordinates = list(set(boss.subordinates) - set([family]) | set([families[-1]]))

                    arr_to = connection[family_id] * weights_ori[i]
                    connection = np.insert(connection, count, arr_to, axis = 0)
                    arr_from = connection[:, family_id]
                    connection = np.insert(connection, count, arr_from, axis = 1)
                    connection[count, count] = 0

                    for j in range(i):
                        connection[count, count - j - 1] = eta * 3
                        connection[count - j - 1, count] = eta * 3
                    count += 1

    families = list(set(families) - set(remove_ls))
    family_ids = [family.family_id for family in families]
    connection = connection[family_ids][:, family_ids]
    connection = connection / np.sum(connection, axis = 0)

    return families, connection, independent_duration, subordinate_duration, rich_duration

# families, connection = state.families, state.connection
def mean(x):
    if len(x) > 0:
        return sum(x) / len(x)
    else:
        return 0

def main():
    states = []
    for j in range(num_states):
        families = [Family(i, max(1, np.random.poisson(lam = exchange)), 1.0, inheritance, 1, [], [], 1, 0, 0) for i in range(num_families)]
        # connection[i, j] represents the weaight of the pass from j to i.

        # connection = np.random.rand(num_families, num_families)
        connection = np.ones([num_families, num_families])
        for i in range(num_families):
            connection[i, i] = 0
        connection = connection / np.sum(connection, axis = 0)
        states.append(State(families, connection))

    last = min (500, 5 * exchange)
    independent_duration_res, subordinate_duration_res, rich_duration_res = [], [], []
    inheritance_series, corelation_series, payback_series = [], [], []
    wealth_whole, connectivity_whole = [], []
    wealth_phase_series, connectivity_phase_series = [], []
    wealth_phase_series2, connectivity_phase_series2 = [], []
    cluster_series, flow_hierarchy_series, wealth_gini_series, connectivity_gini_series, wealth_exp_series, connectivity_power_series, wealth_power_series, connectivity_exp_series, population_ratio_series = [], [], [], [], [], [], [], [], []
    for iter in range(iteration - last):
        remove_ls, duplicate_ls = [], []
        for state in states:
            state.families, state.connection, statuses, wealths, inheritances, payback = generation(state.families, state.connection)
            state.families, state.connection, independent_duration, subordinate_duration, rich_duration = reproduction(state.families, state.connection)
            population = len(state.families)
            if population < num_families / 10:
                remove_ls.append(state)
            elif population > num_families * 2:
                duplicate_ls.append(state)
            else:
                state.families, state.connection = separation(state.families, state.connection)

        for state in duplicate_ls:
            population = len(state.families)
            family_ids = list(range(population))
            random.shuffle(family_ids)
            state.families = np.array(state.families)[family_ids].tolist()
            state.connection = state.connection[family_ids, :]
            state.connection = state.connection[:, family_ids]

            n = math.floor(math.log2(population / num_families))
            k = round(len(state.families) / 2**n)
            for i in [0] * (2**n - 1):
                families = state.families[:k]
                connection = state.connection[:k, :k]
                families, connection = separation(families, connection)
                states.append(State(families, connection))
                state.families = state.families[k:]
                state.connection = state.connection[k:, k:]
            state.families, state.connection = separation(state.families, state.connection)

        if len(states) > num_states:
            random.shuffle(states)
            states = states[:num_states]

    cur_iteration = max(last, 100)
    for iter in range(cur_iteration):
        remove_ls, duplicate_ls = [], []
        wealth_ls, connection_to_ls = [], []
        inheritance_ls, corelation_res, payback_res = [], [], []
        cluster_res, flow_hierarchy_res, wealth_gini_res, connectivity_gini_res, wealth_exp_res, connectivity_exp_res, population_ratio_res = [], [], [], [], [], [], []
        for state in states:
            state.families, state.connection, statuses, wealths, inheritances, payback = generation(state.families, state.connection)
            # state.df[iteration - 100 + iter] = [statuses, wealths, connection_to]
            connection_to = np.sum(state.connection, axis = 1).tolist()
            state.families, state.connection, independent_duration, subordinate_duration, rich_duration = reproduction(state.families, state.connection)
            inheritance_ls.extend(inheritances)
            wealth_ls.extend(wealths)
            connection_to_ls.extend(connection_to)
            if iter > cur_iteration - 100:
                wealth_whole.extend(wealths)
                connectivity_whole.extend(connection_to)

            independent_duration_res.extend(independent_duration)
            subordinate_duration_res.extend(subordinate_duration)
            rich_duration_res.extend(rich_duration)

            corelation_res.append(np.corrcoef(wealths, connection_to)[0,1])
            wealth_gini_res.append(gini2(wealths))
            connectivity_gini_res.append(gini2(connection_to))
            population_ratio_res.append(mean(statuses))
            payback_res.append(payback)

            cur_connection = 1 * (state.connection > eta)
            df = pd.DataFrame(cur_connection)
            G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())
            cluster_res.append(nx.average_clustering(G))
            try:
                flow_hierarchy_res.append(nx.flow_hierarchy(G))
            except:
                pass

            if iter == cur_iteration - 1:
                for family in state.families:
                    if family.status == 1:
                        independent_duration_res.append(family.independent_duration)
                    if family.status == 0:
                        subordinate_duration_res.append(family.subordinate_duration)
                    if family.rich_duration > 0:
                        rich_duration_res.append(family.rich_duration)
                families = state.families
                connection = state.connection
                population_ratio = int(round(mean(statuses), 2) * 100)

            # state.families, state.connection = reproduction(state.families, state.connection)

            population = len(state.families)
            if population < num_families / 10:
                remove_ls.append(state)
            elif population > num_families * 2:
                duplicate_ls.append(state)
            else:
                state.families, state.connection = separation(state.families, state.connection)

        inheritance_series.append(mean(inheritance_ls))
        corelation_series.append(mean(corelation_res))
        cluster_series.append(mean(cluster_res))
        flow_hierarchy_series.append(mean(flow_hierarchy_res))
        wealth_gini_series.append(mean(wealth_gini_res))
        connectivity_gini_series.append(mean(connectivity_gini_res))
        population_ratio_series.append(mean(population_ratio_res))
        payback_series.append(mean(payback_res))


        try:
            community_sizes = np.log(np.array(wealth_ls))
            community_sizes.sort()
            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 10)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=10)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp1 = - 1 / param[0]

            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 3)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=10)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp2 = - 1 / param[0]
            wealth_exp_series.append(wealth_exp2)
            wealth_phase_series.append(wealth_exp1 / wealth_exp2)
        except:
            pass

        try:
            community_sizes = np.log(np.array(connection_to_ls))
            community_sizes.sort()
            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 10)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=10)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            connectivity_exp1 = - 1 / param[0]

            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 3)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=10)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            connectivity_exp2 = - 1 / param[0]
            connectivity_exp_series.append(connectivity_exp2)
            connectivity_phase_series.append(connectivity_exp1 / connectivity_exp2)
        except:
            pass
        # wealth_exp1 - wealth_exp2
        #
        # mse_power / mse_exp
        # plt.plot(val / np.sum(val), c = "b")
        # plt.plot(exp_predict / np.sum(exp_predict), c = "orange")
        # plt.plot(power_predict / np.sum(power_predict), c = "green")



        states = list(set(states) - set(remove_ls))
        # for state in remove_ls:
        #     states.remove(state)
        for state in duplicate_ls:
            population = len(state.families)
            family_ids = list(range(population))
            random.shuffle(family_ids)
            state.families = np.array(state.families)[family_ids].tolist()
            state.connection = state.connection[family_ids, :]
            state.connection = state.connection[:, family_ids]

            n = math.floor(math.log2(population / num_families))
            k = round(len(state.families) / 2**n)
            for i in [0] * (2**n - 1):
                families = state.families[:k]
                connection = state.connection[:k, :k]
                families, connection = separation(families, connection)
                states.append(State(families, connection))
                state.families = state.families[k:]
                state.connection = state.connection[k:, k:]
            state.families, state.connection = separation(state.families, state.connection)

        # print(duplicates)
        if len(states) > num_states:
            random.shuffle(states)
            states = states[:num_states]

    if True:
        wealth_phase, connectivity_phase = 0, 0
        try:
            community_sizes = np.log(np.array(connectivity_whole))
            community_sizes.sort()
            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 100)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=20)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            connectivity_exp1 = - 1 / param[0]

            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 30)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=20)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            connectivity_exp2 = - 1 / param[0]
            connectivity_exp = connectivity_exp2
            connectivity_phase = connectivity_exp1 / connectivity_exp2
        except:
            pass

        try:
            community_sizes = np.log(np.array(wealth_whole))
            community_sizes.sort()
            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 100)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=30)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp1 = - 1 / param[0]

            max_val = min(len(community_sizes[community_sizes > 0]), num_families * 30)
            sizes = community_sizes[::-1][: max_val]
            (val, bins)  = np.histogram(sizes, bins=30)
            bins = np.exp((bins[1:] + bins[:-1]) / 2)
            param, cov = curve_fit(linear_fit, bins, np.log(val))
            wealth_exp2 = - 1 / param[0]
            wealth_exp = wealth_exp2
            wealth_phase = wealth_exp1 / wealth_exp2
        except:
            pass



    if iter == iteration - 1 and trial < 10 and interest in [1.01, 1.1, 1.2, 1.5] and exchange in [1, 10, 100]:
        independents, subordinates, wealths  = [], [], []
        for id in range(num_families):
            wealths.append(families[id].wealth)
            if families[id].status == 1:
                independents.append(id)
            else:
                subordinates.append(id)

        wealths = np.array(wealths)

        statuses_c = []
        for family in families:
            if family.status == 1:
                statuses_c.append("b")
            else:
                statuses_c.append("m")

        cur_connection = 1 * (connection > eta)
        # cur_connection = 1 * (connection > 0.08)
        df = pd.DataFrame(cur_connection)
        G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        nx.draw_networkx(G, node_size = 10, with_labels=False, ax = ax, node_color = statuses_c)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=18)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_eta_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_ls, bins = 50, density = 1)
        ax.set_xlim(0, max(connection_to_ls) + 3)
        ax.set_xlabel("connectivity",fontsize=24)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=18)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connectivity_log_{population_ratio}pc.pdf")
        plt.close('all')

        max_val = max(10, max(connection_to_ls))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_ls, bins = np.logspace(0, np.log10(max(connection_to_ls)), 20), density = 1)
        ax.set_xlabel("connectivity",fontsize=24)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, max_val + 10)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=22)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connectivity_log_log_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealth_ls, bins = 50, density = 1)
        ax.set_xlabel("wealth",fontsize=24)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=18)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_wealth_log_{population_ratio}pc.pdf")
        plt.close('all')

        max_val = max(10, max(wealth_ls))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealth_ls, bins = np.logspace(0, np.log10(max(wealth_ls)), 30), density = 1)
        ax.set_xlabel("wealth",fontsize=24)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, max_val + 10)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=22)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_wealth_log_log_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connectivity_whole, bins = 50, density = 1)
        ax.set_xlim(0, max(connectivity_whole) + 3)
        ax.set_xlabel("connectivity",fontsize=24)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=18)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_2_connectivity_log_{population_ratio}pc.pdf")
        plt.close('all')

        max_val = max(10, max(connectivity_whole))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connectivity_whole, bins = np.logspace(0, np.log10(max(connectivity_whole)), 50), density = 1)
        ax.set_xlabel("connectivity",fontsize=24)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, max_val + 10)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=22)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_2_connectivity_log_log_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealth_whole, bins = 50, density = 1)
        ax.set_xlabel("wealth",fontsize=24)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=18)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_2_wealth_log_{population_ratio}pc.pdf")
        plt.close('all')

        max_val = max(10, max(wealth_whole))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealth_whole, bins = np.logspace(0, np.log10(max(wealth_whole)), 50), density = 1)
        ax.set_xlabel("wealth",fontsize=24)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, max_val + 10)
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=22)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_2_wealth_log_log_{population_ratio}pc.pdf")
        plt.close('all')

    # res = [100 - mean(population_ratio_series[-last:]) * 100, mean(inheritance_series[-last:]), mean(flow_hierarchy_series[-last:]), mean(cluster_series[-last:]),mean(corelation_series[-last:]), mean(independent_duration_res), mean(subordinate_duration_res), mean(rich_duration_res), mean(connectivity_exp_series[-last:]), mean(connectivity_power_series[-last:]), mean(connectivity_gini_series[-last:]),  mean(wealth_exp_series[-last:]), mean(wealth_power_series[-last:]), mean(wealth_gini_series[-last:]), mean(payback_series[-last:]), mean(wealth_phase_series[-last:]),mean(connectivity_phase_series[-last:]), wealth_exp, wealth_power, wealth_phase, connectivity_exp, connectivity_power, connectivity_phase]
    res = [100 - mean(population_ratio_series[-last:]) * 100, mean(inheritance_series[-last:]), mean(flow_hierarchy_series[-last:]), mean(cluster_series[-last:]),mean(corelation_series[-last:]), mean(independent_duration_res), mean(subordinate_duration_res), mean(rich_duration_res), mean(connectivity_exp_series[-last:]), mean(connectivity_power_series[-last:]), mean(connectivity_gini_series[-last:]),  mean(wealth_exp_series[-last:]), mean(wealth_power_series[-last:]), mean(wealth_gini_series[-last:]), mean(payback_series[-last:]), np.median(wealth_phase_series[-last:]),np.median(connectivity_phase_series[-last:]), wealth_phase, connectivity_phase]

    return res

num_states = 50
num_families = 100
iteration = 100
iteration = 1500
trial = 0
interest = 1.5
feedback = 0.3
inheritance = 0.0
eta = 0.05
eta = 0.1
exchange = 10
exchange = 100
trial = 0

# trial = 3
# trial
# trial = 1
# for trial in range(10):
    # iteration = 50 * exchange + 100
#     path = f"4_{num_states}states_{num_families}fam_{exchange}exchange_eta{round(eta * 100)}pc_interest{round(interest * 1000)}pm_inheritance{round(inheritance * 100)}pc"
#     res = main()
#     params = [exchange, eta, interest, num_states, num_families]
#     params.extend(res)
#     df_res[len(df_res.columns)] = params
#     print(df_res)
# interest = [1.01, 1.1, 1.2][int(sys.argv[1])]
# for exchange in [1, 10, 100]:
#     iteration = 50 * exchange
#     path = f"{num_states}states_{num_families}fam_{exchange}exchange_eta{round(eta * 100)}pc_interest{round(interest * 1000)}pm_inheritance{round(inheritance * 100)}pc"
#     res = main()
#     params = [exchange, eta, interest, num_states, num_families]
#     params.extend(res)
#     df_res[len(df_res.columns)] = params
#     print(df_res)
# df_res

# # inheritance = [0.0, 1.0][int(sys.argv[1]) // 2 % 2]
# interest = [1.0001, 1.001, 1.01, 1.1, 1.2][int(sys.argv[1]) // 5 % 5]
interest = [1.0, 1.001, 1.01, 1.1, 1.2, 1.5, 2.0][int(sys.argv[1]) // 7 % 7]
for exchange in [[8, 10], [1, 20], [5, 30], [3, 50], [2, 80], [100], [200]][int(sys.argv[1]) % 7]:
    iteration = 50 * exchange
    # df_res = pd.DataFrame(index = ["exchange", "eta", "interest", "num_states", "num_families", "population_ratio", "inheritance", "hierarchy", "cluster_index", "corrcoef", "independent_duration", "subordinate_duration", "rich_duration", "connectivity_exp", "connectivity_power", "connectivity_gini", "wealth_exp", "wealth_power", "wealth_gini", "payback", "wealth_phase", "connectivity_phase", "wealth_exp_whole", "wealth_power_whole", "wealth_phase_whole", "connectivity_exp_whole", "connectivity_power_whole", "connectivity_phase_whole"])
    df_res = pd.DataFrame(index = ["exchange", "eta", "interest", "num_states", "num_families", "population_ratio", "inheritance", "hierarchy", "cluster_index", "corrcoef", "independent_duration", "subordinate_duration", "rich_duration", "connectivity_exp", "connectivity_power", "connectivity_gini", "wealth_exp", "wealth_power", "wealth_gini", "payback", "wealth_phase", "connectivity_phase", "wealth_phase2", "connectivity_phase2"])
    # path = f"{num_states}states_{num_families}fam_{exchange}exchange_eta{round(eta * 100)}pc_interest{round(interest * 1000)}pm_inheritance{round(inheritance * 100)}pc"
    path = f"{num_states}states_{num_families}fam_{exchange}exchange_eta{round(eta * 100)}pc_interest{round(interest * 1000)}pm_inheritance{round(inheritance * 100)}pc"
    for trial in range(30):
        try:
            res = main()
            params = [exchange, eta, interest, num_states, num_families]
            params.extend(res)
            df_res[len(df_res.columns)] = params
        except:
            pass
    df_res.to_csv(f"res/res_{path}.csv")
