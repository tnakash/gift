import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import math
import random
import networkx as nx
from scipy.optimize import curve_fit
import networkx.algorithms.community as nx_comm



if int(sys.argv[1])==0:
    if not os.path.exists("./figs"):
        os.mkdir("./figs")
        os.mkdir("./figs/graph")
        os.mkdir("./figs/timeseries")
    if not os.path.exists("./res"):
        os.mkdir("./res")


def linear_fit(x, a, b):
    return a * x + b

class State:
    def __init__(self, families, connection):
        self.families = families
        self.connection = connection
        # self.df = pd.DataFrame()

class Family:
    def __init__(self, family_id, wealth, inheritance, gift_ratio, status, debt, subordinates, independent_duration, subordinate_duration, rich_duration):
        self.family_id = family_id
        self.wealth = wealth
        self.inheritance = inheritance
        self.gift_ratio = gift_ratio
        self.status = status
        self.debt = debt
        self.subordinates = subordinates
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


def generation(families, connection):
    for i in range(len(families)):
        families[i].family_id = i
    if exchange == 0:
        for family in families:
            family.wealth += 1 + math.log(1 + family.wealth * feedback)
            # if birth == "linear":
            #     family.wealth += 1 + math.log(1 + family.wealth * feedback)
            # else:
            #     family.wealth += 1 + family.wealth * feedback

    else:
        for e in range(exchange):
            independents = [family.family_id for family in families if family.status == 1]
            # independents = [family.family_id for family in families]
            random.shuffle(independents)
            if len(independents) > 1:
                for doner_id in independents:
                    doner = families[doner_id]
                    if random.random() > exploration:
                        recipient_id = random.choices(independents, weights = connection[independents, doner_id], k= 1)[0]
                    else:
                        recipient_id = random.choice(list(set(independents) - set([doner_id])))
                    if doner_id != recipient_id and doner.wealth * doner.gift_ratio > epsilon:
                        recipient = families[recipient_id]
                        doner.wealth -= epsilon
                        recipient.wealth +=  doner.wealth * doner.gift_ratio
                        recipient.debt.append([doner, doner.wealth * (interest - 1) * doner.gift_ratio])
                        doner.give = doner.wealth * doner.gift_ratio
                        recipient.given += doner.wealth * doner.gift_ratio
                        doner.wealth = doner.wealth * (1 - doner.gift_ratio)
                        connection[recipient_id, doner_id] += eta


            for family in families:
                production = (1 + math.log(1 + family.wealth * feedback)) / exchange
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

            family_ids = list(range(len(families)))
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
            # connection = connection / np.sum(connection, axis = 0)

            while True:
                counter = 0
                for family in families:
                    if family.status == 0 and family.wealth > 0:
                        count = 0
                        for debt in family.debt:
                            owner = debt[0]
                            debt_wealth = debt[1]
                            if family.wealth >= debt_wealth:
                                owner.wealth += debt_wealth
                                connection[owner.family_id, family.family_id] += eta
                                if family in owner.subordinates:
                                    owner.subordinates.remove(family)
                                family.wealth -= debt_wealth
                                count += 1
                            else:
                                owner.wealth += family.wealth
                                connection[owner.family_id, family.family_id] += eta
                                debt[1] -= family.wealth
                                family.wealth = 0
                                break
                        family.debt = family.debt[count:]
                        if len(family.debt) == 0:
                            family.status = 1
                        counter += 1
                if counter == 0:
                    break
    connection = connection / np.sum(connection, axis = 0)

    statuses = []
    num_subordinates =  []
    wealths = []
    connection_to = np.sum(connection, axis = 1)
    inheritances = []
    gift_ratios = []
    for family in families:
        wealths.append(family.wealth)
        statuses.append(family.status)
        num_subordinates.append(len(family.subordinates))
        inheritances.append(family.inheritance)
        gift_ratios.append(family.gift_ratio)

    wealths_ls = wealths[:]
    wealths_ls.sort()
    top_5pc = wealths_ls[round(len(families) * 0.94)]

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
        if family.wealth > top_5pc:
            if family.rich_duration > 0:
                family.rich_duration += 1
            else:
                family.rich_duration = 1
        elif family.rich_duration > 0:
            rich_duration.append(family.rich_duration)
            family.rich_duration = 0

    return families, connection, statuses, num_subordinates, wealths, inheritances, connection_to, independent_duration, subordinate_duration, rich_duration, gift_ratios

# families, connection = state.families, state.connection

def reproduction(families, connection):
    # wealths = [1 + family.wealth * feedback for family in families]
    # births = Counter(random.choices(list(range(num_families)), weights = wealths, k = num_families))
    births = [np.random.poisson(lam = 1 + math.log(1 + family.wealth * feedback)) for family in families]
    count = len(families)
    next_families = []

    for family_id in range(len(families)):
        family = families[family_id]
        num_children = births[family_id]
        if num_children > 0:
            weights = np.exp(- family.inheritance * np.array(range(num_children)))
            weights = weights / np.sum(weights)
            weights_ori = weights[:]
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

                inheritance = family.inheritance + random.gauss(0, mutation)
                gift_ratio = family.gift_ratio + random.gauss(0, mutation)
                if inheritance < 0:
                    inheritance = 0.01
                if gift_ratio < 0:
                    gift_ratio = 0.0
                elif gift_ratio > 1:
                    gift_ratio = 1.0
                gift_ratio = family.gift_ratio
                next_families.append(Family(count - len(families), cur_wealth, inheritance, gift_ratio, 1 * (len(cur_debts) == 0), cur_debts, cur_subordinates, family.independent_duration, family.subordinate_duration, family.rich_duration))
                for subordinate in cur_subordinates:
                    for debt in subordinate.debt:
                        if debt[0] == family:
                            debt[0] = next_families[-1]
                for debt in cur_debts:
                    boss = debt[0]
                    boss.subordinates = list(set(boss.subordinates) - set([family]) | set([next_families[-1]]))

                arr_to = connection[family_id] * np.random.normal(1.0, 0.01, count) * weights_ori[i]
                # arr_to = connection[family_id] * np.random.normal(1.0, 0.01, count)
                connection = np.insert(connection, count, arr_to, axis = 0)
                arr_from = connection[:, family_id] * np.random.normal(1.0, 0.01, count + 1)
                connection = np.insert(connection, count, arr_from, axis = 1)
                # arr_to = connection[family_id]
                # connection = np.insert(connection, count, arr_to, axis = 0)
                # arr_from = connection[:, family_id]
                # connection = np.insert(connection, count, arr_from, axis = 1)
                connection[count, count] = 0

                for j in range(i):
                    # connection[count, count - j - 1] = 1 / num_families / num_children
                    # connection[count - j - 1, count] = 1 / num_families / num_children
                    connection[count, count - j - 1] = eta
                    connection[count - j - 1, count] = eta
                count += 1

    connection = connection[len(families):, len(families):]
    connection = connection / np.sum(connection, axis = 0)
    families = next_families[:]

    return families, connection

def gini2(y):
    y.sort()
    n = len(y)
    nume = 0
    for i in range(n):
        nume = nume + (i+1)*y[i]

    deno = n*sum(y)
    return ((2*nume)/deno - (n+1)/(n))*(n/(n-1))

def gini_plot(families, connection):
    wealth_ginis, connection_ginis = [], []
    for i in range(len(families)):
        families[i].family_id = i
    for e in range(exchange):
        independents = [family.family_id for family in families if family.status == 1]
        # independents = [family.family_id for family in families]
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
                if doner_id != recipient_id and doner.wealth * doner.gift_ratio > epsilon:
                    recipient = families[recipient_id]
                    doner.wealth -= epsilon
                    recipient.wealth +=  doner.wealth * doner.gift_ratio
                    recipient.debt.append([doner, doner.wealth * (interest - 1) * doner.gift_ratio])
                    doner.give = doner.wealth * doner.gift_ratio
                    recipient.given += doner.wealth * doner.gift_ratio
                    doner.wealth = doner.wealth * (1 - doner.gift_ratio)
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

        family_ids = list(range(len(families)))
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

        wealth_ls = [family.wealth for family in families]
        connection_ls = np.sum(connection, axis = 1).tolist()
        wealth_ginis.append(gini2(wealth_ls))
        connection_ginis.append(gini2(connection_ls))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(wealth_ginis, color = "b")
    ax.plot(connection_ginis, color = "orange")
    ax.set_xlabel("exchange",fontsize=18)
    ax.set_ylabel("gini coef.",fontsize=18)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(f"figs/{path}_{trial}_gini.pdf")
    plt.close('all')


# families, connection = state.families, state.connection


def main():
    states = []
    for j in range(num_states):
        families = [Family(i, 1.0, 1.0, 1.0, 1, [], [], 1, 0, 0) for i in range(num_families)]
        # connection[i, j] represents the weaight of the pass from j to i.

        connection = np.random.rand(num_families, num_families)
        for i in range(num_families):
            connection[i, i] = 0
        connection = connection / np.sum(connection, axis = 0)
        states.append(State(families, connection))

    average_cluster_res, num_subordinates_res, wealths_res, connection_to_res, flow_hierarchy_res, population_ratio_res, gift_ratio_res = [], [], [], [], [], [], []
    independent_duration_res, subordinate_duration_res, rich_duration_res, inheritance_res = [], [], [], []
    gift_ratio_plot , inheritance_plot = [], []
    duplicates = 0
    flag = 0
    # exp1_ls, exp2_ls, exp3_ls = [], [], []
    # connection_to2_ls, connection_to_ls, wealths_ls = [], [], []
    for iter in range(iteration):
        remove_ls, duplicate_ls = [], []
        for state in states:
            state.families, state.connection, statuses, num_subordinates, wealths, inheritances, connection_to, independent_duration, subordinate_duration, rich_duration, gift_ratios = generation(state.families, state.connection)
            # state.df[iteration - 100 + iter] = [statuses, num_subordinates, wealths, connection_to]
            inheritance_plot.append(inheritances)
            gift_ratio_plot.append(gift_ratios)
            if iter > iteration * 0.5:
                num_subordinates_res.extend(num_subordinates)
                wealths_res.extend(wealths)
                connection_to_res.extend(connection_to)
                inheritance_res.extend(inheritances)
                wealths = np.array(wealths)
                independent_duration_res.extend(independent_duration)
                subordinate_duration_res.extend(subordinate_duration)
                rich_duration_res.extend(rich_duration)
                population_ratio = int(round(sum(statuses) / len(statuses), 2) * 100)
                population_ratio_res.append(population_ratio)
                gift_ratio_res.extend(gift_ratios)
                # if connection_to.max() > 2 * num_families:
                #     flag = 1
                #     print(len(state.families))
                #     break

                cur_connection = 1 * (state.connection > eta)
                df = pd.DataFrame(cur_connection)
                G = nx.from_pandas_adjacency(df.T, create_using=nx.DiGraph())

                average_cluster_res.append(nx.average_clustering(G))
                try:
                    flow_hierarchy_res.append(nx.flow_hierarchy(G))
                except:
                    pass

                if iter == iteration - 1:
                    for family in state.families:
                        if family.status == 1:
                            independent_duration_res.append(family.independent_duration)
                        if family.status == 0:
                            subordinate_duration_res.append(family.subordinate_duration)
                        if family.rich_duration > 0:
                            rich_duration_res.append(family.rich_duration)
                    families_ = state.families
                    connection_ = state.connection

            state.families, state.connection = reproduction(state.families, state.connection)

            population = len(state.families)
            if population < num_families / 10:
                remove_ls.append(state)
            elif population > num_families * 2:
                duplicate_ls.append(state)
            else:
                state.families, state.connection = separation(state.families, state.connection)

        # if flag == 1:
        #     break
        for state in remove_ls:
            states.remove(state)
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
                duplicates += 1

        # print(duplicates)
        if len(states) > num_states:
            random.shuffle(states)
            states = states[:num_states]

    if True:
        # connection_to_res_OLD = connection_to_res
        community_sizes = np.array(connection_to_res)
        community_sizes.sort()
        sizes = community_sizes[::-1][num_families // 10  * num_states: num_families * 10 * num_states]
        ranks = np.arange(community_sizes.size)[num_families // 10  * num_states: num_families * 10 * num_states]
        param, cov = curve_fit(linear_fit, sizes, np.log(ranks))
        connection_exp = - param[0]
        param, cov = curve_fit(linear_fit, np.log(sizes), np.log(ranks))
        connection_power = - param[0]

        # wealths_res_OLD = wealths_res
        community_sizes = np.array(wealths_res)
        community_sizes = community_sizes[community_sizes > 0]
        community_sizes.sort()
        sizes = community_sizes[::-1][num_families // 10  * num_states: num_families * 10 * num_states]
        ranks = np.arange(community_sizes.size)[num_families // 10  * num_states: num_families * 10 * num_states]
        param, cov = curve_fit(linear_fit, sizes, np.log(ranks))
        wealth_exp = - param[0]
        param, cov = curve_fit(linear_fit, np.log(sizes), np.log(ranks))
        wealth_power = - param[0]


    if trial < 3:
        if exchange > 0:
            gini_plot(state.families, state.connection)
        independents, subordinates, wealths  = [], [], []
        for id in range(num_families):
            wealths.append(families[id].wealth)
            if families[id].status == 1:
                independents.append(id)
            else:
                subordinates.append(id)

        wealths = np.array(wealths)
        if interest == 1.1 and trial == 0:
            my_gift = []
            my_inheritance = []
            for i in range(iteration):
                my_gift.extend([[i, gift] for gift in gift_ratio_plot[i]])
                my_inheritance.extend([[i, inheritance] for inheritance in inheritance_plot[i]])
            df_order = pd.DataFrame(my_gift, columns=["time","value"])
            df_separation = pd.DataFrame(my_inheritance, columns=["time","value"])

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            # ax.scatter(df_order["time"][0], df_order["value"][0], s=5, c='blue', label = r"$\lambda$")
            # ax.scatter(df_separation["time"][0], df_separation["value"][0], s=5, c='red', label = r"$s$")
            ax.scatter(df_order["time"], df_order["value"],s=10, cmap='Blues', label = "")
            ax.scatter(df_separation["time"], df_separation["value"],s=10, cmap=sns.light_palette("red", as_cmap=True), label = "")
            # ax.scatter(df_separation["time"], df_separation["value"],s=1, c=df_separation["hist"], cmap=sns.light_palette("red", as_cmap=True))
            ax.set_xlabel("time",fontsize=20)
            ax.set_ylabel("value",fontsize=20)
            ax.set_ylim(-0.1,2.0)
            ax.tick_params(labelsize=14)
            fig.tight_layout()
            fig.savefig(f"figs/{path}_{trial}_timeseries.pdf")
            plt.close('all')

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
        for family in families_:
            if family.status == 1:
                statuses.append("b")
            else:
                statuses.append("m")

        cur_connection = 1 * (connection_ > eta)
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


        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_res, bins = 50, density = 1)
        ax.set_xlabel("connection_to",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_to_log_{population_ratio}pc.pdf")
        plt.close('all')

        connection_to_res = np.array(connection_to_res)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(connection_to_res[connection_to_res > 1], bins = 50, density = 1)
        ax.set_xlabel("connection_to",fontsize=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_connection_to_log_log_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealths_res, bins = 50, density = 1)
        ax.set_xlabel("wealth",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_wealth_log_{population_ratio}pc.pdf")
        plt.close('all')

        wealths_res = np.array(wealths_res)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(wealths_res[wealths_res > 1], bins = 50, density = 1)
        ax.set_xlabel("wealth",fontsize=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_wealth_log_log_{population_ratio}pc.pdf")
        plt.close('all')

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.hist(num_subordinates_res, bins = 10, density = 1)
        ax.set_xlabel("subordinates",fontsize=18)
        ax.set_yscale('log')
        # ax.set_ylabel(r"$\lambda_i$",fontsize=18)
        # ax.set_ylim(-0.1,1.5)
        ax.tick_params(labelsize=14)
        fig.tight_layout()
        fig.savefig(f"figs/{path}_{trial}_per_subordinates_{population_ratio}pc.pdf")
        plt.close('all')

    res = [100 - np.mean(np.array(population_ratio_res)), np.mean(np.array(inheritance_res)), np.mean(np.array(flow_hierarchy_res)), np.mean(np.array(average_cluster_res)), np.corrcoef(wealths_res, connection_to_res)[0,1], np.mean(np.array(independent_duration_res)), np.mean(np.array(subordinate_duration_res)), np.mean(np.array(rich_duration_res)), connection_exp, connection_power,  wealth_exp, wealth_power, np.mean(np.array(gift_ratio_res))]

    return res


num_states = 30
num_families = 100
iteration = 1000
mutation = 0.03
trial = 0
interest = 1.1
decay = 0.5
feedback  = 0.3
eta = 0.03
ratio = 0.7
epsilon = 0.01
exploration = 0.1
exchange = 30
trial = 0
birth = "linear"
distribution = "exp"
eta_ = 3

for num_families in [30, 50, 100]:
    for interest in [[1.0, 3.0], [1.1, 2.0], [1.2, 1.5]][int(sys.argv[1]) // 9 % 3]:
        # for exchange in [[1, 2, 3, 5, 8, 10, 20, 30, 50, 80, 100, 200, 300][int(sys.argv[1]) % 3]]:
        for exchange in [[0, 50], [1, 30], [2, 20], [3, 10], [5, 8], [80], [100], [200], [300]][int(sys.argv[1]) % 9]:
            df_res = pd.DataFrame(index = ["exchange", "decay", "exploration",  "eta", "feedback", "epsilon", "mutation", "interest", "num_states", "num_families", "distribution", "population_ratio", "inheritance", "hierarchy", "cluster_index", "corrcoef", "independent_duration", "subordinate_duration", "rich_duration", "connection exp", "connection power", "wealth exp", "wealth power", "gift_ratio"])
            eta = eta_ / 100
            path = f"dist{distribution}_birth{birth}_{num_states}states_{num_families}fam_{exchange}exchange_d{round(decay * 100)}pc_ex{round(exploration * 100)}pc_eta{round(eta_ * 100)}pc_f{round(feedback * 100)}pc_epsilon{round(epsilon * 100000)}pm_mu{round(mutation * 1000)}pm_interest{round(interest * 10)}pd"
            for trial in range(100):
                try:
                    res = main()
                    params = [exchange, decay, exploration, eta, feedback, epsilon, mutation, interest, num_states, num_families, distribution]
                    params.extend(res)
                    df_res[len(df_res.columns)] = params
                except:
                    pass
            df_res.to_csv(f"res/res_{path}.csv")
