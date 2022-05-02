import pandas as pd
import scipy.stats
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

def define_measures(df):
    df.replace(88, np.nan, inplace = True)
    df.replace(99, np.nan, inplace = True)

    # df["SCCS1721"].replace(21, 15, inplace = True)
    # df["SCCS1721"].replace(22, 25, inplace = True)
    # df["SCCS1723"].replace(21, 15, inplace = True)
    # df["SCCS1723"].replace(22, 25, inplace = True)
    # df["SCCS1724"].replace(21, 15, inplace = True)
    # df["SCCS1724"].replace(22, 25, inplace = True)

    df[["-SCCS756", "-SCCS762", "-SCCS787", "-SCCS763", "-SCCS784", "-SCCS764"]] = - df[["SCCS756", "SCCS762", "SCCS787", "SCCS763", "SCCS784", "SCCS764"]]

    values = df[["SCCS568", "-SCCS784", "SCCS973", "SCCS1736", "SCCS1733"]]
    values = (values - values.mean()) / values.std(ddof=0)
    df["gift"] = values.T.mean()

    # values = df[["SCCS1723", "SCCS1724", "SCCS1721", "SCCS274"]]
    # values = (values - values.mean()) / values.std(ddof=0)
    # df["economic hierarchy"] = values.T.mean()

    values = df[["SCCS1723", "SCCS1724", "SCCS1721"]]
    values = (values - values.mean()) / values.std(ddof=0)
    df["economic hierarchy"] = values.T.mean()

    values = df[["SCCS91", "-SCCS762", "SCCS158"]]
    # values = df[["SCCS91", "SCCS83", "-SCCS756", "-SCCS762", "SCCS89"]]
    values = (values - values.mean()) / values.std(ddof=0)
    df["structural hierarchy"] = values.T.mean()

    return df



def correlation_analysis(data_pivot, target_col):
    var_whole.index = var_whole["id"]
    df_structure = data_pivot.copy()
    id_ls = var_whole["id"].tolist()
    id_ls.append(target_col)
    df_structure = df_structure[df_structure.columns & id_ls]

    df_structure.replace(88, np.nan, inplace = True)
    df_structure.replace(99, np.nan, inplace = True)

    res = pd.DataFrame(index = ["corr.", "p"])
    for col in df_structure.columns:
        df2 = df_structure[[target_col, col]].dropna()
        x = df2[target_col].values
        y = df2[col].values
        a, b = spearmanr(np.ravel(x), np.ravel(y))
        if b > 0:
            res[col] = [a, b]

    df_res0 = pd.DataFrame()
    df_res0[["corr.", "p"]] = res.T[["corr.", "p"]]
    df_res0["title"] = var_whole.loc[df_res0.index].title
    df_res0["abs. corr."] = abs(res.T["corr."])
    df_res0 = df_res0.sort_values("abs. corr.", ascending =  False)
    df_res0["ratio"] = df_structure.isnull().sum()
    df_res0["ratio"] = round((len(df_structure.index) - df_res0["ratio"]) / len(df_structure.index), 2)
    df_agg = df_res0[df_res0["ratio"] > 0.2]
    df_agg = df_agg.reindex(columns = ["title", "corr.", "p", "ratio", "abs. corr."])
    df_agg.to_csv(f"variables/variables_high_corr_{target_col}_SCCS.csv")

def rand_jitter(arr):
    stdev = .02 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def disparity_plot(df):
    sns.set(style='whitegrid')

    # df["flag"] = 1 * (df["economic hierarchy"] > -1) * (df["structural hierarchy"] > -1)
    # df = df[df["flag"] == 1]
    df["economic hierarchy"] = df["economic hierarchy"] - df["economic hierarchy"].min()
    df["structural hierarchy"] = df["structural hierarchy"] - df["structural hierarchy"].min()
    df["economic hierarchy"] = df["economic hierarchy"] / df["economic hierarchy"].max()
    df["structural hierarchy"] = df["structural hierarchy"] / df["structural hierarchy"].max()
    df["gift"] = df["gift"] - df["gift"].min()
    df["gift"] = df["gift"] / df["gift"].max()
    df["flag"] = 1

    bins = list(np.arange(0.0, 1.1, 0.1))
    bins[0] = bins[0] - 0.01
    bins[-1] = bins[-1] + 0.01
    N = len(bins)
    structural_res = []
    economic_res = []
    hierarchies = pd.DataFrame(index = ["bin", "disparity", "hierarchy"])
    for i in range(N - 1):
        structural_res.append(df[(df["gift"] >= bins[i] - 0.05) & (df["gift"] < bins[i + 1] - 0.05)]["structural hierarchy"].mean())
        economic_res.append(df[(df["gift"] >= bins[i] - 0.05) & (df["gift"] < bins[i + 1] - 0.05)]["economic hierarchy"].mean())
        hierarchies[len(hierarchies.columns)] = [bins[i], economic_res[-1], "economic"]
        hierarchies[len(hierarchies.columns)] = [bins[i], structural_res[-1], "structural"]


    hierarchies = hierarchies.T

    fig = plt.figure()
    ax = fig.add_subplot()
    # sns.barplot(bins[:-1], structural_res, ax = ax, color = "orange", alpha = 0.5)
    # sns.barplot(bins[:-1], economic_res, ax = ax, color = "b", alpha = 0.5, multiple="dodge")
    ax = sns.barplot(data = hierarchies, x = "bin", y = "disparity", hue = "hierarchy", dodge = True)
    ax.set_xlabel("gift", fontsize=20)
    ax.set_ylabel("disparity", fontsize=20)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelsize=12)
    # ax.set_aspect('equal', adjustable='box')
    ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"figs/SCCS_gift_disparity_hist_bin.pdf", bbox_inches='tight')
    plt.close('all')

    bins = list(np.arange(0.0, 1.1, 0.1))
    bins[0] = bins[0] - 0.01
    bins[-1] = bins[-1] + 0.01
    N = len(bins)
    structural_res = []
    economic_res = []
    hierarchies = pd.DataFrame(index = ["bin", "disparity", "hierarchy"])
    for i in range(N - 1):
        structural_res.append(df[(df["gift"] >= bins[i]) & (df["gift"] < bins[i + 1])]["structural hierarchy"].mean())
        economic_res.append(df[(df["gift"] >= bins[i]) & (df["gift"] < bins[i + 1])]["economic hierarchy"].mean())
        hierarchies[len(hierarchies.columns)] = [bins[i], economic_res[-1], "economic"]
        hierarchies[len(hierarchies.columns)] = [bins[i], structural_res[-1], "structural"]


    hierarchies = hierarchies.T

    fig = plt.figure()
    ax = fig.add_subplot()
    # sns.barplot(bins[:-1], structural_res, ax = ax, color = "orange", alpha = 0.5)
    # sns.barplot(bins[:-1], economic_res, ax = ax, color = "b", alpha = 0.5, multiple="dodge")
    ax = sns.barplot(data = hierarchies, x = "bin", y = "disparity", hue = "hierarchy", dodge = True)
    ax.set_xlabel("gift", fontsize=20)
    ax.set_ylabel("disparity", fontsize=20)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ax.set_xticklabels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelsize=12)
    # ax.set_aspect('equal', adjustable='box')
    ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"figs/SCCS_gift_disparity_hist.pdf", bbox_inches='tight')
    plt.close('all')

    bins = list(np.arange(0, 1.05, 0.15))
    bins[0] = bins[0] - 0.01
    bins[-1] = bins[-1] + 0.01
    N = len(bins)
    structural_res = []
    economic_res = []
    hierarchies = pd.DataFrame(index = ["bin", "disparity", "hierarchy"])
    for i in range(N - 1):
        structural_res.append(df[(df["gift"] >= bins[i]) & (df["gift"] < bins[i + 1])]["structural hierarchy"].mean())
        economic_res.append(df[(df["gift"] >= bins[i]) & (df["gift"] < bins[i + 1])]["economic hierarchy"].mean())
        hierarchies[len(hierarchies.columns)] = [bins[i], economic_res[-1], "economic"]
        hierarchies[len(hierarchies.columns)] = [bins[i], structural_res[-1], "structural"]


    hierarchies = hierarchies.T

    fig = plt.figure()
    ax = fig.add_subplot()
    # sns.barplot(bins[:-1], structural_res, ax = ax, color = "orange", alpha = 0.5)
    # sns.barplot(bins[:-1], economic_res, ax = ax, color = "b", alpha = 0.5, multiple="dodge")
    ax = sns.barplot(data = hierarchies, x = "bin", y = "disparity", hue = "hierarchy", dodge = True)
    ax.set_xlabel("gift", fontsize=20)
    ax.set_ylabel("disparity", fontsize=20)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_xticklabels([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9])
    ax.tick_params(labelsize=12)
    # ax.set_aspect('equal', adjustable='box')
    ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"figs/SCCS_gift_disparity_hist2.pdf", bbox_inches='tight')
    plt.close('all')

    # df["economic hierarchy2"] = rand_jitter(df[df["flag"] == 1]["economic hierarchy"])
    # df["structural hierarchy2"] = rand_jitter(df[df["flag"] == 1]["structural hierarchy"])
    df["economic hierarchy2"] = rand_jitter(df[df["economic hierarchy"] > -1]["economic hierarchy"])
    df["structural hierarchy2"] = rand_jitter(df[df["structural hierarchy"] > -1]["structural hierarchy"])

    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(data = df[df["flag"] == 1],x = "gift", y = "economic hierarchy2", ax = ax, s = 80)
    sns.scatterplot(data = df[df["flag"] == 1],x = "gift", y = "structural hierarchy2", ax = ax, s = 80)
    ax.set_xlabel("gift", fontsize=20)
    ax.set_ylabel("disparity", fontsize=20)
    ax.set_xlim(-0.1, 1.1)
    ax.tick_params(labelsize=12)
    # ax.set_aspect('equal', adjustable='box')
    # ax.get_legend().remove()
    fig=ax.get_figure()
    plt.tight_layout()
    fig.savefig(f"figs/gift_disparity_SCCS.pdf", bbox_inches='tight')
    plt.close('all')


    return df

sccs_dir = "SCCS"
data_whole = pd.read_csv(os.path.join(sccs_dir, "data.csv"))
var_whole = pd.read_csv(os.path.join(sccs_dir,"variables.csv"))
society_whole = pd.read_csv(os.path.join(sccs_dir,"societies.csv"))
societies = society_whole[["id", "pref_name_for_society", "alt_names_by_society", "Lat", "Long"]]


data_pivot = data_whole.pivot_table(index = "soc_id", columns = "var_id", values="code")

# data_pivot = gift_degree(data_pivot)
# correlation_analysis(data_pivot)

data_pivot = define_measures(data_pivot)
data_pivot = disparity_plot(data_pivot)
for target_col in ["gift", "gift1", "gift2"]:
    correlation_analysis(data_pivot, target_col)

societies.index = societies["id"]
data_pivot[["id", "pref_name_for_society", "alt_names_by_society", "Lat", "Long"]] = societies
society_ls = data_pivot[["pref_name_for_society", "alt_names_by_society", "Lat", "Long", "gift", "economic hierarchy", "structural hierarchy"]]
society_ls.replace(np.nan, "---", inplace = True)
society_ls.to_csv("society_ls.csv")


data_pivot[data_pivot["economic hierarchy"] > 0]
data_pivot[(data_pivot["structural hierarchy"] > 0)]
data_pivot[(data_pivot["structural hierarchy"] > -100) | (data_pivot["economic hierarchy"] > -100)]
