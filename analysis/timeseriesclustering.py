import pandas as pd
import numpy as np
from ema_workbench import load_results, save_results
from ema_workbench.analysis import clusterer

path = '../results/'
file_name = '2000_scen__40_reps__0612'
file_ext  = '.tar.gz'

experiments, outcomes = load_results(path+file_name+file_ext)

oois = ['Household Population', 'GDP', 'Gini Coefficient']

MIN_K = 2
MAX_K = 21 
all_clusters = {ooi:{} for ooi in oois}
explained_variances = {ooi:{} for ooi in oois}
delta_EVs = {ooi:{} for ooi in oois}

for ooi in oois:
    # Calculate TS distances
    data = outcomes[ooi]
    distances = clusterer.calculate_cid(data)

    # Calculate overall centroid
    overall_centroid = outcomes[ooi].mean(axis=0)

    # Calculate sum mean squared error for total dataset
    assert(len(outcomes[ooi][0]) == len(overall_centroid))
    MSEs_all = [np.power(run - overall_centroid, 2).mean() for run in outcomes[ooi]]

    SMSE_all = sum(MSEs_all)

    # Test across K values (candidate #s of clusters)
    prev_EV = 0
    for K in range(MIN_K, MAX_K): 
        # Compute clusters
        all_clusters[ooi][K] = clusterer.apply_agglomerative_clustering(distances, n_clusters=K)

        # Calculate within-cluster error (sum mean squared error)
        runs_by_cluster = [[] for clust in range(K)]
        for n, run in enumerate(outcomes[ooi]):
            clust = all_clusters[ooi][K][n]
            runs_by_cluster[clust].append(run)
        runs_by_cluster = [np.array(runs) for runs in runs_by_cluster]

        SMSEs_within = []
        for clust in range(K):
            runs = runs_by_cluster[clust]
            clust_centroid = runs.mean(axis=0)
            MSEs_within = [np.power(run - clust_centroid, 2).mean() for run in runs]
            SMSEs_within.append(sum(MSEs_within))

        explained_variances[ooi][K] = 1 - (sum(SMSEs_within) / SMSE_all)

        delta_EVs[ooi][K] = explained_variances[ooi][K] - prev_EV
        prev_EV = explained_variances[ooi][K]

    # Save memory
    del distances

# Convert (Delta) Expected Variance data format
index = [K for K in range(MIN_K, MAX_K)]

EV_lists = {ooi:[EV for _, EV in explained_variances[ooi].items()] for ooi in oois}
delta_EV_lists = {ooi:[d_EV for _, d_EV in delta_EVs[ooi].items()] for ooi in oois}

EV_lists['K'] = delta_EV_lists['K'] = index

EV_df = pd.DataFrame(EV_lists)
delta_EV_df = pd.DataFrame(delta_EV_lists)

# Save values for plotting elbow plots locally
EV_df.to_csv(path + 'cluster_evs/' + file_name + 'EV.csv')
delta_EV_df.to_csv(path + 'cluster_evs/' + file_name + 'delta_EV.csv')

# Select optimal K and save clusters
for ooi in oois:
    for K in range(MIN_K+1, MAX_K):
        if delta_EVs[ooi][K] < 0.05:
            experiments[f"Cluster ({ooi})"] = all_clusters[ooi][K-1].astype("object")
            break

results = experiments, outcomes
save_results(results, path+file_name+'__with_clusters'+file_ext)
