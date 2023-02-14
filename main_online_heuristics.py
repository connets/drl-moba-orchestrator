from mec_moba.policy_models.mip_online import compute_online_mip_solution
from mec_moba.policy_models.best_fit import compute_best_fit_solution

if __name__ == '__main__':
    for seed in [1000]:
        #compute_online_mip_solution(seed, n_games_per_epoch=6000)  # , n_games_per_epoch=7200)

        compute_best_fit_solution(seed, n_games_per_epoch=6000)
