import numpy as np
from sklearn.cluster import KMeans
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
import matplotlib.pyplot as plt

def calculate_probabilities(X,labels,weights,nclust):
    mu, stds, ws = np.zeros(nclust), np.zeros(nclust), np.zeros(nclust)
    for i in range(nclust):
        weights_clust = weights[labels == i]
        samples = X[labels == i]
        mu[i] = np.sum(weights_clust * samples) / np.sum(weights_clust)
        stds[i] = np.sqrt(np.sum(weights_clust * (samples-mu[i])**2) / np.sum(weights_clust))
        ws[i] =  np.sum(weights_clust)
    return mu, stds, ws / ws.sum()


def calculate_probabilities_2d(X,labels,weights,nclust):
    mu, covs, ws = np.zeros([nclust, 2]), np.zeros([nclust, 2, 2]), np.zeros(nclust)
    for i in range(nclust):
        weights_clust = weights[labels == i][:, np.newaxis]
        samples = X[labels == i]
        mu[i] = np.sum(weights_clust * samples, axis=0) / np.sum(weights_clust)
        covs[i] = np.cov(samples.T, aweights=weights_clust.flatten())
        ws[i] =  np.sum(weights_clust)
    return mu, covs, ws / ws.sum()


def plot_pmf(X,
             weights,
             nclust,
             ax=None,
             color='k',
             label=None,
             gmm=True,
             kT=2.4943395,
             cmap='coolwarm',
             levels=np.linspace(0, 40, 20),
             verbose=False):

    # if X is 1D data
    if X.shape[1] == 1:
        if verbose:
            print('X.shape', X.shape)
        if gmm:
            print('Using GMM')
            model_parameters = []
            clf = KMeans(n_clusters=nclust).fit(X,
                                    weights)
            mu, stds, mod_weights = calculate_probabilities(X, clf.labels_, weights, nclust)
            for i in range(nclust):
                model_parameters.append(Normal(means=[mu[i]], covs=[stds[i]], covariance_type='diag'))
            
            model = GeneralMixtureModel(model_parameters, mod_weights)

            model.fit(X=X, sample_weight=weights)

            """ Calculate Free Energy diagram """
            xrange = np.max(X) - np.min(X)
            xlim = [np.min(X), np.max(X)]
            X_samp = np.linspace(xlim[0],xlim[1],1000)[:, np.newaxis]
            P_X = model.probability(X_samp).numpy()
            dG = kT * -np.log(P_X/np.max(P_X))
            if ax is not None:
                ax.plot(X_samp, dG, lw=3, color=color, label=label)
            return X_samp, dG
        else:
            print('Using histogram')
            # use histogram
            X = X.T[0]
            hist, bin_edges = np.histogram(X, bins=100, weights=weights, density=True)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            dG = kT * -np.log(hist/np.max(hist))
            if ax is not None:
                ax.plot(bin_centers, dG, lw=3, color=color, label=label)
            return bin_centers, dG
    elif X.shape[1] == 2:
        if verbose:
            print('X.shape', X.shape)
        if gmm:
            print('Using GMM')
            X_standard = (X - X.mean(axis=0)) / X.std(axis=0)
            clf = KMeans(n_clusters=nclust).fit(X_standard,
                                    weights)
            if verbose:
                fig, ax_cluster = plt.subplots(figsize=(7,9))
                ax_cluster.scatter(X.T[0], X.T[1], c=clf.labels_, cmap='tab20')
                ax_cluster.set_xlabel('CV1')
                ax_cluster.set_ylabel('CV2')
                ax_cluster.set_title('Clustering')

            mu, covs, mod_weights = calculate_probabilities_2d(X, clf.labels_, weights, nclust)
            if verbose:
                print('mu', mu)
            model_parameters = []
            for i in range(nclust):
                model_parameters.append(Normal(means=mu[i], covs=covs[i], covariance_type='full'))

            model = GeneralMixtureModel(model_parameters, mod_weights)

            model.fit(X, weights)
            
            """ Calculate Free Energy diagram """
            CV1_minmax = [X.T[0].min(), X.T[0].max()]
            CV2_minmax = [X.T[1].min(), X.T[1].max()]
            X_grid, Y_grid = np.meshgrid(np.linspace(CV1_minmax[0], CV1_minmax[1], 1000),
                                        np.linspace(CV2_minmax[0], CV2_minmax[1], 1000))
            X_samp = np.vstack([X_grid.flatten(), Y_grid.flatten()]).T

            P_X = model.probability(X_samp).numpy()
            dG = kT * -np.log(P_X/np.max(P_X))
            if ax is not None:
                mappable = ax.contourf(X_grid,
                            Y_grid,
                            dG.reshape(X_grid.shape),
                            levels=levels,
                            cmap=cmap)
                plt.colorbar(mappable, label='Free Energy (kJ/mol)')
            return X_grid, Y_grid, dG
        else:
            print('Using histogram')
            # use histogram
            hist, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=100, weights=weights, density=True)
            xcenters = (xedges[1:] + xedges[:-1]) / 2
            ycenters = (yedges[1:] + yedges[:-1]) / 2
            dG = kT * -np.log(hist/np.max(hist))
            dG = dG.T
            if ax is not None:
                mappable = ax.contourf(xcenters, ycenters, dG,
                            levels=levels,
                            cmap=cmap)
                plt.colorbar(mappable, label='Free Energy (kJ/mol)')
            return xcenters, ycenters, dG
    else:
        raise ValueError('X must be 1D or 2D array')