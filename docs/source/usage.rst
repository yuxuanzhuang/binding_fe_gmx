=====
Usage
=====

This is an example to analyze a 2D AWH ensemble simulation.

.. code-block:: python

    import numpy as np
    import fe_gmx
    from fe_gmx import AWH_Ensemble, AWH_2D_Ensemble
    import matplotlib.pyplot as plt
    import seaborn as sns


    # Create an AWH ensemble with two CVs.
    awh_ensemble = AWH_2D_Ensemble('../AWH_2D_PORE_ANGLE', regenerate_awh=True)

    # print metadata about whether the ensemble is out of initial stage
    print(awh_ensemble.awh_log[0])

    # plot PMF from last timestep
    time = awh_ensemble.awh_results.timeseries[-1]
    awh_pmf = awh_ensemble.awh_results.pmf[-1]

    awh_cv1 = awh_pmf.T[0][0]
    awh_cv2 = awh_pmf[0].T[1]
    awh_fes = awh_pmf[:,:,2].T

    fig, ax = plt.subplots(figsize=(7,9))
    mappable = ax.contourf(
                awh_cv1,
                awh_cv2,
                awh_fes,
    #                vmax=100,
                levels=20)
    plt.colorbar(mappable)
    plt.show()

.. code-block:: python

    # access pullx data
    # it is stored inside a pandas dataframe
    # awh_ensemble.awh_pullx[i].data

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_palette('Reds', n_colors=5)
    fig, ax = plt.subplots(figsize=(8,4))

    # dim_1 is always time
    # dim_2 will be pull_coord_1
    # and so on
    for awh_pullx in awh_ensemble.awh_pullx:
        sns.lineplot(x='dim_1', y='dim_2', data=awh_pullx.data)