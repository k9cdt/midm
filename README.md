## Mutual Information Distance Matrix

Generate a Mutual Information Distance (MID) Matrix from simulation timeseries.

Initially this was created with KDE to compute join probability distribution. 
This turned out to be a terrible idea, as finite bandwidth will give non-zero off-diagonal terms,
leading to non-zero self-MID. 
Decreasing bandwidth to smaller than one grid width will make this effectively a histogram-based method.
Despite this, the was written with CUDA so it should be fairly fast in all conditions.