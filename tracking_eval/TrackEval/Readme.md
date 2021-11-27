# TrackEval on JRDB:

We deleted some code that is not used in our evaluation, added OSPA metric and 3D tracking evaluations. In general, our format follows KITTI tracking dataset. 

For 2D tracking, run: python scripts/run_jrdb.py

For 3D tracking, run: python scripts/run_jrdb_3d.py

You can specific the path of your groundtruth and pred in trackeval/dataset/jrdb_2d_train.py and trackeval/dataset/jrdb_3d_train.py 

In general the previous command are all you need to evaluate, for more details, please read original [TrackEval](https://github.com/JonathonLuiten/TrackEval).
