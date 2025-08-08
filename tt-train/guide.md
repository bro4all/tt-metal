## Using the cluster

Exabox Slurm User Key should be shared with you. Use that for logging into the login pod

```
ssh -p 32222 root@metal-wh-09
```

You can use `metal-wh-10` or `metal-wh-11` too, it all points to the same login pod

`/data` hosts a shared mount across the login and compute pods

Make a folder for yourself on it

```
# mkdir -p /data/<your-name>
```

Clone and do all your builds inside the above folder

Git clones and builds will be slower than what you're normally used to (disk on a local node)

Right now, there is just one slurm node partition named debug that has 3 nodes
```
# sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
debug        up   infinite      3   idle debug-[0-2]
all*         up   infinite      3   idle debug-[0-2]
```

The 3 nodes are connected like this
```
09 <--> 10
|
<-----> 11
```
and these are their roles
```
11 - compute
09 - aggregator
10 - optimizer
```

If you want to manually reset all 3 nodes at once `srun -N 3 -l /data/tt-smi/.venv/bin/tt-smi -r`

The following 2 files from this branch are useful for submitting jobs
- `3tier.sh` - this is a slurm batch job submit script. this defines a bunch of params about the job to send slurm, sets some global environment vars (common across all nodes) and uses mpirun to launch the below script
- `launch.sh` - this is the script that gets run on each node eventually. So anything that you want to do differently on each node, you do here (like setting a different env var or launch command for each node)

You submit a job via
```
# sbatch 3tier.sh 
Submitted batch job 45
```

You can check it's status using 
```
# squeue
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
   45     debug nano_gpt     root  R       0:16      3 debug-[0-2]
```

The job's stdout and stderr should be written to files in the current folder
```
tail -f nano_gpt_45.out
```
or
```
tail -f nano_gpt_45.err
```