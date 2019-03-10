
`energypy` uses `yaml` config files to setup experiments.  A single *experiment* is composed of multiple *runs*.  

```
example of the config file

```

This config file is passed directly into the experiment CLI, which uses `click`:

```bash
$ ep-expt expt_config run
```

Where you store these config files is up to you - `~/ep-configs/my_experiment.yaml` is suggested.  Note that a copy of the config file will be copied into the results directory.

`energypy` saves results from experiments into a folder in your `~` directory (i.e. `$HOME` on Unix machines):

```bash
~/ep-results/my_experiment/...
```


## TODO

Pretraining from existing memory (see old expt.py)

## TESTS

Check seed by rolling out envs / randomly selecting actions

Check that configs get through correctly (check both class and the dumped config)
