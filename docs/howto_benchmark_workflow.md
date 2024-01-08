# How To Run the Benchmark and Update the Leaderboard Workflow

### Running the benchmark

Detailed documentation about setting up the virtual environment, installing libraries, and more details about the content below is provided in [howto_run_benchmark](https://github.com/clembench/clembench/blob/main/docs/howto_run_benchmark.md).

The benchmark is run for a particular game with a particular model -- for example, the taboo game on GPT-3.5-turbo -- using the following command:  

```
python3 scripts/cli.py -m gpt-3.5-turbo--gpt-3.5-turbo run taboo
```

Or run the following command to run all existing games on the chosen model (GPT-3.5-turbo):

```
python3 scripts/cli.py -m gpt-3.5-turbo--gpt-3.5-turbo run all
```

Alternatively, the benchmark run can be scripted as follows to run multiple games and model combinations using the following bash script (here, only GPT-3.5-turbo and GPT-4 are chosen as references):

```
#!/bin/bash
# Preparation: ./setup.sh
echo
echo "==================================================="
echo "PIPELINE: Starting"
echo "==================================================="
echo
game_runs=(
  # Single-player: privateshared
  "privateshared gpt-3.5-turbo"
  "privateshared gpt-4"
  # Single-player: wordle
  "wordle gpt-3.5-turbo"
  "wordle gpt-4"
  # Single-player: wordle_withclue
  "wordle_withclue gpt-3.5-turbo"
  "wordle_withclue gpt-4"
  # Multi-player taboo
  "taboo gpt-3.5-turbo--gpt-3.5-turbo"
  "taboo gpt-4--gpt-4"
  # Multi-player referencegame
  "referencegame gpt-3.5-turbo--gpt-3.5-turbo"
  "referencegame gpt-4--gpt-4"
  # Multi-player imagegame
  "imagegame gpt-3.5-turbo--gpt-3.5-turbo"
  "imagegame gpt-4--gpt-4"
  # Multi-player wordle_withcritic
  "wordle_withcritic gpt-3.5-turbo--gpt-3.5-turbo"
  "wordle_withcritic gpt-4--gpt-4"
)
total_runs=${#game_runs[@]}
echo "Number of benchmark runs: $total_runs"
current_runs=1
for run_args in "${game_runs[@]}"; do
  echo "Run $current_runs of $total_runs: $run_args"
  bash -c "./run.sh ${run_args}"
  ((current_runs++))
done
echo "==================================================="
echo "PIPELINE: Finished"
echo "==================================================="
```

Once the benchmark runs are finished, the `results` folder will include all run files organized under specific model, game, episode, and experiment folders.

### Transcribe and Score Dialogues

Run the following command to generate transcriptions of the dialogues. The script generates LaTeX and HTML files of the dialogues under each episode of a particular experiment.

```
python3 scripts/cli.py transcribe all
```

Next, run the scoring command that calculates turn & episode-specific metrics defined for each game. This script generates `scores.json` file stored under the same folder as transcripts and other files under a specific episode. 

```
python3 scripts/cli.py score all
```

### Evaluate the Benchmark Run & Update the Leaderboard

#### Benchmark-Runs

All benchmark run files (experiment/episode files, transcripts, scores) are shared publicly on the [clembench-runs](https://github.com/clembench/clembench-runs). After a certain run is finished, the resulting files are added to the repository under a certain version number. 

#### Versioning the benchmark

The set of game realisations (i.e., prompts + game logic) defines the **major version number**. That is, any time the set of games is changed, a new version number is required -- and all included models need to be re-run against the benchmark.
Result scores are generally not expected to be comparable across major versions of the benchmark.


The set of instances of each realisation defines the **minor number**. For example, when new target words for the *wordle* game are selected, this constitutes a new minor version of the benchmark. Results scores are likely comparable across minor version updates, although whether this really is the case is an empirical question that needs more thorough investigation.

#### Evaluation

Once a certain version is identified, the resulting files of the benchmark run are added under this version. It means that a folder with the new version is created under `clembench-runs` and all files from the benchmark run are copied there, e.g. [v1.0](https://github.com/clembench/clembench-runs/tree/main/v1.0)

If the version already exists (meaning the benchmark run is evaluating a new model using existing games and their instances) then the resulting files from the benchmark run need to be merged with the existing one. Merging here stands for copying the model-specific folder under `results` of the benchmark run into the existing version game folder. E.g., copy `results/imagegame/records/CodeLlama-34b-Instruct-hf-t0.0--CodeLlama-34b-Instruct-hf-t0.0` into `v1.0/imagegame/records/` (assuming the new run evaluated CodeLlama-34b-Instruct-hf that did not exist before).

Once the files are under a certain version folder, run the benchmark evaluation command by providing the path of the files for that specific version:

```
python3 evaluation/bencheval.py --results_path <PATH>
```

The variable `<PATH>` is the absolute path to all files of a certain version, e.g. `../clembench-runs/v1.0`. If the script above results in `No Module Found Error` then run first `source prepare_path.sh` and try again.


This script above generates `results.csv`and `results.html` files of the benchmark run. All benchmark files, along with these two generated results files, need to be pushed into the `clembench-runs` repository under the assigned version.

#### Leaderboard

The repository `clembench-leaderboard`(https://github.com/clembench/clembench-leaderboard) is a Gradio app that automatically pulls results files from the `clembench-runs` repository and serves the leaderboard. In order for the automatic deployment to work, the [`benchmark_runs.json`](https://github.com/clembench/clembench-runs/blob/main/benchmark_runs.json) file includes the existing versions and the resulting CSV files. If a new version of the benchmark is added, then this information needs to be appended to the existing versions.

```
{
   "versions":[
      {
         "version":"v0.9",
         "result_file":"v0.9/results.csv"
      },
      {
         "version":"v1.0",
         "result_file":"v1.0/results.csv"
      }
   ]
}
```



The leaderboard is currently deployed under [HuggingFace Space CLEM-Leaderboard](https://huggingface.co/spaces/colab-potsdam/clem-leaderboard). The leaderboard will be automatically updated once new versions of `results.csv` are available under `clembench-runs`.