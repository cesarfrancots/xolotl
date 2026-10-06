# Xolotl

A small terminal coding agent for GLM 5.3 Flash (Z.ai), written in Rust. The command is `xol`.

## Install

```
cargo install --path . --locked
```

This puts `xol` in `~/.cargo/bin`, which `rustup` already added to your PATH. Run the same command again after pulling changes to update it.

Set your Z.ai key once as a user environment variable named `ZAI_API_KEY`. `xol` also finds it there on Windows when a terminal was opened before you set it.

## Use

```
xol                      # interactive, in the current directory
xol --cwd D:\some\repo   # interactive, in another project
xol -p "fix the failing test" --mode yolo   # one task, headless (JSON events on stdout)
xol -p "..." --deadline 840   # also tell it the run is killed after 840 s: it lowers its reasoning effort, adds time notes as the budget runs down, and cuts a thought that runs far too long without acting
xol --version
```

Inside the REPL:
- `/help` lists the slash commands.
- `/mode ask|writes|yolo` sets what needs your approval (default `writes`: edits and shell commands ask first).
- `/compact` summarizes old history.
- Ctrl+C cancels the current step. `/exit` or Ctrl+D quits.

Sessions are kept under `~/.xolotl/projects/<project>/sessions/` (`events.jsonl` is the full trace).

## Benchmarks

- `bench/`: the local 12-task suite (`py bench/run.py`)
- `bench/tb2/`: Terminal-Bench 2.0 via Harbor (`py bench/tb2/run.py`)

Each has its own README and a `results/HISTORY.md`.
