# RLMAX

A Rocket League physics simulator for NVIDIA cards: thousands of games at once, on the GPU.

## Results

Measured 5 November 2026 with release 0.1.1 on an RTX 5090 (Linux), against RocketGPU 22a1fa7 and RocketSim v3 0.3.5, the latest of each that day.

- **Faster in 29 of 31 configurations**: up to 2.0x at large batches. The one loss is 3v3 at 1,024 arenas, by 2.3%.
- **Less GPU memory in every configuration**: 48% less for 1v1 at 1,024 arenas.
- **Exact on the strict accuracy suite**: 121 recorded game situations, 12,038,088 bodies stepped from the game's full state, 100% on both the CPU and the card.
- **More accurate from a public start state**: 90.1% of segments exact, against 81.9% for RocketGPU.

Release 0.1.2 is also exact on the suite's newer version: 144 recordings, 16 of them stepped from controls alone, with the arena running its own boost pads, demolitions, respawns and kickoff countdown. It has not been re-timed on the 5090 yet. On an RTX 3080 at 8,192 arenas it is 1.7% slower than 0.1.1 for 1v1 and 1.9% faster for 3v3.

Speed, millions of arena-ticks per second (higher is better), 8 ticks per decision, median of 3 seeds:

| Arenas | 1v1 RLMAX | 1v1 RocketGPU | 3v3 RLMAX | 3v3 RocketGPU |
|---:|---:|---:|---:|---:|
| 1,024 | **5.72** | 5.21 | 3.40 | **3.48** |
| 4,096 | **15.81** | 13.08 | **8.47** | 6.78 |
| 8,192 | **24.34** | 17.70 | **10.72** | 8.15 |
| 16,384 | **30.87** | 20.10 | **11.15** | 8.40 |
| 65,536 | **40.27** | 20.28 | **16.06** | 7.86 |

Peak GPU memory, MiB (lower is better):

| Arenas | 1v1 RLMAX | 1v1 RocketGPU | 3v3 RLMAX | 3v3 RocketGPU |
|---:|---:|---:|---:|---:|
| 1,024 | **836** | 1,614 | **1,188** | 1,786 |
| 8,192 | **1,370** | 2,354 | **2,412** | 3,384 |
| 65,536 | **7,062** | 7,276 | **15,598** | 16,290 |

Accuracy against recorded games, % of car-ticks within tolerance (higher is better):

| Recording | RLMAX | RocketGPU | RocketSim v3 |
|---|---:|---:|---:|
| Nexto 1v1 | **99.633** | 99.483 | 96.547 |
| Ripple | **99.752** | 99.438 | 98.285 |
| Partycannon | **98.900** | 98.648 | 96.277 |
| Daizen 2v2 | 99.504 | 99.504 | 97.113 |
| Veai 3v3 | 99.875 | 99.875 | 97.907 |
| Wisp | 99.976 | 99.976 | 98.956 |

The card and the CPU build give identical results, bit for bit, from 64 to 65,536 arenas.

## You need

- An NVIDIA card with a current driver. The build compiles for the card it finds.
- A CUDA toolkit: 12.4 or later, 12.8 or later for an RTX 50.
- Rust 1.95 or later, and on Windows the Visual Studio build tools with clang-cl.
- The soccar collision meshes from your own copy of the game. They aren't included.

## Build and run

```sh
mkdir -p ~/rlmax-data/meshes/soccar && cp /path/to/soccar/*.cmf ~/rlmax-data/meshes/soccar/
export RLMAX_DATA=~/rlmax-data RLMAX_CUDA_PATH=/usr/local/cuda-13.2
cargo run --release -p rlmax --example speed
```

The first build takes a long time and about 7 GB of RAM at its largest step. To make it quicker:

- `RLMAX_NVCC_SPLIT=2` compiles more than one kernel at once, in about the same RAM. On a 24-thread desktop with a training run beside it, the full build took 34 minutes this way; 0.1.1 took 61, one kernel at a time. Raise it if you have cores and RAM to spare.
- `RLMAX_NCS=2` builds only the 1v1 kernels (`2,6` for 1v1 and 3v3). A batch of any other size then fails to load.
- Compiled kernels are kept in `$RLMAX_DATA/kernel_cache`, so rebuilding the same source, or another project that links RLMAX, reuses them. `RLMAX_KERNEL_CACHE=off` turns it off.

The speed test prints millions of arena-ticks per second for 1v1, 3v3 and a mix. Keep the card otherwise idle while it runs. To pick sizes, add them to the command: `-- 1024 8192`.

## Check it

`cargo test --release -p rlmax equals_cpu_every_word -- --nocapture --test-threads=1` runs the card and the CPU build side by side. Every line should say `0 of`.

## Use it

`rlmax::gpu::GpuBatch` holds the arenas on the card and `rlmax::cpu::Arena` runs one on the CPU. `crates/rlmax/examples/speed.rs` shows the whole loop in about 100 lines.

## Other cards

Card settings are chosen automatically. To build for a card that isn't in this machine, set `RLMAX_CUDA_ARCHS`, for example `86` for an RTX 30 or `120` for an RTX 50.
