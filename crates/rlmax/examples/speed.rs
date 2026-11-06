//! Speed test: 450 decisions of 8 ticks, one random action per car per decision, 1v1, 3v3 and a mix of the two.
//! `cargo run --release -p rlmax --example speed -- 1024 8192`; no sizes runs 1,024 to 65,536.
use rlmax::cpu::{Arena, ArenaData};
use rlmax::driver::{Buffer, Device};
use rlmax::gpu::{GpuArenaData, GpuBatch};
use std::time::Instant;

const DECISIONS: u64 = 450;
const TICKS: u32 = 8;

fn table() -> Vec<f32> {
    let mut t = Vec::new();
    for throttle in [-1.0f32, 0.0, 1.0] {
        for steer in [-1.0f32, 0.0, 1.0] {
            for boost in [0.0f32, 1.0] {
                for handbrake in [0.0f32, 1.0] {
                    if boost == 0.0 || throttle == 1.0 {
                        t.extend([throttle, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                    }
                }
            }
        }
    }
    for pitch in [-1.0f32, 0.0, 1.0] {
        for yaw in [-1.0f32, 0.0, 1.0] {
            for roll in [-1.0f32, 0.0, 1.0] {
                for jump in [0.0f32, 1.0] {
                    for boost in [0.0f32, 1.0] {
                        if (jump == 1.0 && yaw != 0.0) || (pitch == 0.0 && roll == 0.0 && jump == 0.0) {
                            continue;
                        }
                        let handbrake = f32::from(jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0));
                        t.extend([boost, yaw, pitch, yaw, roll, jump, boost, handbrake]);
                    }
                }
            }
        }
    }
    t
}

fn row(x: u64) -> u8 {
    let mut x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (((x >> 32) * 90) >> 32) as u8
}

struct Run {
    batch: GpuBatch,
    rows: Vec<Vec<u8>>,
    buf: Buffer<u8>,
    goals: Buffer<u32>,
    arenas: u64,
}

fn run(dev: &Device, gdata: &GpuArenaData, table: &Buffer<f32>, cars: u32, arenas: usize) -> Result<Run, Box<dyn std::error::Error>> {
    let words: Vec<u32> = (0..arenas).flat_map(|_| Arena::new(cars).words().to_vec()).collect();
    let mut batch = GpuBatch::new(dev, cars, &words)?;
    let n = arenas * cars as usize;
    let rows: Vec<Vec<u8>> = (0..DECISIONS).map(|d| (0..n as u64).map(|i| row(d << 32 ^ i)).collect()).collect();
    let buf = Buffer::from_slice(dev, &rows[0])?;
    let goals = Buffer::from_slice(dev, &vec![0u32; arenas])?;
    batch.kickoff_all(1)?;
    for r in rows.iter().take(32) {
        buf.write(r)?;
        batch.decide(gdata, TICKS, &buf, table, &goals)?;
    }
    dev.synchronize()?;
    batch.kickoff_all(1)?;
    dev.synchronize()?;
    Ok(Run { batch, rows, buf, goals, arenas: arenas as u64 })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sizes: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("arena count")).collect();
    let sizes = if sizes.is_empty() { vec![1024, 2048, 4096, 8192, 16384, 32768, 65536] } else { sizes };
    let dev = Device::open(0)?;
    let data = ArenaData::soccar_from_env()?;
    let gdata = GpuArenaData::upload(&dev, &data)?;
    let flat = table();
    let tb = Buffer::from_slice(&dev, &flat)?;
    println!("million arena-ticks per second\narenas\t1v1\t3v3\tmix");
    for n in sizes {
        print!("{n}");
        for mix in [&[(2u32, 1usize)][..], &[(6, 1)], &[(2, 2), (6, 2)]] {
            let mut runs = Vec::new();
            for &(cars, part) in mix {
                runs.push(run(&dev, &gdata, &tb, cars, n / part)?);
            }
            let t0 = Instant::now();
            for d in 0..DECISIONS as usize {
                for r in runs.iter_mut() {
                    r.buf.write(&r.rows[d])?;
                    r.batch.decide(&gdata, TICKS, &r.buf, &tb, &r.goals)?;
                }
            }
            dev.synchronize()?;
            let total: u64 = runs.iter().map(|r| r.arenas).sum::<u64>() * DECISIONS * TICKS as u64;
            print!("\t{:.2}", total as f64 / t0.elapsed().as_secs_f64() / 1e6);
        }
        println!();
    }
    Ok(())
}
