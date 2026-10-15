// Tunes the Vulkan backend's tiled products (`cleave-tile-matmuls-for-gpu`)
// on nanoLM's product shapes (`bench/gpu_matmul/matmul.cleave`): each
// configuration `bm,bn,bk,tm,tn,split,vec` forced in turn
// (`CLEAVE_GPU_MATMUL`, applied to every product it fits), the kernels'
// device time per shape read from `CLEAVE_VK_PROFILE`, the fastest
// configuration kept for each shape and compared with the compiler's own
// choice. The device's clock varies: every configuration runs once, each
// shape's few fastest again (the least of `RUNS` runs). First the tilings, with
// each choice of vector loads (`vec`: bit 1 the operands whose rows are
// contiguous, bit 2 those whose `k` is), with the compiler's split, then each shape's
// best with every split. Writes the
// winners that beat it to `target/gpu_matmul_table.txt`, one line per shape:
// `m n k a_k_contiguous b_k_contiguous bm bn bk tm tn split vec`; with
// `--install`, to the compiler's own table too (`cleave-mlir/cpp/
// matmul_table.inc`, which the compiler then uses: rebuild it).
//
//   rustc --edition 2024 -O bench/gpu_matmul/tune.rs -o target/gpu_matmul_tune.exe
//   target/gpu_matmul_tune.exe [path/to/cleave.exe] [--install]      (from the repository root)
use std::collections::HashMap;
use std::process::Command;

/// How many times `matmul.cleave` calls each shape (its `REPS`).
const REPS: f64 = 20.0;
/// Runs per finalist (each shape's `FINALISTS` fastest after one run of
/// every configuration), the least time of each kept.
const RUNS: usize = 3;
const FINALISTS: usize = 3;

/// (function in `matmul.cleave`, m, n, k, whether `k` is contiguous in A
/// (row-major `m x k`) and in B (`n x k`, transposed)): the product of an
/// m x k by a k x n.
const SHAPES: &[(&str, i64, i64, i64, bool, bool)] = &[
    ("qkvo_fwd", 1024, 768, 768, true, false),
    ("gate_up", 1024, 2048, 768, true, false),
    ("down", 1024, 768, 2048, true, false),
    ("head", 1024, 4096, 768, true, true),
    ("head_dx", 1024, 768, 4096, true, false),
    ("head_dw", 4096, 768, 1024, false, false),
    ("dw_qkvo", 768, 768, 1024, false, false),
    ("dw_up", 768, 2048, 1024, false, false),
    ("dx_up", 1024, 768, 2048, true, true),
    ("attn_qk", 1024, 1024, 64, true, true),
    ("attn_pv", 1024, 64, 1024, true, false),
    ("muon_xxt", 768, 768, 768, true, true),
    ("muon_wide", 768, 768, 2048, true, true),
];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Config {
    bm: i64,
    bn: i64,
    bk: i64,
    tm: i64,
    tn: i64,
    split: i64,
    vec: i64,
}

impl Config {
    fn env(&self) -> String {
        format!("{},{},{},{},{},{},{}", self.bm, self.bn, self.bk, self.tm, self.tn, self.split, self.vec)
    }
    /// The compiler's `ProductTiling::fits`.
    fn fits(&self, m: i64, n: i64, k: i64) -> bool {
        if self.bm % self.tm != 0 || self.bn % self.tn != 0 {
            return false;
        }
        let threads = (self.bm / self.tm) * (self.bn / self.tn);
        let shared = 4 * self.bk * ((self.bm + 4) + (self.bn + 4));
        m % self.bm == 0
            && n % self.bn == 0
            && k % self.bk == 0
            && (32..=1024).contains(&threads)
            && (self.bm * self.bk) % threads == 0
            && (self.bn * self.bk) % threads == 0
            && shared <= 48 * 1024
            && (self.split <= 1 || k % (self.split * self.bk) == 0)
    }
}

/// Device microseconds per call of each function, summed over its kernels,
/// and the value the program prints, running `matmul.cleave` with `config`
/// forced (or the compiler's own choice).
fn measure(cleave: &str, config: Option<&Config>) -> Option<(HashMap<String, f64>, f64)> {
    let mut command = Command::new(cleave);
    command
        .args(["--backend", "vulkan", "--no-openmp", "--no-debug-info", "--run", "bench/gpu_matmul/matmul.cleave"])
        .env("CLEAVE_VK_PROFILE", "1")
        // Without the compiler's table: the reference is its own heuristic.
        .env("CLEAVE_GPU_MATMUL_TABLE", "target/gpu_matmul_empty_table.txt");
    if let Some(config) = config {
        command.env("CLEAVE_GPU_MATMUL", config.env());
    }
    let output = command.output().unwrap_or_else(|e| panic!("cannot run {cleave}: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.contains("main returned: 0") {
        eprintln!("  {config:?} failed: {}", String::from_utf8_lossy(&output.stderr).lines().last().unwrap_or(""));
        return None;
    }
    let check: f64 = stdout.lines().next()?.trim().parse().ok()?;
    let mut times = HashMap::new();
    for line in String::from_utf8_lossy(&output.stderr).lines() {
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.len() < 8 || words[1] != "ms" || words[4] != "x" || words[6] != "us" {
            continue;
        }
        let (Ok(dispatches), Ok(mean)) = (words[3].parse::<f64>(), words[5].parse::<f64>()) else { continue };
        let Some(function) = words[7].strip_suffix("_kernel") else { continue };
        *times.entry(function.to_string()).or_insert(0.0) += mean * dispatches / REPS;
    }
    Some((times, check))
}

fn main() {
    let install = std::env::args().any(|a| a == "--install");
    let cleave = std::env::args()
        .skip(1)
        .find(|a| a != "--install")
        .unwrap_or_else(|| "target/release/cleave.exe".to_string());
    std::fs::write("target/gpu_matmul_empty_table.txt", "").expect("cannot write an empty table");
    let (baseline, expected) = measure(&cleave, None).expect("the compiler's own choice fails");
    // Each shape's best: (microseconds, configuration).
    // Each shape's fastest so far, at most `keep` of them: (microseconds,
    // configuration).
    type Ranking<'a> = HashMap<&'a str, Vec<(f64, Config)>>;
    let mut tried = 0;
    let mut run = |config: &Config, runs: usize, keep: usize, ranking: &mut Ranking| {
        if !SHAPES.iter().any(|&(_, m, n, k, _, _)| config.fits(m, n, k)) {
            return;
        }
        tried += 1;
        let mut times: HashMap<String, f64> = HashMap::new();
        let mut check = f64::NAN;
        for _ in 0..runs {
            let Some((run, value)) = measure(&cleave, Some(config)) else { return };
            check = value;
            for (function, us) in run {
                let least = times.entry(function).or_insert(f64::MAX);
                *least = least.min(us);
            }
        }
        if (check - expected).abs() > 1e-4 * expected.abs().max(1.0) {
            eprintln!("  {config:?}: {check}, not {expected}");
            return;
        }
        for &(function, m, n, k, _, _) in SHAPES {
            let Some(&us) = times.get(function) else { continue };
            if !config.fits(m, n, k) {
                continue;
            }
            let entries = ranking.entry(function).or_default();
            entries.push((us, *config));
            entries.sort_by(|a, b| a.0.total_cmp(&b.0));
            entries.truncate(keep);
        }
        eprint!("\r  {tried} configurations tried");
    };
    // The tilings, the compiler's split, once each.
    let mut tilings = Vec::new();
    // Where the winners were on an RTX 4070 SUPER over a wider grid (tiles
    // up to 256, blocks from 2, every `vec`): tiles of 64 and 128, blocks of
    // 4 and 8, `vec` 2 alone never ahead. Each configuration costs a compile
    // and the driver building its kernels (about 1.5 s): the grid kept small.
    for bm in [32, 64, 128] {
        for bn in [32, 64, 128] {
            for bk in [8, 16, 32] {
                for tm in [4, 8] {
                    for tn in [4, 8] {
                        let threads = (bm / tm) * (bn / tn);
                        if (64..=512).contains(&threads) && tm * tn <= 64 {
                            for vec in [0, 1, 3] {
                                tilings.push(Config { bm, bn, bk, tm, tn, split: 0, vec });
                            }
                        }
                    }
                }
            }
        }
    }
    eprintln!("{} tilings", tilings.len());
    let mut first = Ranking::new();
    for config in &tilings {
        run(config, 1, FINALISTS, &mut first);
    }
    // Each shape's finalists again, `RUNS` times.
    let mut finalists: Vec<Config> = first.values().flatten().map(|&(_, c)| c).collect();
    finalists.sort_by_key(|c| c.env());
    finalists.dedup();
    let mut ranking = Ranking::new();
    for config in &finalists {
        run(config, RUNS, 1, &mut ranking);
    }
    let mut best: HashMap<&str, (f64, Config)> = ranking.iter().map(|(&f, e)| (f, e[0])).collect();
    // Each shape's best tiling with every split.
    let winners: Vec<Config> = best.values().map(|&(_, c)| c).collect();
    for tiling in winners {
        for split in [1, 2, 4, 8] {
            run(&Config { split, ..tiling }, RUNS, 1, &mut ranking);
        }
    }
    for (&function, entries) in &ranking {
        let entry = best.entry(function).or_insert(entries[0]);
        if entries[0].0 < entry.0 {
            *entry = entries[0];
        }
    }
    eprintln!();
    let mut table = String::new();
    println!("{:>10} {:>16} {:>9} {:>9} {:>7}  best (bm,bn,bk,tm,tn,split)", "shape", "m x n x k", "now us", "best us", "gain");
    let (mut now_total, mut best_total) = (0.0, 0.0);
    for &(function, m, n, k, a_k, b_k) in SHAPES {
        let now = baseline.get(function).copied().unwrap_or(f64::NAN);
        let Some(&(us, config)) = best.get(function) else {
            println!("{function:>10} {m:>5}x{n}x{k} {now:>9.1}        -");
            continue;
        };
        now_total += now;
        best_total += us.min(now);
        println!(
            "{function:>10} {:>16} {now:>9.1} {us:>9.1} {:>6.0}%  {}  ({:.1} TFLOPS)",
            format!("{m}x{n}x{k}"),
            100.0 * (now - us) / now,
            config.env(),
            2.0 * (m * n * k) as f64 / us / 1e6
        );
        if us < now {
            table += &format!(
                "{m} {n} {k} {} {} {} {} {} {} {} {} {}\n",
                a_k as i32, b_k as i32, config.bm, config.bn, config.bk, config.tm, config.tn, config.split, config.vec
            );
        }
    }
    println!("total: {now_total:.1} us now, {best_total:.1} us tuned ({:.0}% less)", 100.0 * (now_total - best_total) / now_total);
    std::fs::write("target/gpu_matmul_table.txt", &table).expect("cannot write target/gpu_matmul_table.txt");
    if install {
        let path = "cleave-mlir/cpp/matmul_table.inc";
        let header = std::fs::read_to_string(path).expect("cannot read the compiler's table");
        let comment: String = header.lines().take_while(|l| l.starts_with("//")).map(|l| format!("{l}\n")).collect();
        std::fs::write(path, format!("{comment}R\"(\n{table})\"\n")).expect("cannot write the compiler's table");
        println!("written to {path}: rebuild the compiler");
    }
}
