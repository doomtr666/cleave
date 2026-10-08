<#
Compares the training step time of several nanoLM binaries, run in turn
(each once per round, every round in the same order) so that whatever drifts
during the comparison (clocks, temperature, background load) weighs on all
alike. Each run is `bench 0 1 <steps>`: the same batches, the same losses
(printed, to check they compute the same thing), the time per step of its
one round. Prints every run, then each binary's minimum and median, and each
one's relative to the first's.

  scripts/ab.ps1 target/release/nanolm_base.exe, target/release/nanolm.exe
  scripts/ab.ps1 (Get-ChildItem target/release/nanolm_*.exe).FullName -Rounds 3

Run it with nothing else busy: a single core taken by another program stalls
every one of the workers' barriers (VS Code with the focus, a build).
#>
param(
    [Parameter(Mandatory = $true, Position = 0)][string[]]$Exes,
    [int]$Rounds = 3,
    [int]$Steps = 10
)

function Measure-Step([string]$exe) {
    $out = & $exe bench 0 1 $Steps 2>$null | Out-String
    $ms = [regex]::Match($out, '(\d+) ms/step').Groups[1].Value
    $train = [regex]::Match($out, 'train ([\d.]+)').Groups[1].Value
    if (-not $ms) { throw "no time per step in the output of $exe" }
    [pscustomobject]@{ Ms = [int]$ms; Train = $train }
}

function Median([int[]]$xs) {
    $s = @($xs | Sort-Object)
    if ($s.Count % 2) { $s[[int][math]::Floor($s.Count / 2)] } else { ($s[$s.Count / 2 - 1] + $s[$s.Count / 2]) / 2 }
}

$times = @{}
foreach ($exe in $Exes) { $times[$exe] = @() }
for ($i = 1; $i -le $Rounds; $i++) {
    foreach ($exe in $Exes) {
        $r = Measure-Step $exe
        $times[$exe] += $r.Ms
        "round {0}: {1,6} ms/step (train {2})  {3}" -f $i, $r.Ms, $r.Train, (Split-Path $exe -Leaf)
    }
}
""
$first = $Exes[0]
$refMin = ($times[$first] | Measure-Object -Minimum).Minimum
foreach ($exe in $Exes) {
    $min = ($times[$exe] | Measure-Object -Minimum).Minimum
    $med = Median $times[$exe]
    "{0,-28} min {1,6} ms  median {2,6} ms  min vs first {3,7:P1}" -f (Split-Path $exe -Leaf), $min, $med, ($min / $refMin - 1)
}
