<#
Runs the test suite at one of two levels.

  smoke      (default) the compiler's own tests: `cargo test -p cleave`. The
             working loop, about a minute and a half after a change to the
             compiler: ~45 s relinking the test binaries (each embeds
             MLIR/LLVM), ~50 s running them.
  workspace  every crate, the example kernels included (nanoLM, MNIST,
             digits): each one's `build.rs` recompiles its kernel with the
             new compiler, several minutes for nanoLM alone. Before a commit
             touching code generation, or at a milestone.

Any further arguments go to the test binaries (a name filter, `--exact`,
`--include-ignored`...):

  scripts/test.ps1                      # smoke
  scripts/test.ps1 smoke leaks          # smoke, tests whose name contains "leaks"
  scripts/test.ps1 workspace

A test that costs seconds where a smaller program would check the same thing
should be made smaller rather than skipped: `a_large_light_struct_crosses_a_
call_by_pointer` took 80 s of the suite's ~90 s with a 57 KB struct, 0.8 s
with 168 bytes, past the same by-pointer threshold.
#>
param(
    [ValidateSet("smoke", "workspace")]
    [string]$Level = "smoke",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TestArgs = @()
)

$scope = if ($Level -eq "workspace") { @("--workspace") } else { @("-p", "cleave") }
$cargoArgs = @("test", "--release") + $scope + @("--no-fail-fast")
if ($TestArgs.Count -gt 0) { $cargoArgs += @("--") + $TestArgs }

$watch = [Diagnostics.Stopwatch]::StartNew()
& cargo @cargoArgs
$code = $LASTEXITCODE
"{0} tests: {1:N0} s" -f $Level, $watch.Elapsed.TotalSeconds
exit $code
