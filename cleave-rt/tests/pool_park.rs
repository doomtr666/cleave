//! `CLEAVE_POOL_PARK` (`cleave_rt::depot_push`): freed blocks too large for
//! the thread caches are parked only up to a fraction of the most ever live,
//! the rest given back to the system. Its own test binary: the variable is
//! read once per process.

#[test]
fn large_freed_blocks_are_parked_up_to_a_fraction_of_the_peak() {
    // SAFETY: set before any thread reads the environment (this binary's
    // only test, the runtime not yet used).
    unsafe { std::env::set_var("CLEAVE_POOL_PARK", "0.5") };
    const MIB: i64 = 1 << 20;
    let blocks: Vec<*mut u8> = (0..8).map(|_| cleave_rt::cleave_alloc_rc(MIB)).collect();
    let (held, parked) = cleave_rt::pool_large_bytes();
    assert_eq!(parked, 0);
    let block = held / 8;
    for &b in &blocks {
        // SAFETY: each block is live and released once.
        unsafe { cleave_rt::cleave_release(b) };
        let (_, parked) = cleave_rt::pool_large_bytes();
        assert!(parked <= 4 * block, "{parked} bytes parked, the peak is {}", 8 * block);
    }
    // Half the peak kept, for the next blocks of that size; the rest given
    // back.
    assert_eq!(cleave_rt::pool_large_bytes(), (4 * block, 4 * block));
    let again: Vec<*mut u8> = (0..6).map(|_| cleave_rt::cleave_alloc_rc(MIB)).collect();
    assert_eq!(cleave_rt::pool_large_bytes(), (6 * block, 0), "four reused, two taken from the system");
    for b in again {
        unsafe { cleave_rt::cleave_release(b) };
    }
}
