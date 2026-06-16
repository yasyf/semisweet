//! The real, spawnable orphan daemon binary.
//!
//! With no arguments it *is* the daemon entry point (`run_daemon`). `--spawn-only`
//! makes it a killable spawner: it lazily starts the daemon, prints `READY`, and
//! hangs so an integration test can `kill -9` it and prove the orphan survives.

use std::io::Write;
use std::process::ExitCode;
use std::time::Duration;

use semisweet::{Error, Launcher, connect_or_spawn, run_daemon};

fn main() -> ExitCode {
    let result = match std::env::args().nth(1).as_deref() {
        None => run_daemon(),
        Some("--spawn-only") => spawn_only(),
        Some(other) => Err(Error::Daemon(format!("unknown mode: {other}"))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("semisweet-daemon: {error}");
            ExitCode::FAILURE
        }
    }
}

fn spawn_only() -> Result<(), Error> {
    let exe = std::env::current_exe()?;
    let _stub = connect_or_spawn(&Launcher::Exe(exe))?;
    println!("READY");
    std::io::stdout().flush()?;
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}
