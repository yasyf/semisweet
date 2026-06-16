//! The orphan daemon: a lazily-started, single-instance, per-user process that
//! survives its spawner, tracks clients, and idle-shuts-down.
//!
//! `run_daemon` is the process entry point. It forks once into an orphan,
//! claims the single-instance lock, and runs the async supervisor.

mod conn;
mod inference;
mod state;
mod supervisor;
mod writer;

use std::ffi::OsStr;
use std::path::PathBuf;
use std::time::Duration;

use nix::unistd::ForkResult;

use crate::error::{Error, Result};
use crate::{paths, protocol};

const DEFAULT_IDLE_SECS: u64 = 300;
const ENV_IDLE_SECS: &str = "SEMISWEET_IDLE_SECS";
const ENV_PROTOCOL: &str = "SEMISWEET_PROTOCOL";

pub(crate) struct Config {
    socket: PathBuf,
    lock: PathBuf,
    pid: PathBuf,
    idle: Duration,
    protocol: u32,
    daemon_version: String,
}

impl Config {
    fn from_env() -> Result<Self> {
        let protocol = match std::env::var_os(ENV_PROTOCOL) {
            Some(value) => parse_u32(&value, ENV_PROTOCOL)?,
            None => protocol::PROTOCOL_VERSION,
        };
        if protocol != protocol::PROTOCOL_VERSION {
            return Err(Error::ProtocolVersionMismatch {
                client: protocol,
                daemon: protocol::PROTOCOL_VERSION,
            });
        }
        let idle = match std::env::var_os(ENV_IDLE_SECS) {
            Some(value) => Duration::from_secs(parse_u64(&value, ENV_IDLE_SECS)?),
            None => Duration::from_secs(DEFAULT_IDLE_SECS),
        };
        Ok(Self {
            socket: paths::socket_path()?,
            lock: paths::lock_path()?,
            pid: paths::pid_path()?,
            idle,
            protocol,
            daemon_version: env!("CARGO_PKG_VERSION").to_owned(),
        })
    }
}

fn parse_u32(value: &OsStr, var: &str) -> Result<u32> {
    value
        .to_str()
        .and_then(|raw| raw.parse().ok())
        .ok_or_else(|| Error::Daemon(format!("invalid {var}: {value:?}")))
}

fn parse_u64(value: &OsStr, var: &str) -> Result<u64> {
    value
        .to_str()
        .and_then(|raw| raw.parse().ok())
        .ok_or_else(|| Error::Daemon(format!("invalid {var}: {value:?}")))
}

pub fn run_daemon() -> Result<()> {
    let config = Config::from_env()?;
    daemonize()?;
    run_orphan(config)
}

fn daemonize() -> Result<()> {
    // SAFETY: this runs at the process entry point, before any tokio runtime or
    // extra threads exist, so the single live thread forks cleanly. The launcher
    // (parent) exits immediately; the child is reparented to init as the orphan.
    match unsafe { nix::unistd::fork() }.map_err(std::io::Error::from)? {
        ForkResult::Parent { .. } => std::process::exit(0),
        ForkResult::Child => Ok(()),
    }
}

fn run_orphan(config: Config) -> Result<()> {
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&config.lock)?;
    let mut instance_lock = fd_lock::RwLock::new(lock_file);
    let _guard = match instance_lock.try_write() {
        Ok(guard) => guard,
        // Another daemon already owns the single-instance lock; stand down quietly.
        Err(_) => return Ok(()),
    };

    if config.socket.exists() {
        std::fs::remove_file(&config.socket)?;
    }
    std::fs::write(&config.pid, std::process::id().to_string())?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(supervisor::run(&config));

    let _ = std::fs::remove_file(&config.socket);
    let _ = std::fs::remove_file(&config.pid);
    result
}
