//! Launch the daemon via fork+exec. Never bare-fork from the client: CPython is
//! multi-threaded and macOS Core Foundation aborts across a fork, so the daemon
//! is always reached through an `exec`'d launcher.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::error::{Error, Result};
use crate::paths;
use crate::protocol::PROTOCOL_VERSION;

const ENV_IDLE_SECS: &str = "SEMISWEET_IDLE_SECS";
const ENV_MODEL_CACHE: &str = "SEMISWEET_MODEL_CACHE";

// The compiled extension is the `semisweet.core` submodule (maturin mixed layout:
// `python/semisweet/` is the package, `module-name = "semisweet.core"`). Import the daemon
// entry point straight from that submodule: `_run_daemon` is deliberately kept out of the
// package's public `__all__`, so `semisweet._run_daemon` does not exist; the submodule path
// is the stable handle the launcher targets.
const DAEMON_BOOTSTRAP: &str = "from semisweet.core import _run_daemon; _run_daemon()";

// Credential and model-location variables the daemon's backends read at
// construction. The daemon runs detached with `cwd = /`, so the spawning client
// forwards exactly the variables its chosen backends need.
const PASSTHROUGH_ENV: &[&str] = &[
    "VOYAGE_API_KEY",
    "TURBOPUFFER_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "S3_ENDPOINT",
    "HF_HOME",
];

#[derive(Debug, Clone)]
pub enum Launcher {
    Python { executable: PathBuf },
    Exe(PathBuf),
}

fn default_model_cache() -> Option<PathBuf> {
    directories::BaseDirs::new().map(|base| base.cache_dir().join("semisweet").join("models"))
}

pub(crate) fn spawn_daemon(launcher: &Launcher) -> Result<()> {
    let socket = paths::socket_path()?;
    let lock = paths::lock_path()?;
    let log = paths::log_path()?;

    let stdout = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)?;
    let stderr = stdout.try_clone()?;

    let mut command = match launcher {
        Launcher::Python { executable } => {
            let mut command = Command::new(executable);
            command.args(["-c", DAEMON_BOOTSTRAP]);
            command
        }
        Launcher::Exe(path) => Command::new(path),
    };

    command
        .env("SEMISWEET_SOCKET", &socket)
        .env("SEMISWEET_LOCK", &lock)
        .env("SEMISWEET_LOG", &log)
        .env("SEMISWEET_PROTOCOL", PROTOCOL_VERSION.to_string())
        .current_dir("/")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    for var in PASSTHROUGH_ENV {
        if let Some(value) = std::env::var_os(var) {
            command.env(var, value);
        }
    }

    // fastembed's default model cache is relative to the working directory; the
    // daemon runs with `cwd = /`, so pin an absolute cache dir. Honour an explicit
    // override, otherwise default to a stable per-user cache path.
    match std::env::var_os(ENV_MODEL_CACHE) {
        Some(value) => {
            command.env(ENV_MODEL_CACHE, value);
        }
        None => {
            if let Some(cache) = default_model_cache() {
                command.env(ENV_MODEL_CACHE, cache);
            }
        }
    }

    if let Some(idle) = std::env::var_os(ENV_IDLE_SECS) {
        command.env(ENV_IDLE_SECS, idle);
    }

    // SAFETY: setsid is async-signal-safe — the only kind of work permitted between
    // fork and exec. It detaches the launcher into a fresh session so the daemon
    // grandchild can never acquire a controlling terminal.
    unsafe {
        command.pre_exec(|| {
            nix::unistd::setsid()
                .map(|_| ())
                .map_err(std::io::Error::from)
        });
    }

    let mut child = command.spawn()?;
    // Reap the launcher. It forks the orphan daemon and exits immediately; the
    // orphan is reparented to init and outlives both the launcher and this client.
    // A non-zero status means the launch failed before the fork (bad import, bad
    // config), so surface it now instead of letting the connect poll mistake it
    // for a 30s-later shutdown.
    let status = child.wait()?;
    if !status.success() {
        return Err(Error::Daemon(format!(
            "daemon launcher exited with {status}; see log at {}",
            log.display()
        )));
    }
    Ok(())
}
