//! Launch the daemon via fork+exec. Never bare-fork from the client: CPython is
//! multi-threaded and macOS Core Foundation aborts across a fork, so the daemon
//! is always reached through an `exec`'d launcher.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::error::Result;
use crate::paths;
use crate::protocol::PROTOCOL_VERSION;

const ENV_PYTHON: &str = "SEMISWEET_PYTHON";
const ENV_IDLE_SECS: &str = "SEMISWEET_IDLE_SECS";

#[derive(Debug, Clone)]
pub enum Launcher {
    Python,
    Exe(PathBuf),
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
        Launcher::Python => {
            // TODO(phase 4): capture sys.executable from the live interpreter and
            // pass it here; semisweet._run_daemon() lands in Phase 4. SEMISWEET_PYTHON
            // overrides the interpreter until then.
            let python = std::env::var_os(ENV_PYTHON).unwrap_or_else(|| "python3".into());
            let mut command = Command::new(python);
            command.args(["-c", "import semisweet; semisweet._run_daemon()"]);
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
    child.wait()?;
    Ok(())
}
