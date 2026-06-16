//! Per-user runtime directory and the daemon's socket/lock/log/pid paths.
//!
//! Each path honours an environment override first (so tests and tooling can
//! redirect the daemon), then falls back to `{runtime}/semisweet-{uid}/v{N}.*`.
//! The socket name is kept short because macOS caps `sun_path` near 104 bytes.

use std::fs;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::PathBuf;

use crate::error::Result;
use crate::protocol::PROTOCOL_VERSION;

const PROTOCOL_MAJOR: u32 = PROTOCOL_VERSION;
const DIR_MODE: u32 = 0o700;

const ENV_SOCKET: &str = "SEMISWEET_SOCKET";
const ENV_LOCK: &str = "SEMISWEET_LOCK";
const ENV_LOG: &str = "SEMISWEET_LOG";

fn base_runtime_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        if let Some(dir) = directories::BaseDirs::new()
            .and_then(|base| base.runtime_dir().map(std::path::Path::to_path_buf))
        {
            return dir;
        }
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        return PathBuf::from(tmp);
    }
    PathBuf::from("/tmp")
}

fn runtime_dir() -> Result<PathBuf> {
    let uid = nix::unistd::getuid().as_raw();
    let dir = base_runtime_dir().join(format!("semisweet-{uid}"));
    fs::DirBuilder::new()
        .recursive(true)
        .mode(DIR_MODE)
        .create(&dir)?;
    fs::set_permissions(&dir, fs::Permissions::from_mode(DIR_MODE))?;
    Ok(dir)
}

fn default_name(ext: &str) -> String {
    format!("v{PROTOCOL_MAJOR}.{ext}")
}

fn env_or_default(var: &str, ext: &str) -> Result<PathBuf> {
    match std::env::var_os(var) {
        Some(value) => Ok(PathBuf::from(value)),
        None => Ok(runtime_dir()?.join(default_name(ext))),
    }
}

pub fn socket_path() -> Result<PathBuf> {
    env_or_default(ENV_SOCKET, "sock")
}

pub fn lock_path() -> Result<PathBuf> {
    env_or_default(ENV_LOCK, "lock")
}

pub fn log_path() -> Result<PathBuf> {
    env_or_default(ENV_LOG, "log")
}

pub fn spawn_lock_path() -> Result<PathBuf> {
    Ok(lock_path()?.with_extension("spawn"))
}

pub fn pid_path() -> Result<PathBuf> {
    Ok(lock_path()?.with_extension("pid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_dir_is_created_with_owner_only_perms() {
        let dir = runtime_dir().unwrap();
        assert!(dir.is_dir());
        assert!(
            dir.file_name()
                .and_then(|n| n.to_str())
                .unwrap()
                .starts_with("semisweet-")
        );
        let mode = fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, DIR_MODE);
    }

    #[test]
    fn default_names_carry_protocol_major_and_extension() {
        assert_eq!(default_name("sock"), format!("v{PROTOCOL_MAJOR}.sock"));
        assert_eq!(default_name("lock"), format!("v{PROTOCOL_MAJOR}.lock"));
        assert_eq!(default_name("log"), format!("v{PROTOCOL_MAJOR}.log"));
    }

    #[test]
    fn spawn_and_pid_derive_from_the_lock_path() {
        let lock = lock_path().unwrap();
        assert_eq!(spawn_lock_path().unwrap(), lock.with_extension("spawn"));
        assert_eq!(pid_path().unwrap(), lock.with_extension("pid"));
    }

    #[test]
    fn default_socket_path_lives_under_the_runtime_dir() {
        // The lib unit-test process never sets the override env vars.
        let socket = socket_path().unwrap();
        assert_eq!(
            socket.file_name().and_then(|n| n.to_str()).unwrap(),
            default_name("sock")
        );
        assert!(socket.parent().unwrap().is_dir());
    }
}
