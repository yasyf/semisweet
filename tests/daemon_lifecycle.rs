//! Real lifecycle integration tests: the orphan daemon is spawned via the
//! `semisweet-daemon` binary (no Python/maturin needed) and driven through the
//! public `ClientStub`. Each test gets a unique runtime dir via the env overrides
//! and a short idle so the suite is deterministic and never hangs.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use semisweet::{Launcher, connect_or_spawn};

const BIN: &str = env!("CARGO_BIN_EXE_semisweet-daemon");

// The path overrides live in the process environment, which is shared across the
// test threads, so the tests are serialized while each owns the environment.
static ENV_LOCK: Mutex<()> = Mutex::new(());
static COUNTER: AtomicU32 = AtomicU32::new(0);

struct TestEnv {
    dir: PathBuf,
    socket: PathBuf,
    pid: PathBuf,
    _guard: MutexGuard<'static, ()>,
}

impl TestEnv {
    fn new(idle_secs: u64) -> Self {
        let guard = ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("ss{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        let socket = dir.join("d.sock");
        let lock = dir.join("d.lock");
        let log = dir.join("d.log");
        let pid = dir.join("d.pid");
        // SAFETY: the tests are serialized by ENV_LOCK, so no other thread reads or
        // writes the environment while these overrides are installed.
        unsafe {
            std::env::set_var("SEMISWEET_SOCKET", &socket);
            std::env::set_var("SEMISWEET_LOCK", &lock);
            std::env::set_var("SEMISWEET_LOG", &log);
            std::env::set_var("SEMISWEET_IDLE_SECS", idle_secs.to_string());
        }
        Self {
            dir,
            socket,
            pid,
            _guard: guard,
        }
    }

    fn launcher(&self) -> Launcher {
        Launcher::Exe(PathBuf::from(BIN))
    }

    fn daemon_pid(&self) -> u32 {
        // The pid file is written before the socket binds, so it always exists once
        // a connection has succeeded.
        std::fs::read_to_string(&self.pid)
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        if let Ok(raw) = std::fs::read_to_string(&self.pid)
            && let Ok(pid) = raw.trim().parse::<u32>()
        {
            kill9(pid);
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn kill9(pid: u32) {
    let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
}

fn pid_alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .unwrap()
        .success()
}

fn ppid_of(pid: u32) -> u32 {
    let output = Command::new("ps")
        .args(["-o", "ppid=", "-p", &pid.to_string()])
        .output()
        .unwrap();
    String::from_utf8(output.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

fn wait_until<F: FnMut() -> bool>(timeout: Duration, mut condition: F) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    condition()
}

fn wait_for_ready(child: &mut Child) {
    let stdout = child.stdout.take().unwrap();
    let mut line = String::new();
    BufReader::new(stdout).read_line(&mut line).unwrap();
    assert_eq!(line.trim(), "READY", "spawner failed to start the daemon");
}

fn spawn_killable_spawner(env: &TestEnv) -> Child {
    Command::new(BIN)
        .arg("--spawn-only")
        .env("SEMISWEET_SOCKET", &env.socket)
        .env("SEMISWEET_LOCK", env.dir.join("d.lock"))
        .env("SEMISWEET_LOG", env.dir.join("d.log"))
        .env("SEMISWEET_IDLE_SECS", "60")
        .stdout(Stdio::piped())
        .spawn()
        .unwrap()
}

#[test]
fn orphan_survives_spawner_death() {
    let env = TestEnv::new(60);

    let mut spawner = spawn_killable_spawner(&env);
    wait_for_ready(&mut spawner);

    let daemon_pid = env.daemon_pid();
    assert!(pid_alive(daemon_pid));

    // Kill the process that triggered the spawn; the daemon is a reparented orphan.
    kill9(spawner.id());
    let _ = spawner.wait();

    let mut stub = connect_or_spawn(&env.launcher()).unwrap();
    stub.ping().unwrap();

    assert_eq!(
        ppid_of(daemon_pid),
        1,
        "daemon must be reparented to init, proving it outlived its spawner"
    );
    assert_eq!(
        env.daemon_pid(),
        daemon_pid,
        "the surviving daemon must be the same process, not a respawn"
    );
}

#[test]
fn idle_shutdown_after_timeout() {
    let env = TestEnv::new(1);

    let mut stub = connect_or_spawn(&env.launcher()).unwrap();
    let daemon_pid = env.daemon_pid();
    assert!(pid_alive(daemon_pid));

    stub.bye().unwrap();
    drop(stub);

    let shut_down = wait_until(Duration::from_secs(6), || {
        !env.socket.exists() && !pid_alive(daemon_pid)
    });
    assert!(
        shut_down,
        "daemon should exit and remove its socket after the idle timeout"
    );
}

#[test]
fn two_clients_share_one_daemon() {
    let env = TestEnv::new(60);

    let mut a = connect_or_spawn(&env.launcher()).unwrap();
    let first_pid = env.daemon_pid();

    let mut b = connect_or_spawn(&env.launcher()).unwrap();
    let second_pid = env.daemon_pid();

    assert_eq!(
        first_pid, second_pid,
        "the second client must reuse the daemon, not spawn another"
    );
    assert!(pid_alive(first_pid));

    a.ping().unwrap();
    b.ping().unwrap();
}

#[test]
fn stale_socket_recovery() {
    let env = TestEnv::new(60);

    let mut stub = connect_or_spawn(&env.launcher()).unwrap();
    let old_pid = env.daemon_pid();
    stub.ping().unwrap();
    drop(stub);

    // kill -9 leaves the socket file behind with no listener.
    kill9(old_pid);
    assert!(wait_until(Duration::from_secs(5), || !pid_alive(old_pid)));
    assert!(
        env.socket.exists(),
        "a kill -9'd daemon should leave a stale socket file"
    );

    let mut fresh = connect_or_spawn(&env.launcher()).unwrap();
    let new_pid = env.daemon_pid();
    assert_ne!(
        old_pid, new_pid,
        "connect_or_spawn should clean the stale socket and start a new daemon"
    );
    fresh.ping().unwrap();
}
