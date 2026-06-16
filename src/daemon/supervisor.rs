//! Accept loop, connection refcount, and idle-shutdown state machine.
//!
//! A connection is counted on `Hello`, never on raw accept, so a connection that
//! never handshakes cannot pin the daemon. When the count reaches zero an idle timer
//! arms; if it fires while still zero the daemon shuts down gracefully.

use std::sync::Arc;
use std::time::Duration;

use tokio::net::UnixListener;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::mpsc;
use tokio::time::{Instant, sleep};

use super::Config;
use super::conn;
use super::state::DaemonState;
use crate::error::Result;

/// How long shutdown waits for the write-behind queue to drain before giving up, so a
/// wedged commit can't hang the daemon's exit forever.
const SHUTDOWN_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);
const SHUTDOWN_DRAIN_POLL: Duration = Duration::from_millis(50);

pub(super) enum ClientEvent {
    Connected,
    Disconnected,
}

/// Live-connection refcount. `Connected`/`Disconnected` are strictly paired (a
/// connection is only counted once it handshakes), so the daemon is idle exactly
/// when the count is zero.
struct Clients {
    active: usize,
}

impl Clients {
    fn new() -> Self {
        Self { active: 0 }
    }

    fn connected(&mut self) {
        self.active += 1;
    }

    /// Records a dropped connection and reports whether the daemon is now idle.
    fn disconnected(&mut self) -> bool {
        self.active -= 1;
        self.is_idle()
    }

    fn is_idle(&self) -> bool {
        self.active == 0
    }
}

pub(super) async fn run(config: &Config) -> Result<()> {
    let listener = UnixListener::bind(&config.socket)?;
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<ClientEvent>();

    let state = Arc::new(DaemonState::new());
    let mut clients = Clients::new();

    let idle = config.idle;
    let idle_timer = sleep(idle);
    tokio::pin!(idle_timer);

    let mut sigterm = signal(SignalKind::terminate())?;

    loop {
        tokio::select! {
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _addr)) => {
                        tokio::spawn(conn::serve(
                            stream,
                            events_tx.clone(),
                            state.clone(),
                            config.daemon_version.clone(),
                            config.protocol,
                        ));
                    }
                    // A single failed accept (e.g. fd exhaustion) must not kill the
                    // daemon; log it and keep serving the connections we already have.
                    Err(error) => {
                        eprintln!("semisweet-daemon: accept failed: {error}");
                        continue;
                    }
                }
            }
            Some(event) = events_rx.recv() => {
                match event {
                    ClientEvent::Connected => clients.connected(),
                    ClientEvent::Disconnected => {
                        if clients.disconnected() {
                            idle_timer.as_mut().reset(Instant::now() + idle);
                        }
                    }
                }
            }
            _ = &mut idle_timer, if clients.is_idle() && state.writer.is_drained() => {
                break;
            }
            _ = sigterm.recv() => {
                break;
            }
        }
    }
    // Both exit paths (idle and SIGTERM) drain the write-behind queue before returning so
    // accepted-but-not-yet-committed writes still land, bounded so a stuck commit can't
    // hang shutdown forever.
    let deadline = Instant::now() + SHUTDOWN_DRAIN_TIMEOUT;
    while !state.writer.is_drained() && Instant::now() < deadline {
        sleep(SHUTDOWN_DRAIN_POLL).await;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_until_first_connection_then_idle_again_when_last_drops() {
        let mut clients = Clients::new();
        assert!(clients.is_idle());

        clients.connected();
        assert!(!clients.is_idle());
        assert!(
            clients.disconnected(),
            "the last connection dropping is idle"
        );
    }

    #[test]
    fn two_connections_keep_the_daemon_busy_until_both_drop() {
        let mut clients = Clients::new();

        clients.connected();
        clients.connected();
        assert!(!clients.disconnected(), "one of two dropping is not idle");
        assert!(clients.disconnected(), "the second dropping is idle");
    }
}
