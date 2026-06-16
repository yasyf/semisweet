//! A persistent blocking client connection to the daemon, with lazy spawn and a
//! single transparent reconnect.

use std::io::ErrorKind;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::{Duration, Instant};

use super::spawn::{Launcher, spawn_daemon};
use crate::error::{Error, Result};
use crate::paths;
use crate::protocol::{self, ClientId, PROTOCOL_VERSION, ProtocolError, Request, Response};

const SPAWN_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(25);

pub struct ClientStub {
    stream: UnixStream,
    client: ClientId,
    launcher: Launcher,
}

pub fn connect_or_spawn(launcher: &Launcher) -> Result<ClientStub> {
    let stream = connect_with_spawn(launcher)?;
    let mut stub = ClientStub {
        stream,
        client: ClientId::generate(),
        launcher: launcher.clone(),
    };
    // Retry the handshake once on a transient connection drop: a client racing the
    // daemon's idle shutdown can connect to a socket whose listener is already gone,
    // so the first `hello` reads EOF. Reconnecting respawns the daemon transparently.
    match stub.hello() {
        Ok(()) => Ok(stub),
        Err(error) if is_transient_drop(&error) => {
            stub.reconnect()?;
            Ok(stub)
        }
        Err(other) => Err(other),
    }
}

fn dead_socket(error: &std::io::Error) -> bool {
    matches!(
        error.kind(),
        ErrorKind::NotFound | ErrorKind::ConnectionRefused
    )
}

// Only a dropped connection or a corrupt frame warrants a transparent reconnect;
// every other variant — a protocol version mismatch above all — is fatal and must
// surface as itself rather than being retried or masked into `DaemonShutdown`.
fn is_transient_drop(error: &Error) -> bool {
    match error {
        Error::Io(_) | Error::Codec(_) => true,
        Error::EmptyQuery
        | Error::EmptyContext
        | Error::EmptyKey
        | Error::EmptyEntity
        | Error::EmptyNamespace
        | Error::InvalidNamespace(_)
        | Error::EmptyEmbedding
        | Error::ZeroEmbedding
        | Error::NonFiniteEmbedding
        | Error::DimMismatch { .. }
        | Error::MissingEnv(_)
        | Error::UnknownBackend(_)
        | Error::InvalidConfig(_)
        | Error::NamespaceMissing(_)
        | Error::EntityExtraction(_)
        | Error::Embedding(_)
        | Error::VectorStorage(_)
        | Error::ObjectStorage(_)
        | Error::DaemonShutdown
        | Error::Daemon(_)
        | Error::ProtocolVersionMismatch { .. } => false,
    }
}

fn try_connect(socket: &Path) -> Result<Option<UnixStream>> {
    match UnixStream::connect(socket) {
        Ok(stream) => Ok(Some(stream)),
        Err(error) if dead_socket(&error) => Ok(None),
        Err(error) => Err(Error::Io(error)),
    }
}

fn connect_with_spawn(launcher: &Launcher) -> Result<UnixStream> {
    let socket = paths::socket_path()?;

    if let Some(stream) = try_connect(&socket)? {
        return Ok(stream);
    }

    // Serialize spawning across clients with a dedicated mutex file. This is a
    // *different* lock from the daemon's single-instance lock, so holding it can
    // never block the daemon's own startup.
    let spawn_lock_path = paths::spawn_lock_path()?;
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(&spawn_lock_path)?;
    let mut spawn_lock = fd_lock::RwLock::new(lock_file);
    let _guard = spawn_lock.write()?;

    // Another client may have won the race and spawned the daemon while we waited.
    if let Some(stream) = try_connect(&socket)? {
        return Ok(stream);
    }

    spawn_daemon(launcher)?;

    let deadline = Instant::now() + SPAWN_TIMEOUT;
    loop {
        if let Some(stream) = try_connect(&socket)? {
            return Ok(stream);
        }
        if Instant::now() >= deadline {
            return Err(Error::DaemonShutdown);
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

impl ClientStub {
    pub fn client_id(&self) -> ClientId {
        self.client
    }

    fn hello(&mut self) -> Result<()> {
        let request = Request::Hello {
            client: self.client,
            protocol: PROTOCOL_VERSION,
            pid: std::process::id(),
        };
        protocol::write_frame(&mut self.stream, &request)?;
        match protocol::read_frame(&mut self.stream)? {
            Response::Welcome { protocol, .. } if protocol == PROTOCOL_VERSION => Ok(()),
            Response::Error(ProtocolError::VersionMismatch { client, daemon }) => {
                Err(Error::ProtocolVersionMismatch { client, daemon })
            }
            other => Err(Error::Daemon(format!(
                "unexpected handshake response: {other:?}"
            ))),
        }
    }

    fn round_trip(&mut self, request: &Request) -> Result<Response> {
        protocol::write_frame(&mut self.stream, request)?;
        protocol::read_frame(&mut self.stream)
    }

    fn reconnect(&mut self) -> Result<()> {
        self.stream = connect_with_spawn(&self.launcher)?;
        self.hello()
    }

    pub fn request(&mut self, request: &Request) -> Result<Response> {
        match self.round_trip(request) {
            Ok(response) => Ok(response),
            Err(error) if is_transient_drop(&error) => {
                self.reconnect()?;
                self.round_trip(request)
            }
            Err(other) => Err(other),
        }
    }

    pub fn ping(&mut self) -> Result<()> {
        match self.request(&Request::Ping)? {
            Response::Pong => Ok(()),
            other => Err(Error::Daemon(format!("expected Pong, got {other:?}"))),
        }
    }

    pub fn bye(&mut self) -> Result<()> {
        protocol::write_frame(&mut self.stream, &Request::Bye)?;
        match protocol::read_frame(&mut self.stream)? {
            Response::Goodbye => Ok(()),
            other => Err(Error::Daemon(format!("expected Goodbye, got {other:?}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Error as IoError, ErrorKind};

    use super::is_transient_drop;
    use crate::error::Error;

    #[test]
    fn io_and_codec_failures_are_transient_drops() {
        let io = Error::Io(IoError::new(ErrorKind::BrokenPipe, "peer hung up"));
        let codec = Error::Codec(postcard::from_bytes::<u32>(&[]).unwrap_err());
        assert!(is_transient_drop(&io));
        assert!(is_transient_drop(&codec));
    }

    #[test]
    fn protocol_version_mismatch_is_fatal_not_transient() {
        let mismatch = Error::ProtocolVersionMismatch {
            client: 1,
            daemon: 2,
        };
        assert!(!is_transient_drop(&mismatch));
    }

    #[test]
    fn lifecycle_and_validation_errors_are_not_transient_drops() {
        assert!(!is_transient_drop(&Error::DaemonShutdown));
        assert!(!is_transient_drop(&Error::Daemon("listener gone".into())));
        assert!(!is_transient_drop(&Error::EmptyQuery));
    }
}
