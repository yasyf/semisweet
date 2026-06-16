//! Per-connection serving task over a length-delimited postcard frame stream.
//!
//! The first frame must be `Hello`; only then is the connection counted (via a
//! `Connected` event). Subsequent frames are served against the namespace registry:
//! `Get`/`Del` run on the blocking inference pool, `Set` enqueues on the write-behind
//! queue and acknowledges immediately. A `Disconnected` event fires on EOF — this is
//! how a `kill -9`'d client is detected.

use std::collections::BTreeSet;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use tokio::net::UnixStream;
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::codec::{Framed, LengthDelimitedCodec};

use super::state::{DaemonState, NamespaceEntry, PendingWrites};
use super::supervisor::ClientEvent;
use super::writer::WriteJob;
use crate::error::{Error, Result};
use crate::newtype::{Context, EntryId, Key, QueryText};
use crate::protocol::{self, MAX_FRAME_BYTES, ProtocolError, Request, Response};
use crate::registry::{self, NamespaceConfig};

type Conn = Framed<UnixStream, LengthDelimitedCodec>;
type Parts = (QueryText, BTreeSet<Key>, Option<Context>);

/// Pairs the connection refcount across the serve task's whole life: sends `Connected`
/// when constructed (after a successful handshake) and `Disconnected` on drop, so an
/// early return or panic still decrements the count.
struct ConnectionGuard {
    events: UnboundedSender<ClientEvent>,
}

impl ConnectionGuard {
    fn enter(events: UnboundedSender<ClientEvent>) -> Self {
        let _ = events.send(ClientEvent::Connected);
        Self { events }
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        let _ = self.events.send(ClientEvent::Disconnected);
    }
}

pub(super) async fn serve(
    stream: UnixStream,
    events: UnboundedSender<ClientEvent>,
    state: Arc<DaemonState>,
    daemon_version: String,
    protocol: u32,
) {
    // The client frames with a 64 MiB ceiling; the daemon must accept frames just as
    // large or a multi-megabyte `Set` payload is silently rejected by the default
    // 8 MiB codec limit.
    let codec = LengthDelimitedCodec::builder()
        .max_frame_length(MAX_FRAME_BYTES)
        .new_codec();
    let mut framed = Framed::new(stream, codec);
    match handshake(&mut framed, &daemon_version, protocol).await {
        Ok(true) => {}
        Ok(false) | Err(_) => return,
    }
    let _guard = ConnectionGuard::enter(events);
    let _ = message_loop(&mut framed, &state, &daemon_version, protocol).await;
}

async fn handshake(framed: &mut Conn, daemon_version: &str, protocol: u32) -> Result<bool> {
    let frame = match framed.next().await {
        Some(frame) => frame?,
        None => return Ok(false),
    };
    let request: Request = protocol::decode(&frame)?;
    // The opening message must be Hello. Any other request is a protocol violation;
    // close without counting so an un-handshaked connection cannot pin the daemon.
    let Request::Hello {
        protocol: client_protocol,
        ..
    } = request
    else {
        return Ok(false);
    };
    if client_protocol != protocol {
        let response = Response::Error(ProtocolError::VersionMismatch {
            client: client_protocol,
            daemon: protocol,
        });
        send(framed, &response).await?;
        return Ok(false);
    }
    send(framed, &welcome(daemon_version, protocol)).await?;
    Ok(true)
}

async fn message_loop(
    framed: &mut Conn,
    state: &Arc<DaemonState>,
    daemon_version: &str,
    protocol: u32,
) -> Result<()> {
    loop {
        let frame = match framed.next().await {
            Some(frame) => frame?,
            None => return Ok(()),
        };
        let request: Request = protocol::decode(&frame)?;
        let response = match request {
            Request::Ping => Response::Pong,
            Request::Bye => {
                send(framed, &Response::Goodbye).await?;
                return Ok(());
            }
            Request::Hello {
                protocol: client_protocol,
                ..
            } => {
                if client_protocol == protocol {
                    welcome(daemon_version, protocol)
                } else {
                    Response::Error(ProtocolError::VersionMismatch {
                        client: client_protocol,
                        daemon: protocol,
                    })
                }
            }
            Request::RegisterNamespace {
                namespace,
                config_json,
            } => handle_register(state, namespace, config_json).await,
            Request::Get {
                namespace,
                query,
                keys,
                context,
            } => handle_get(state, namespace, query, keys, context).await,
            Request::Set {
                namespace,
                query,
                keys,
                context,
                value,
            } => handle_set(state, namespace, query, keys, context, value.into_vec()),
            Request::Del {
                namespace,
                query,
                keys,
                context,
            } => handle_del(state, namespace, query, keys, context).await,
        };
        send(framed, &response).await?;
    }
}

async fn handle_register(
    state: &Arc<DaemonState>,
    namespace: String,
    config_json: String,
) -> Response {
    // Held across the resolve-build-register sequence so the namespace builds at most
    // once: a concurrent registration of the same name blocks here, then resolves the
    // first registration's entry below instead of building a duplicate cache.
    let build_lock = match state.build_lock(&namespace) {
        Ok(build_lock) => build_lock,
        Err(error) => return backend_error(error),
    };
    let _build_guard = build_lock.lock().await;
    match state.resolve(&namespace) {
        Ok(Some(_)) => return Response::Registered,
        Ok(None) => {}
        Err(error) => return backend_error(error),
    }
    let config: NamespaceConfig = match serde_json::from_str(&config_json) {
        Ok(config) => config,
        Err(error) => {
            return Response::Error(ProtocolError::InvalidRequest(format!(
                "invalid namespace config: {error}"
            )));
        }
    };
    let build_namespace = namespace.clone();
    let built = state
        .inference
        .run(move || registry::build_cache(&build_namespace, &config))
        .await;
    match built {
        Ok(Ok(cache)) => {
            let entry = Arc::new(NamespaceEntry {
                cache: Arc::new(cache),
                pending: Arc::new(PendingWrites::default()),
            });
            match state.register(namespace, entry) {
                Ok(()) => Response::Registered,
                Err(error) => backend_error(error),
            }
        }
        // Either the build failed (inner) or the pool is shutting down (outer).
        Ok(Err(error)) | Err(error) => backend_error(error),
    }
}

async fn handle_get(
    state: &Arc<DaemonState>,
    namespace: String,
    query: String,
    keys: Vec<String>,
    context: Option<String>,
) -> Response {
    let entry = match resolve(state, &namespace) {
        Ok(entry) => entry,
        Err(response) => return response,
    };
    let (query, keys, context) = match reconstruct(query, keys, context) {
        Ok(parts) => parts,
        Err(response) => return response,
    };
    // Read-after-write: an in-flight `set` for this exact entry is served straight
    // from the pending shadow, before (and instead of) the embed + vector-search path.
    let id = EntryId::derive(&query, &keys);
    match entry.pending.get(&id) {
        Ok(Some(value)) => {
            return Response::Value(Some(serde_bytes::ByteBuf::from(value.to_vec())));
        }
        Ok(None) => {}
        Err(error) => return backend_error(error),
    }
    let cache = entry.cache.clone();
    match state
        .inference
        .run(move || cache.get(&query, &keys, &context))
        .await
    {
        Ok(Ok(value)) => Response::Value(value.map(serde_bytes::ByteBuf::from)),
        Ok(Err(error)) | Err(error) => backend_error(error),
    }
}

fn handle_set(
    state: &Arc<DaemonState>,
    namespace: String,
    query: String,
    keys: Vec<String>,
    context: Option<String>,
    value: Vec<u8>,
) -> Response {
    let entry = match resolve(state, &namespace) {
        Ok(entry) => entry,
        Err(response) => return response,
    };
    let (query, keys, context) = match reconstruct(query, keys, context) {
        Ok(parts) => parts,
        Err(response) => return response,
    };
    let id = EntryId::derive(&query, &keys);
    let value: Arc<[u8]> = Arc::from(value);
    // Record the shadow before enqueueing: were the order reversed, a worker could run
    // and remove the (not-yet-inserted) id, leaking the value into the buffer forever.
    if let Err(error) = entry.pending.insert(id, value.clone()) {
        return backend_error(error);
    }
    let job = WriteJob {
        cache: entry.cache.clone(),
        pending: entry.pending.clone(),
        id,
        query,
        keys,
        context,
        value: value.clone(),
    };
    let accepted = state.writer.try_push(job);
    if !accepted {
        // The bounded queue is full, so this write will never run; drop the shadow
        // rather than advertise a read-after-write hit that never lands. `remove_if` so a
        // concurrent accepted same-id write's (different-Arc) shadow is never dropped.
        if let Err(error) = entry.pending.remove_if(&id, &value) {
            return backend_error(error);
        }
    }
    Response::Accepted(accepted)
}

async fn handle_del(
    state: &Arc<DaemonState>,
    namespace: String,
    query: String,
    keys: Vec<String>,
    context: Option<String>,
) -> Response {
    let entry = match resolve(state, &namespace) {
        Ok(entry) => entry,
        Err(response) => return response,
    };
    let (query, keys, context) = match reconstruct(query, keys, context) {
        Ok(parts) => parts,
        Err(response) => return response,
    };
    // Drop any in-flight shadow so the delete both wins over a pending hit and counts
    // as a real deletion even before the write-behind has made the entry durable. A
    // delete observed before the matching write commits also makes that job skip (it no
    // longer `holds` its value); a delete that races in mid-commit is not ordered against
    // the write, so that one entry can still land durably — an accepted bound of the
    // lock-free write-behind, not a guarantee.
    let id = EntryId::derive(&query, &keys);
    let removed_pending = match entry.pending.remove(&id) {
        Ok(removed) => removed,
        Err(error) => return backend_error(error),
    };
    let cache = entry.cache.clone();
    match state
        .inference
        .run(move || cache.delete(&query, &keys, &context))
        .await
    {
        Ok(Ok(deleted)) => Response::Deleted(removed_pending || deleted),
        Ok(Err(error)) | Err(error) => backend_error(error),
    }
}

fn welcome(daemon_version: &str, protocol: u32) -> Response {
    Response::Welcome {
        daemon_version: daemon_version.to_owned(),
        protocol,
    }
}

fn resolve(
    state: &DaemonState,
    namespace: &str,
) -> std::result::Result<Arc<NamespaceEntry>, Response> {
    match state.resolve(namespace) {
        Ok(Some(entry)) => Ok(entry),
        Ok(None) => Err(Response::Error(ProtocolError::UnknownNamespace(
            namespace.to_owned(),
        ))),
        Err(error) => Err(backend_error(error)),
    }
}

fn reconstruct(
    query: String,
    keys: Vec<String>,
    context: Option<String>,
) -> std::result::Result<Parts, Response> {
    let query = QueryText::new(query).map_err(invalid_request)?;
    let mut key_set = BTreeSet::new();
    for key in keys {
        key_set.insert(Key::new(key).map_err(invalid_request)?);
    }
    let context = match context {
        Some(context) => Some(Context::new(context).map_err(invalid_request)?),
        None => None,
    };
    Ok((query, key_set, context))
}

fn invalid_request(error: Error) -> Response {
    Response::Error(ProtocolError::InvalidRequest(error.to_string()))
}

fn backend_error(error: Error) -> Response {
    let protocol_error = match error {
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
        | Error::InvalidConfig(_) => ProtocolError::InvalidRequest(error.to_string()),
        Error::NamespaceMissing(namespace) => ProtocolError::UnknownNamespace(namespace),
        Error::ProtocolVersionMismatch { client, daemon } => {
            ProtocolError::VersionMismatch { client, daemon }
        }
        Error::EntityExtraction(_)
        | Error::Embedding(_)
        | Error::VectorStorage(_)
        | Error::ObjectStorage(_)
        | Error::DaemonShutdown
        | Error::Daemon(_)
        | Error::Io(_)
        | Error::Codec(_) => ProtocolError::BackendInit(error.to_string()),
    };
    Response::Error(protocol_error)
}

async fn send(framed: &mut Conn, response: &Response) -> Result<()> {
    let body = protocol::encode(response)?;
    framed.send(Bytes::from(body)).await?;
    Ok(())
}
