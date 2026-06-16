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

use super::state::{DaemonState, NamespaceEntry};
use super::supervisor::ClientEvent;
use super::writer::WriteJob;
use crate::error::{Error, Result};
use crate::newtype::{Context, Key, QueryText};
use crate::protocol::{self, MAX_FRAME_BYTES, ProtocolError, Request, Response};
use crate::registry::{self, NamespaceConfig};

type Conn = Framed<UnixStream, LengthDelimitedCodec>;
type Parts = (QueryText, BTreeSet<Key>, Option<Context>);

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
    let _ = events.send(ClientEvent::Connected);
    let _ = message_loop(&mut framed, &state, &daemon_version, protocol).await;
    let _ = events.send(ClientEvent::Disconnected);
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
    match state.resolve(&namespace) {
        Ok(Some(_)) => return Response::Registered { ready: true },
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
        Ok(cache) => {
            let entry = Arc::new(NamespaceEntry {
                cache: Arc::new(cache),
            });
            match state.register(namespace, entry) {
                Ok(()) => Response::Registered { ready: true },
                Err(error) => backend_error(error),
            }
        }
        Err(error) => Response::Error(ProtocolError::BackendInit(error.to_string())),
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
    let cache = entry.cache.clone();
    match state
        .inference
        .run(move || cache.get(&query, &keys, &context))
        .await
    {
        Ok(value) => Response::Value(value.map(serde_bytes::ByteBuf::from)),
        Err(error) => backend_error(error),
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
    let job = WriteJob {
        cache: entry.cache.clone(),
        query,
        keys,
        context,
        value,
    };
    Response::Accepted(state.writer.try_push(job))
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
    let cache = entry.cache.clone();
    match state
        .inference
        .run(move || cache.delete(&query, &keys, &context))
        .await
    {
        Ok(deleted) => Response::Deleted(deleted),
        Err(error) => backend_error(error),
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
    Response::Error(ProtocolError::BackendInit(error.to_string()))
}

async fn send(framed: &mut Conn, response: &Response) -> Result<()> {
    let body = protocol::encode(response)?;
    framed.send(Bytes::from(body)).await?;
    Ok(())
}
