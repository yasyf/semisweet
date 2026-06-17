//! The daemon IPC wire protocol: message types and length-delimited framing.
//!
//! Pure Rust, no pyo3. Messages carry primitives (`String`/`Vec<String>`/bytes);
//! the daemon reconstructs the validating newtypes (`QueryText::new`, etc.) in
//! Phase 3b, so the validating newtypes deliberately do not derive serde here.

use std::io::{Read, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use uuid::Uuid;

use crate::error::{Error, Result};

pub const PROTOCOL_VERSION: u32 = 1;

pub(crate) const MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

// read_frame grows the body buffer one chunk at a time rather than allocating the whole
// declared length up front, so a peer that announces a large frame but never sends it
// can't force that allocation before a byte arrives. MAX_FRAME_BYTES still caps the total.
const READ_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClientId(Uuid);

impl ClientId {
    pub fn generate() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn as_uuid(self) -> Uuid {
        self.0
    }
}

impl Serialize for ClientId {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.0.as_bytes().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ClientId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let bytes = <[u8; 16]>::deserialize(deserializer)?;
        Ok(Self(Uuid::from_bytes(bytes)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Request {
    Hello {
        client: ClientId,
        protocol: u32,
        pid: u32,
    },
    RegisterNamespace {
        namespace: String,
        config_json: String,
    },
    Get {
        namespace: String,
        query: String,
        keys: Vec<String>,
        context: Option<String>,
    },
    Set {
        namespace: String,
        query: String,
        keys: Vec<String>,
        context: Option<String>,
        value: serde_bytes::ByteBuf,
    },
    Del {
        namespace: String,
        query: String,
        keys: Vec<String>,
        context: Option<String>,
    },
    Ping,
    Bye,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Response {
    Welcome {
        daemon_version: String,
        protocol: u32,
    },
    Registered,
    Value(Option<serde_bytes::ByteBuf>),
    Accepted(bool),
    Deleted(bool),
    Pong,
    Goodbye,
    Error(ProtocolError),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProtocolError {
    VersionMismatch { client: u32, daemon: u32 },
    UnknownNamespace(String),
    InvalidRequest(String),
    BackendInit(String),
}

pub fn encode<T: Serialize>(message: &T) -> Result<Vec<u8>> {
    Ok(postcard::to_stdvec(message)?)
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    Ok(postcard::from_bytes(bytes)?)
}

pub fn write_frame<W: Write, T: Serialize>(writer: &mut W, message: &T) -> Result<()> {
    let body = encode(message)?;
    if body.len() > MAX_FRAME_BYTES {
        return Err(Error::Daemon(format!(
            "frame too large: {} bytes (max {})",
            body.len(),
            MAX_FRAME_BYTES
        )));
    }
    let len = u32::try_from(body.len())
        .map_err(|_| Error::Daemon(format!("frame too large: {} bytes", body.len())))?;
    writer.write_all(&len.to_be_bytes())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

fn read_body<R: Read>(reader: &mut R, len: usize) -> Result<Vec<u8>> {
    let mut body = Vec::with_capacity(len.min(READ_CHUNK_BYTES));
    while body.len() < len {
        let start = body.len();
        let want = (len - start).min(READ_CHUNK_BYTES);
        body.resize(start + want, 0);
        reader.read_exact(&mut body[start..])?;
    }
    Ok(body)
}

pub fn read_frame<R: Read, T: DeserializeOwned>(reader: &mut R) -> Result<T> {
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(Error::Daemon(format!("frame too large: {len} bytes")));
    }
    let body = read_body(reader, len)?;
    decode(&body)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, ErrorKind};

    use super::*;

    fn requests() -> Vec<Request> {
        let client = ClientId::generate();
        vec![
            Request::Hello {
                client,
                protocol: PROTOCOL_VERSION,
                pid: 4242,
            },
            Request::RegisterNamespace {
                namespace: "prod".to_owned(),
                config_json: r#"{"backend":"memory"}"#.to_owned(),
            },
            Request::Get {
                namespace: "prod".to_owned(),
                query: "what is aspirin".to_owned(),
                keys: vec!["patient1".to_owned(), "patient2".to_owned()],
                context: Some("clinical".to_owned()),
            },
            Request::Set {
                namespace: "prod".to_owned(),
                query: "what is aspirin".to_owned(),
                keys: vec!["patient1".to_owned()],
                context: None,
                value: serde_bytes::ByteBuf::from(vec![0u8, 1, 2, 255]),
            },
            Request::Del {
                namespace: "prod".to_owned(),
                query: "q".to_owned(),
                keys: vec![],
                context: None,
            },
            Request::Ping,
            Request::Bye,
        ]
    }

    fn responses() -> Vec<Response> {
        vec![
            Response::Welcome {
                daemon_version: "0.1.0".to_owned(),
                protocol: PROTOCOL_VERSION,
            },
            Response::Registered,
            Response::Value(Some(serde_bytes::ByteBuf::from(vec![9u8, 8, 7]))),
            Response::Value(None),
            Response::Accepted(true),
            Response::Deleted(false),
            Response::Pong,
            Response::Goodbye,
            Response::Error(ProtocolError::VersionMismatch {
                client: 2,
                daemon: 1,
            }),
            Response::Error(ProtocolError::UnknownNamespace("nope".to_owned())),
            Response::Error(ProtocolError::InvalidRequest("bad query".to_owned())),
            Response::Error(ProtocolError::BackendInit("boom".to_owned())),
        ]
    }

    #[test]
    fn every_request_round_trips_through_encode_decode() {
        for req in requests() {
            let bytes = encode(&req).unwrap();
            let back: Request = decode(&bytes).unwrap();
            assert_eq!(req, back);
        }
    }

    #[test]
    fn every_response_round_trips_through_encode_decode() {
        for resp in responses() {
            let bytes = encode(&resp).unwrap();
            let back: Response = decode(&bytes).unwrap();
            assert_eq!(resp, back);
        }
    }

    #[test]
    fn frames_round_trip_over_a_reader_writer() {
        let sent = requests();
        let mut buf: Vec<u8> = Vec::new();
        for req in &sent {
            write_frame(&mut buf, req).unwrap();
        }
        let mut cursor = Cursor::new(buf);
        for expected in &sent {
            let got: Request = read_frame(&mut cursor).unwrap();
            assert_eq!(*expected, got);
        }
    }

    #[test]
    fn a_frame_larger_than_a_chunk_round_trips() {
        // A body spanning several READ_CHUNK_BYTES exercises read_frame's grow-by-chunk loop.
        let big = Request::Set {
            namespace: "prod".to_owned(),
            query: "q".to_owned(),
            keys: vec![],
            context: None,
            value: serde_bytes::ByteBuf::from(vec![0xABu8; READ_CHUNK_BYTES * 3 + 7]),
        };
        let mut buf: Vec<u8> = Vec::new();
        write_frame(&mut buf, &big).unwrap();
        let mut cursor = Cursor::new(buf);
        let got: Request = read_frame(&mut cursor).unwrap();
        assert_eq!(big, got);
    }

    #[test]
    fn read_frame_errors_on_a_truncated_body() {
        // Header announces 1024 bytes but only 10 follow: the short read must surface as
        // Error::Io(UnexpectedEof), never a partial frame or a hang.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&1024u32.to_be_bytes());
        buf.extend_from_slice(&[0u8; 10]);
        let mut cursor = Cursor::new(buf);
        let err = read_frame::<_, Request>(&mut cursor).unwrap_err();
        assert!(
            matches!(err, Error::Io(ref e) if e.kind() == ErrorKind::UnexpectedEof),
            "expected Error::Io(UnexpectedEof), got {err:?}"
        );
    }

    #[test]
    fn write_frame_rejects_a_body_exceeding_the_max() {
        let oversized = Request::Set {
            namespace: "prod".to_owned(),
            query: "q".to_owned(),
            keys: vec![],
            context: None,
            value: serde_bytes::ByteBuf::from(vec![0u8; MAX_FRAME_BYTES + 1]),
        };
        let mut buf: Vec<u8> = Vec::new();
        let err = write_frame(&mut buf, &oversized).unwrap_err();
        assert!(
            matches!(err, Error::Daemon(ref msg) if msg.contains("frame too large")),
            "expected Error::Daemon frame-too-large, got {err:?}"
        );
        assert!(
            buf.is_empty(),
            "write_frame must fail before writing the length prefix"
        );
    }

    #[test]
    fn client_id_round_trips_without_uuid_serde_feature() {
        let id = ClientId::generate();
        let bytes = encode(&id).unwrap();
        let back: ClientId = decode(&bytes).unwrap();
        assert_eq!(id, back);
    }
}
