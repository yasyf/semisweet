//! The client side of the daemon protocol: lazy spawn + a persistent stub.

mod spawn;
mod stub;

pub use spawn::Launcher;
pub use stub::{ClientStub, connect_or_spawn};
