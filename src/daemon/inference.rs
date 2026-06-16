//! A dedicated blocking worker pool for the CPU/IO-bound read path (`Get`/`Del`) and
//! the slow namespace build (model load). Worker threads live off the tokio runtime,
//! so a slow embed or disk read never starves the accept loop. `run` bridges a
//! blocking closure to async via a oneshot.

use std::thread;

use tokio::sync::oneshot;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub(super) struct InferencePool {
    sender: crossbeam_channel::Sender<Job>,
}

impl InferencePool {
    pub(super) fn new() -> Self {
        let workers = thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1);
        let (sender, receiver) = crossbeam_channel::unbounded::<Job>();
        for _ in 0..workers {
            let receiver = receiver.clone();
            thread::spawn(move || {
                while let Ok(job) = receiver.recv() {
                    job();
                }
            });
        }
        Self { sender }
    }

    pub(super) async fn run<F, R>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let job: Job = Box::new(move || {
            let _ = tx.send(work());
        });
        match self.sender.send(job) {
            Ok(()) => match rx.await {
                Ok(value) => value,
                // The worker dropped the result channel without sending: only reachable
                // once the pool's threads are torn down at daemon shutdown.
                Err(_) => std::future::pending::<R>().await,
            },
            // Every worker thread is gone: the daemon is shutting down around us.
            Err(_) => std::future::pending::<R>().await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_closure_off_runtime_and_returns_value() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let pool = InferencePool::new();
        let value = runtime.block_on(pool.run(|| 6 * 7));
        assert_eq!(value, 42);
    }
}
