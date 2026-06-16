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
                    // A panicking job drops its oneshot `tx`, so the awaiting handler
                    // already unblocks with `Err(DaemonShutdown)`; catching here keeps the
                    // worker thread alive so one bad job can't permanently kill it.
                    if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))
                    {
                        eprintln!("semisweet-daemon: inference job panicked: {panic:?}");
                        continue;
                    }
                }
            });
        }
        Self { sender }
    }

    pub(super) async fn run<F, R>(&self, work: F) -> crate::error::Result<R>
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
                Ok(value) => Ok(value),
                // The worker dropped the result channel without sending (a torn-down pool
                // at shutdown, or a panicked job): surface it as a clean shutdown error.
                Err(_) => Err(crate::error::Error::DaemonShutdown),
            },
            // Every worker thread is gone: the daemon is shutting down around us.
            Err(_) => Err(crate::error::Error::DaemonShutdown),
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
        let value = runtime.block_on(pool.run(|| 6 * 7)).unwrap();
        assert_eq!(value, 42);
    }

    #[test]
    fn panicking_job_surfaces_shutdown_and_worker_survives() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let pool = InferencePool::new();

        let panicked = runtime.block_on(pool.run(|| -> i32 { panic!("boom") }));
        assert!(matches!(panicked, Err(crate::error::Error::DaemonShutdown)));

        // The worker survived the panic, so a subsequent job still completes.
        let value = runtime.block_on(pool.run(|| 6 * 7)).unwrap();
        assert_eq!(value, 42);
    }
}
