use crate::config::BatcherConfig;
use alphazero_nn::AlphaZeroNN;
use candle_core::{Tensor, D};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
pub struct BatcherHandle(JoinHandle<Result<(), anyhow::Error>>, CancellationToken);

impl BatcherHandle {
    pub fn abort(&self) {
        tracing::info!("BatcherHandle aborting batcher task");
        self.1.cancel();
        self.0.abort();
    }
}

#[derive(Debug)]
pub struct BatcherRequest {
    pub state_tensor: Tensor,
    pub response_channel: tokio::sync::oneshot::Sender<BatcherResponse>,
}

#[derive(Debug)]
pub struct BatcherResponse {
    pub output_tensor: Tensor,
    pub value: f32,
}

pub struct BatchService {
    config: BatcherConfig,
    cancellation_token: tokio_util::sync::CancellationToken,
    request_receiver: tokio::sync::mpsc::Receiver<BatcherRequest>,
    model: AlphaZeroNN,
}

impl BatchService {
    pub fn new(
        config: BatcherConfig,
        model: AlphaZeroNN,
    ) -> (Self, tokio::sync::mpsc::Sender<BatcherRequest>) {
        use tokio::sync::mpsc;
        let (tx, rx) = mpsc::channel(config.max_batch_size * 2);
        let cancellation_token = tokio_util::sync::CancellationToken::new();
        (
            BatchService {
                config,
                cancellation_token,
                request_receiver: rx,
                model,
            },
            tx,
        )
    }

    pub fn start(mut self) -> BatcherHandle {
        let ct = self.cancellation_token.clone();
        let ct2 = self.cancellation_token.clone();
        BatcherHandle(tokio::spawn(async move { self.start_async(ct).await }), ct2)
    }

    pub async fn start_async(
        &mut self,
        cancellation_token: CancellationToken,
    ) -> Result<(), anyhow::Error> {
        tracing::info!("Batcher started");
        let max_batch_size = self.config.max_batch_size;

        let receiver = &mut self.request_receiver;

        while !cancellation_token.is_cancelled() {
            tracing::info!("Batcher waiting for requests");
            let mut requests = Vec::with_capacity(max_batch_size);

            receiver.recv_many(&mut requests, max_batch_size).await;

            tracing::info!("Batcher received {} requests", requests.len());

            let request_tensors = requests
                .iter()
                .map(|r| r.state_tensor.clone())
                .collect::<Vec<_>>();
            let batched_input = Tensor::stack(request_tensors.as_slice(), 0)?;

            let (policy, value) = self.model.forward_t(&batched_input, false)?;

            requests.into_iter().enumerate().for_each(|(i, req)| {
                let policy = policy.get(i).expect("Failed to get policy tensor");
                let value = value.get(i).expect("Failed to get value tensor");
                let response = BatcherResponse {
                    output_tensor: policy,
                    value: value
                        .squeeze(D::Minus1)
                        .expect("Failed to squeeze value tensor")
                        .to_scalar::<f32>()
                        .expect("Failed to convert value tensor to scalar"),
                };

                req.response_channel
                    .send(response)
                    .expect("Failed to send response");
            });
        }

        Ok(())
    }
}

impl Drop for BatcherHandle {
    fn drop(&mut self) {
        self.abort();
    }
}
