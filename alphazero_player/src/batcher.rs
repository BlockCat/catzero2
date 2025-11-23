use candle_core::{Tensor, D};
use tokio::task::JoinHandle;

pub struct BatcherHandle(JoinHandle<Result<(), anyhow::Error>>);

#[derive(Debug)]
pub struct InferenceRequest {
    pub state_tensor: Tensor,
    pub response_channel: tokio::sync::oneshot::Sender<InferenceResponse>,
}

#[derive(Clone, Debug)]
pub struct InferenceResponse {
    pub output_tensor: Tensor,
    pub value: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct BatcherConfig {
    pub max_wait: std::time::Duration,
    pub max_batch_size: usize,
    pub min_batch_size: usize,
}

pub struct BatchService {
    config: BatcherConfig,
    request_receiver: tokio::sync::mpsc::Receiver<InferenceRequest>,
    model: alphazero_nn::AlphaZeroNN,
}

impl BatchService {
    pub fn new(
        config: BatcherConfig,
        model: alphazero_nn::AlphaZeroNN,
    ) -> (Self, tokio::sync::mpsc::Sender<InferenceRequest>) {
        use tokio::sync::mpsc;
        let (tx, rx) = mpsc::channel(config.max_batch_size * 2);

        (
            BatchService {
                config,
                request_receiver: rx,
                model,
            },
            tx,
        )
    }

    pub fn start(mut self) -> BatcherHandle {
        BatcherHandle(tokio::spawn(async move { self.start_async().await }))
    }

    pub async fn start_async(&mut self) -> Result<(), anyhow::Error> {
        tracing::info!("Batcher started");
        let max_batch_size = self.config.max_batch_size;
        let min_batch_size = self.config.min_batch_size;
        assert!(min_batch_size <= max_batch_size);
        let receiver = &mut self.request_receiver;

        loop {
            tracing::info!("Batcher waiting for requests");
            let mut requests = Vec::with_capacity(max_batch_size);

            while requests.len() < min_batch_size {
                let rl = requests.len();
                receiver.recv_many(&mut requests, max_batch_size - rl).await;
            }

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
                let response = InferenceResponse {
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
    }
}

impl Drop for BatcherHandle {
    fn drop(&mut self) {
        tracing::info!("BatchService stopping, quitting batcher task");
        self.0.abort();
        
    }
}

impl Drop for BatchService {
    fn drop(&mut self) {
        tracing::info!("BatchService stopped");
    }
}