use std::env;

fn get_env_var(key: &str, default: &str) -> String {
    env::var(key).unwrap_or(default.to_string())
}
fn get_env_var_usize(key: &str, default: usize) -> usize {
    env::var(key)
        .unwrap_or(default.to_string())
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("Could not parse {}", key))
}

#[derive(Clone, Debug)]
pub struct ApplicationConfig {
    pub host: String,
    pub port: String,
    pub server_workers: usize,

    pub inference_config: InferenceConfig,
    pub runner_config: RunnerConfig,
    pub alpha_zero_config: alphazero_nn::Config,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InferenceConfig {
    pub max_batch_size: usize,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            max_batch_size: 200,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub threads: usize,
    pub parallel_games: usize,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig {
            threads: 6,
            parallel_games: 200,
        }
    }
}

impl ApplicationConfig {
    pub fn load() -> Self {
        let host = get_env_var("HOST", "127.0.0.1");
        let port = get_env_var("PORT", "8080");
        let server_workers = get_env_var_usize("WORKERS", 4);

        ApplicationConfig {
            host,
            port,
            server_workers,
            inference_config: create_batcher_config(),
            runner_config: create_runner_config(),
            alpha_zero_config: create_alpha_zero_config(),
        }
    }
}

fn create_batcher_config() -> InferenceConfig {
    let max_batch_size = get_env_var_usize("MAX_BATCH_SIZE", 200);
    InferenceConfig { max_batch_size }
}

fn create_runner_config() -> RunnerConfig {
    let threads = get_env_var_usize("PLAY_CORES", 6);
    let parallel_games = get_env_var_usize("PARALLEL_GAMES", 200);
    RunnerConfig {
        threads,
        parallel_games,
    }
}

#[cfg(feature = "chess")]
fn create_alpha_zero_config() -> alphazero_nn::Config {
    use alphazero_nn::AlphaGame as _;

    let n_input_channels = get_env_var_usize("N_INPUT_CHANNELS", 17);
    let n_residual_blocks = get_env_var_usize("N_RESIDUAL_BLOCKS", 10);
    let kernel_size = get_env_var_usize("KERNEL_SIZE", 3);
    let n_filters = get_env_var_usize("N_FILTERS", 256);

    alphazero_nn::Config {
        n_input_channels,
        n_residual_blocks,
        kernel_size,
        game_shape: candle_core::shape::Shape::from_dims(&[8, 8]),
        n_filters,
        policy_output_type: crate::runner::chess::Chess::policy_output_type(),
    }
}
