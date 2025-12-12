use candle_core::{Result, Shape, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, ModuleT, VarBuilder, batch_norm,
    conv2d, linear, linear_no_bias,
};

pub struct Config {
    pub n_input_channels: usize,
    pub n_residual_blocks: usize,
    pub n_filters: usize,

    pub kernel_size: usize,
    pub game_shape: Shape,
    pub policy_output_type: PolicyOutputType,
}

impl Config {
    pub fn new(
        n_input_channels: usize,
        n_residual_blocks: usize,
        n_filters: usize,
        kernel_size: usize,
        game_shape: Shape,
        policy_output_type: PolicyOutputType,
    ) -> Self {
        Self {
            n_input_channels,
            n_residual_blocks,
            n_filters,
            kernel_size,
            game_shape,
            policy_output_type,
        }
    }
}

#[derive(Debug)]
pub struct AlphaZeroNN {
    input_block: InputBlock,
    residual_blocks: Vec<ResidualBlock>,

    policy_head: PolicyHead,
    value_head: ValueHead,
}

impl AlphaZeroNN {
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let input_block = InputBlock::new(
            config.n_input_channels,
            config.n_filters,
            config.kernel_size,
            vb.pp("input_block"),
        )?;

        let mut residual_blocks = Vec::with_capacity(config.n_residual_blocks);
        for i in 0..config.n_residual_blocks {
            residual_blocks.push(ResidualBlock::new(
                config.n_filters,
                config.kernel_size,
                vb.pp(format!("residual_block_{}", i)),
            )?);
        }

        let policy_head = PolicyHead::new(
            config.n_filters,
            config.policy_output_type,
            config.kernel_size,
            &config.game_shape,
            vb.pp("policy_head"),
        )?;
        let value_head = ValueHead::new(config.n_filters, &config.game_shape, vb.pp("value_head"))?;

        Ok(Self {
            input_block,
            residual_blocks,
            policy_head,
            value_head,
        })
    }

    pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let mut x = input.apply_t(&self.input_block, train)?;

        for block in &self.residual_blocks {
            x = block.forward_t(&x, train)?;
        }

        let output = self.policy_head.forward_t(&x, train)?;
        let value = self.value_head.forward_t(&x, train)?;

        Ok((output, value))
    }
}

#[derive(Debug)]
struct InputBlock {
    input: Conv2d,
    initial_conv: Conv2d,
    initial_batch_norm: BatchNorm,
}

impl InputBlock {
    fn new(
        input_channel: usize,
        output_channel: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(kernel_size % 2 == 1, "kernel_size must be odd");
        let conv2d_config = Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };

        let input = candle_nn::conv::conv2d(
            input_channel,
            output_channel,
            kernel_size,
            conv2d_config,
            vb.pp("input"),
        )?;

        let initial_conv = candle_nn::conv::conv2d(
            output_channel,
            output_channel,
            kernel_size,
            conv2d_config,
            vb.pp("initial_conv"),
        )?;

        let initial_batch_norm = batch_norm(
            output_channel,
            BatchNormConfig {
                ..Default::default()
            },
            vb.pp("initial_batch_norm"),
        )?;

        Ok(Self {
            input,
            initial_conv,
            initial_batch_norm,
        })
    }
}

impl ModuleT for InputBlock {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        input
            .apply_t(&self.input, train)?
            .apply_t(&self.initial_conv, train)?
            .apply_t(&self.initial_batch_norm, train)?
            .relu()
    }
}
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    batch_norm1: BatchNorm,
    conv2: Conv2d,
    batch_norm2: BatchNorm,
}

impl ResidualBlock {
    fn new(n_filters: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        let batch_norm_config = BatchNormConfig {
            ..Default::default()
        };
        let conv1 = candle_nn::conv::conv2d(
            n_filters,
            n_filters,
            kernel_size,
            conv_config,
            vb.pp("conv1"),
        )?;

        let batch_norm1 =
            candle_nn::batch_norm::batch_norm(n_filters, batch_norm_config, vb.pp("batch_norm1"))?;

        let conv2 = candle_nn::conv::conv2d(
            n_filters,
            n_filters,
            kernel_size,
            conv_config,
            vb.pp("conv2"),
        )?;
        let batch_norm2 =
            candle_nn::batch_norm::batch_norm(n_filters, batch_norm_config, vb.pp("batch_norm2"))?;

        Ok(Self {
            conv1,
            batch_norm1,
            conv2,
            batch_norm2,
        })
    }
}

impl ModuleT for ResidualBlock {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let x = input
            .apply_t(&self.conv1, train)?
            .apply_t(&self.batch_norm1, train)?
            .relu()?
            .apply_t(&self.conv2, train)?
            .apply_t(&self.batch_norm2, train)?;
        let x = (input + x)?;
        x.relu()
    }
}

#[derive(Debug)]
pub enum PolicyOutputType {
    Flat(usize),
    Conv(usize),
}

#[derive(Debug)]
enum PolicyOutputLayer {
    Flat(Linear),
    Conv(Conv2d),
}

#[derive(Debug)]
struct PolicyHead {
    conv1: Conv2d,
    batch_norm1: BatchNorm,
    output: PolicyOutputLayer,
}

impl PolicyHead {
    fn new(
        features: usize,
        output: PolicyOutputType,
        kernel_size: usize,
        shape: &Shape,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        let conv1 = conv2d(features, features, kernel_size, cfg, vb.pp("conv1"))?;

        let batch_norm1 = batch_norm(
            features,
            BatchNormConfig {
                ..Default::default()
            },
            vb.pp("batch_norm1"),
        )?;

        let output = match output {
            PolicyOutputType::Flat(size) => {
                let input_dim = features * shape.elem_count();
                PolicyOutputLayer::Flat(linear(input_dim, size, vb.pp("linear"))?)
            }
            PolicyOutputType::Conv(channels) => PolicyOutputLayer::Conv(conv2d(
                features,
                channels,
                kernel_size,
                cfg,
                vb.pp("conv2"),
            )?),
        };

        Ok(Self {
            conv1,
            batch_norm1,
            output,
        })
    }
}

impl ModuleT for PolicyHead {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let x = input
            .apply_t(&self.conv1, train)?
            .apply_t(&self.batch_norm1, train)?
            .relu()?;

        match &self.output {
            PolicyOutputLayer::Flat(fc) => x.flatten_from(1)?.apply_t(fc, train),
            PolicyOutputLayer::Conv(conv) => x.apply_t(conv, train),
        }
    }
}

#[derive(Debug)]
struct ValueHead {
    conv1: Conv2d,
    batch_norm1: BatchNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl ValueHead {
    fn new(channels: usize, shape: &Shape, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d(
            channels,
            1,
            1,
            Conv2dConfig {
                stride: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let batch_norm1 = batch_norm(
            1,
            BatchNormConfig {
                ..Default::default()
            },
            vb.pp("batch_norm1"),
        )?;

        let fc1 = linear_no_bias(shape.elem_count(), 256, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(256, 1, vb.pp("fc2"))?;

        Ok(Self {
            conv1,
            batch_norm1,
            fc1,
            fc2,
        })
    }
}

impl ModuleT for ValueHead {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        input
            .apply_t(&self.conv1, train)?
            .apply_t(&self.batch_norm1, train)?
            .relu()?
            .flatten_from(1)?
            .apply_t(&self.fc1, train)?
            .relu()?
            .apply_t(&self.fc2, train)?
            .tanh()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use candle_core::{DType, Device, Shape};
    use candle_nn::VarMap;

    #[test]
    fn test_alphazero_nn_forward() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = Config::new(
            119,
            19,
            256,
            3,
            Shape::from_dims(&[8, 8]),
            PolicyOutputType::Conv(73),
        );

        let model = AlphaZeroNN::new(config, vs)?;

        let batch_size = 500;
        // 5 seconds per move.

        let start = Instant::now();

        let input = Tensor::randn(1.0f32, 2.0, &[batch_size, 119, 8, 8], &device)?;

        println!("Input tensor created in {:?}", start.elapsed());

        let (policy_output, value_output) = model.forward_t(&input, false)?;

        println!("Model forward pass completed in {:?}", start.elapsed());

        assert_eq!(
            policy_output.shape(),
            &Shape::from_dims(&[batch_size, 73, 8, 8])
        );
        assert_eq!(value_output.shape(), &Shape::from_dims(&[batch_size, 1]));

        Ok(())
    }

    #[test]
    fn test_residual_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = ResidualBlock::new(64, 3, vs)?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = block.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 64, 8, 8]));

        Ok(())
    }

    #[test]
    fn test_input_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = InputBlock::new(17, 64, 5, vs)?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 17, 8, 8], &device)?;
        let output = block.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 64, 8, 8]));

        Ok(())
    }

    #[test]
    fn test_policy_head_forward_flat() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = PolicyHead::new(
            64,
            PolicyOutputType::Flat(362),
            3,
            &Shape::from_dims(&[8, 8]),
            vs,
        )?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = head.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 362]));

        Ok(())
    }

    #[test]
    fn test_policy_head_forward_conv() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = PolicyHead::new(
            64,
            PolicyOutputType::Conv(73),
            3,
            &Shape::from_dims(&[8, 8]),
            vs,
        )?;
        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = head.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 73, 8, 8]));
        Ok(())
    }

    #[test]
    fn test_value_head_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = ValueHead::new(64, &Shape::from_dims(&[8, 8]), vs)?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = head.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 1]));

        println!("Value head output: {:?}", output.to_vec2::<f32>()?);
        Ok(())
    }
}
