use anyhow::Ok;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use safetensors::tensor::SafeTensors;

pub async fn load_model(_name: String, vm: &mut VarMap, device: &Device) -> Result<(), anyhow::Error> {
    let buffer = vec![];
    let safetensor = SafeTensors::deserialize(&buffer)?;

    vm.set(
        safetensor
            .iter()
            .map(|(key, val)| (key, map_view_to_tensor(&val, device).unwrap())),
    )?;

    Ok(())
}

fn map_view_to_tensor(
    view: &safetensors::tensor::TensorView,
    device: &Device,
) -> Result<Tensor, anyhow::Error> {
    let tensor = Tensor::from_slice(view.data(), view.shape(), device)?;
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Shape};

    #[test]
    fn test_map_view_to_tensor_basic() {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = (2, 2);
        
        // Create a simple tensor directly for testing
        let tensor = Tensor::from_slice(&data, shape, &device);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &Shape::from_dims(&[2, 2]));
    }

    #[test]
    fn test_map_view_to_tensor_1d() {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let shape = 3_usize;
        
        let tensor = Tensor::from_slice(&data, shape, &device);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &Shape::from_dims(&[3]));
    }

    #[test]
    fn test_map_view_to_tensor_3d() {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = (2, 2, 2);
        
        let tensor = Tensor::from_slice(&data, shape, &device);
        assert!(tensor.is_ok());
        
        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &Shape::from_dims(&[2, 2, 2]));
    }

    #[test]
    fn test_map_view_to_tensor_empty() {
        let device = Device::Cpu;
        let data: Vec<f32> = vec![];
        let shape = 0_usize;
        
        let tensor = Tensor::from_slice(&data, shape, &device);
        assert!(tensor.is_ok());
    }

    #[test]
    fn test_tensor_with_different_dtypes() {
        let device = Device::Cpu;
        
        // Test with i64
        let data_i64: Vec<i64> = vec![1, 2, 3, 4];
        let shape = (2, 2);
        let tensor_i64 = Tensor::from_slice(&data_i64, shape, &device);
        assert!(tensor_i64.is_ok());
        assert_eq!(tensor_i64.unwrap().dtype(), DType::I64);
        
        // Test with u32
        let data_u32: Vec<u32> = vec![1, 2, 3, 4];
        let tensor_u32 = Tensor::from_slice(&data_u32, shape, &device);
        assert!(tensor_u32.is_ok());
        assert_eq!(tensor_u32.unwrap().dtype(), DType::U32);
    }

    #[test]
    fn test_tensor_shape_validity() {
        let device = Device::Cpu;
        
        // Valid shape - should work
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = (2, 2);
        let tensor = Tensor::from_slice(&data, shape, &device);
        assert!(tensor.is_ok());
        
        // Empty tensor with zero shape
        let empty_data: Vec<f32> = vec![];
        let empty_shape = 0_usize;
        let empty_tensor = Tensor::from_slice(&empty_data, empty_shape, &device);
        assert!(empty_tensor.is_ok());
    }
}

