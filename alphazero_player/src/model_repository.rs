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
