use burn::module::Param;
use burn::prelude::*;
use safetensors::SafeTensors;
use burn_mamba::prelude::*;

#[cfg(feature = "mamba1")]
pub fn safetensors_load_mamba1(
    mamba_safetensors_bytes: &[u8],
    mamba_config: MambaVocabNetConfig,
    device: &Device,
) -> anyhow::Result<MambaVocabNet> {
    let mut outer_mamba: MambaVocabNet = mamba_config.init(&device);
    #[allow(unreachable_patterns)]
    let mamba = match outer_mamba {
        MambaVocabNet::Mamba1(ref mut inner) => inner,
        _ => panic!(),
    };
    let tensors = &safetensors::SafeTensors::deserialize(&mamba_safetensors_bytes)?;
    // log::info!("{:?}", tensors.names());
    //

    let name = |n: &str| format!("backbone.{n}");
    load_param_f32_to_f32(
        &mut mamba.embedding.weight,
        name("embedding.weight"),
        tensors,
        device,
        false,
    )?;
    load_param_f32_to_f32(
        &mut mamba.norm_f.gamma,
        name("norm_f.weight"),
        tensors,
        device,
        false,
    )?;

    #[allow(unreachable_patterns)]
    let n_layers = match mamba_config {
        MambaVocabNetConfig::Mamba1 {n_real_layers, ..} => n_real_layers,
        _ => panic!(),
    };

    for i in 0..n_layers {
        let layer = &mut mamba.layers.real_layers[i];
        let name = |n: &str| format!("backbone.layers.{i}.{n}");
        load_param_f32_to_f32(
            &mut layer.norm.gamma,
            name("norm.weight"),
            tensors,
            device,
            false,
        )?;
        let name = |n: &str| format!("backbone.layers.{i}.mixer.{n}");
        let mamba_block = &mut layer.mamba_block;
        load_param_f32_to_f32(
            &mut mamba_block.a_log,
            name("A_log"),
            tensors,
            device,
            false,
        )?;
        load_param_f32_to_f32(&mut mamba_block.d, name("D"), tensors, device, false)?;
        load_param_f32_to_f32(
            &mut mamba_block.conv1d.weight,
            name("conv1d.weight"),
            tensors,
            device,
            false,
        )?;
        load_param_f32_to_f32(
            &mut mamba_block.conv1d.bias.as_mut().unwrap(),
            name("conv1d.bias"),
            tensors,
            device,
            false,
        )?;
        load_param_f32_to_f32(
            &mut mamba_block.dt_proj.weight,
            name("dt_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param_f32_to_f32(
            &mut mamba_block.dt_proj.bias.as_mut().unwrap(),
            name("dt_proj.bias"),
            tensors,
            device,
            false,
        )?;
        load_param_f32_to_f32(
            &mut mamba_block.in_proj.weight,
            name("in_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param_f32_to_f32(
            &mut mamba_block.out_proj.weight,
            name("out_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param_f32_to_f32(
            &mut mamba_block.x_proj.weight,
            name("x_proj.weight"),
            tensors,
            device,
            true,
        )?;
    }

    let param = mamba.embedding.weight.val();
    let param = param.swap_dims(1, 0);
    // ensure the tensor is contiguous
    let param: Tensor<2> = Tensor::from_data(param.into_data(), device);

    mamba.lm_head = Some(burn::nn::Linear {
        weight: Param::from_tensor(param),
        bias: None,
    });

    Ok(outer_mamba)
}

#[allow(dead_code)]
#[cfg(feature = "mamba2")]
pub fn safetensors_load_mamba2(
    mamba_safetensors_bytes: &[u8],
    mamba_config: MambaVocabNetConfig,
    device: &Device,
) -> anyhow::Result<MambaVocabNet> {
    let mut outer_mamba: MambaVocabNet = mamba_config.init(&device);
    #[allow(unreachable_patterns)]
    let mamba = match outer_mamba {
        MambaVocabNet::Mamba2(ref mut inner) => inner,
        _ => panic!(),
    };
    let tensors = &safetensors::SafeTensors::deserialize(&mamba_safetensors_bytes)?;
    // log::info!("{:?}", tensors.names());
    //

    let name = |n: &str| format!("backbone.{n}");
    load_param_f16_to_f32(
        &mut mamba.embedding.weight,
        name("embedding.weight"),
        tensors,
        device,
        false,
    )?;
    load_param_f16_to_f32(
        &mut mamba.norm_f.gamma,
        name("norm_f.weight"),
        tensors,
        device,
        false,
    )?;

    #[allow(unreachable_patterns)]
    let n_layers = match mamba_config {
        MambaVocabNetConfig::Mamba2 {n_real_layers, ..} => n_real_layers,
        _ => panic!(),
    };

    for i in 0..n_layers {
        // note: only real layers are used
        let layer = &mut mamba.layers.real_layers[i];
        let name = |n: &str| format!("backbone.layers.{i}.{n}");
        load_param_f16_to_f32(
            &mut layer.norm.gamma,
            name("norm.weight"),
            tensors,
            device,
            false,
        )?;
        let name = |n: &str| format!("backbone.layers.{i}.mixer.{n}");
        let mamba_block = &mut layer.mamba_block;
        load_param_f16_to_f32(
            &mut mamba_block.norm.gamma,
            name("norm.weight"),
            tensors,
            device,
            false,
        )?;
        load_param_f16_to_f32(
            &mut mamba_block.a_log_h,
            name("A_log"),
            tensors,
            device,
            false,
        )?;
        load_param_f32_to_f32(&mut mamba_block.d_h, name("D"), tensors, device, false)?;
        load_param_f16_to_f32(
            &mut mamba_block.conv1d.weight,
            name("conv1d.weight"),
            tensors,
            device,
            false,
        )?;
        load_param_f16_to_f32(
            &mut mamba_block.conv1d.bias.as_mut().unwrap(),
            name("conv1d.bias"),
            tensors,
            device,
            false,
        )?;
        load_param_f16_to_f32(
            &mut mamba_block.dt_bias_h,
            name("dt_bias"),
            tensors,
            device,
            false,
        )?;
        load_param_f16_to_f32(
            &mut mamba_block.in_proj.weight,
            name("in_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param_f16_to_f32(
            &mut mamba_block.out_proj.weight,
            name("out_proj.weight"),
            tensors,
            device,
            true,
        )?;
    }

    let param = mamba.embedding.weight.val();
    let param = param.swap_dims(0, 1);
    // ensure the tensor is contiguous
    let param: Tensor<2> = Tensor::from_data(param.into_data(), device);

    mamba.lm_head = Some(burn::nn::Linear {
        weight: Param::from_tensor(param),
        bias: None,
    });

    Ok(outer_mamba)
}

#[allow(dead_code)]
pub fn load_param_f16_to_f32<const D: usize>(
    param: &mut Param<Tensor<D>>,
    name: String,
    tensors: &SafeTensors,
    device: &Device,
    swap_dims: bool,
) -> anyhow::Result<()> {
    let data = tensors.tensor(&name)?.data();

    // converts u8 data to f16 (used as f32)
    let mut data_f32 = Vec::with_capacity(data.len() / 8);
    let mut buf = [0; 2];
    for chunk in data.chunks(2) {
        buf.copy_from_slice(chunk);
        let data_u = u16::from_le_bytes(buf);
        data_f32.push(f32::from(half::f16::from_bits(data_u)));
    }

    let shape = param.dims();
    let tensor: Tensor<1> = Tensor::from_data(data_f32.as_slice(), device);
    let tensor = if swap_dims {
        // transpose some linear layers

        let mut temp_shape = shape.clone();
        temp_shape[0] = shape[1];
        temp_shape[1] = shape[0];
        let tensor = tensor.reshape(temp_shape).swap_dims(0, 1);
        // ensure the tensor is contiguous
        let tensor: Tensor<D> = Tensor::from_data(tensor.into_data(), device);

        let tensor = tensor.reshape(shape);
        tensor
    } else {
        tensor.reshape(shape)
    };
    *param = Param::from_tensor(tensor);

    Ok(())
}

#[allow(dead_code)]
pub fn load_param_f32_to_f32<const D: usize>(
    param: &mut Param<Tensor<D>>,
    name: String,
    tensors: &SafeTensors,
    device: &Device,
    swap_dims: bool,
) -> anyhow::Result<()> {
    let data = tensors.tensor(&name)?.data();

    // converts u8 data to f32
    let mut data_f32 = Vec::with_capacity(data.len() / (8 / 2));
    let mut buf = [0; 4];
    for chunk in data.chunks(4) {
        buf.copy_from_slice(chunk);
        let data_u = u32::from_le_bytes(buf);
        data_f32.push(f32::from_bits(data_u));
    }

    let shape = param.dims();
    let tensor: Tensor<1> = Tensor::from_data(data_f32.as_slice(), device);
    let tensor = if swap_dims {
        // transpose some linear layers
        let mut temp_shape = shape.clone();
        temp_shape[0] = shape[1];
        temp_shape[1] = shape[0];
        let tensor = tensor.reshape(temp_shape).swap_dims(0, 1);
        // ensure the tensor is contiguous
        let tensor: Tensor<D> = Tensor::from_data(tensor.into_data(), device);
        tensor
    } else {
        tensor.reshape(shape)
    };
    *param = Param::from_tensor(tensor);

    Ok(())
}
