use burn::module::Param;
use burn::prelude::*;
use safetensors::SafeTensors;

pub fn safetensors_load<B: Backend>(
    mamba_safetensors_bytes: &[u8],
    mamba_config: burn_mamba::MambaConfig,
    device: &B::Device,
) -> anyhow::Result<burn_mamba::Mamba<B>> {
    let mut mamba = mamba_config.init::<B>(&device);
    let tensors = &safetensors::SafeTensors::deserialize(&mamba_safetensors_bytes)?;
    // info!("{:?}", tensors.names());
    //

    let name = |n: &str| format!("backbone.{n}");
    load_param(
        &mut mamba.embedding.weight,
        name("embedding.weight"),
        tensors,
        device,
        false,
    )?;
    load_param(
        &mut mamba.norm_f.gamma,
        name("norm_f.weight"),
        tensors,
        device,
        false,
    )?;

    for i in 0..24 {
        let layer = &mut mamba.layers[i];
        let name = |n: &str| format!("backbone.layers.{i}.{n}");
        load_param(
            &mut layer.norm.gamma,
            name("norm.weight"),
            tensors,
            device,
            false,
        )?;
        let name = |n: &str| format!("backbone.layers.{i}.mixer.{n}");
        let mamba_block = &mut layer.mamba_block;
        load_param(
            &mut mamba_block.a_log,
            name("A_log"),
            tensors,
            device,
            false,
        )?;
        load_param(&mut mamba_block.d, name("D"), tensors, device, false)?;
        load_param(
            &mut mamba_block.conv1d.weight,
            name("conv1d.weight"),
            tensors,
            device,
            false,
        )?;
        load_param(
            &mut mamba_block.conv1d.bias.as_mut().unwrap(),
            name("conv1d.bias"),
            tensors,
            device,
            false,
        )?;
        load_param(
            &mut mamba_block.dt_proj.weight,
            name("dt_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param(
            &mut mamba_block.dt_proj.bias.as_mut().unwrap(),
            name("dt_proj.bias"),
            tensors,
            device,
            false,
        )?;
        load_param(
            &mut mamba_block.in_proj.weight,
            name("in_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param(
            &mut mamba_block.out_proj.weight,
            name("out_proj.weight"),
            tensors,
            device,
            true,
        )?;
        load_param(
            &mut mamba_block.x_proj.weight,
            name("x_proj.weight"),
            tensors,
            device,
            true,
        )?;
    }

    let param = mamba.embedding.weight.val();
    let param = param.movedim(1, 0);
    mamba.lm_head = Some(burn::nn::Linear {
        weight: Param::from_tensor(param),
        bias: None,
    });

    Ok(mamba)
}

pub fn load_param<B: Backend, const D: usize>(
    param: &mut Param<Tensor<B, D>>,
    name: String,
    tensors: &SafeTensors,
    device: &B::Device,
    movedim: bool,
) -> anyhow::Result<()> {
    let data = tensors.tensor(&name)?.data();

    // converts u8 data to f32
    let mut data_f32 = Vec::with_capacity(data.len() / 4);
    let mut buf = [0; 4];
    for chunk in data.chunks(4) {
        buf.copy_from_slice(chunk);
        let data_u = u32::from_le_bytes(buf);
        data_f32.push(f32::from_bits(data_u));
    }

    let shape = param.dims();
    let tensor: Tensor<B, 1> = Tensor::from_data(data_f32.as_slice(), device);
    let tensor = if movedim {
        // transpose some linear layers
        let mut temp_shape = shape.clone();
        temp_shape[0] = shape[1];
        temp_shape[1] = shape[0];
        tensor.reshape(temp_shape).movedim(0, 1)
    } else {
        tensor.reshape(shape)
    };
    *param = Param::from_tensor(tensor);

    Ok(())
}
