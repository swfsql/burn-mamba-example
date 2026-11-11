//! The native binary. The module `burn_mamba_example::native` holds its code.

/// Runs [burn_mamba_example::native::main].
pub fn main() -> anyhow::Result<()> {
    burn_mamba_example::native::main()
}
