from tests import ddp_jax_multi


def run_ddp_step(
    batch_size=64,
    input_size=4096,
    output_size=4096,
    warmup=3,
    iters=10,
    dtype_name="bfloat16",
):
    # Same model as ddp_jax_multi: N processes x 1 GPU each, pmap + cross-process
    # pmean. On a single node this gives 8 processes x 1 GPU via XGMI; on
    # multiple nodes it extends naturally to cross-node via Slingshot.
    return ddp_jax_multi.run_ddp_step(
        batch_size=batch_size,
        input_size=input_size,
        output_size=output_size,
        warmup=warmup,
        iters=iters,
        dtype_name=dtype_name,
    )
