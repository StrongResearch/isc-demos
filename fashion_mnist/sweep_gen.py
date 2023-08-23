import toml

def gen_config(lr, batch_size):
    config = dict(
        experiment_name = "fashion_mnist",
        gpu_type = "24GB VRAM GPU",
        nnodes = 10,
        venv_path = "~/.venv/bin/activate",
        output_path = "/mnt/Client/strongcompute_chris/fashion_mnist/output_fashion_mnist",
        command = f"isc_optimization_tutorial.py --lr={lr} --batch-size {batch_size}"
    )
    return toml.dumps(config)

def main():
    for lr in [0.1, 0.01, 0.001, 0.0001]:
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            fname = f"sweep_configs/fashion_mnist_lr_{lr}_batch_size_{batch_size}.isc"
            with open(fname, 'w') as f:
                f.write(gen_config(lr, batch_size))
            print(f"isc train {fname}")

main()
