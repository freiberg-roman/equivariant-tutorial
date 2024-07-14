# equivariant-tutorial

## Run the code through podman or docker

Build the torch and jax version through
```bash
podman/docker build --target simple-cnn-torch -t eq:simple-torch .
podman/docker build --target simple-cnn-jax -t eq:simple-jax .
podman/docker build --target simple-eq-torch -t eq:eq-torch .
podman/docker build --target simple-eq-jax -t eq:eq-jax .
```
Train and Evaluate using
```bash
podman/docker run --rm -t eq:simple-torch
podman/docker run --rm -t eq:simple-jax
podman/docker run --rm -t eq:eq-torch
podman/docker run --rm -t eq:eq-jax
```
