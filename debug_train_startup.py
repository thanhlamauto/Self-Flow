import sys


def mark(message):
    print(message, flush=True)


def step(label, fn):
    mark(f"[debug] importing {label} ...")
    fn()
    mark(f"[debug] imported {label}")


def main():
    mark(f"[debug] python={sys.version.split()[0]}")

    step("jax", lambda: __import__("jax"))
    import jax

    mark(f"[debug] jax.device_count() -> {jax.device_count()}")

    step("jax.numpy", lambda: __import__("jax.numpy"))
    step("optax", lambda: __import__("optax"))
    step("wandb", lambda: __import__("wandb"))
    step("flax.training", lambda: __import__("flax.training"))
    step("flax", lambda: __import__("flax"))
    step("numpy", lambda: __import__("numpy"))
    step("grain.python", lambda: __import__("grain.python"))
    step("src.model", lambda: __import__("src.model"))
    step("src.sampling", lambda: __import__("src.sampling"))
    step("src.utils", lambda: __import__("src.utils"))

    mark("[debug] all imports completed")


if __name__ == "__main__":
    main()
