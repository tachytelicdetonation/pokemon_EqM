import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate Pokemon samples")
    
    # General
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--model", type=str, default="EqM-XL/2", help="Model architecture")
    parser.add_argument("--image-size", type=int, default=256, help="Image size")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="samples", help="Output directory")
    
    # VAE
    parser.add_argument("--vae", type=str, default="ema", help="VAE model name")
    parser.add_argument("--vae-path", type=str, default=None, help="Path to VAE checkpoint")
    parser.add_argument("--vae-channels", type=int, default=4, help="VAE latent channels")
    
    # Sampling
    parser.add_argument("--sampler", type=str, default="gd", choices=["ode_dopri5", "ode_euler", "ode_heun", "gd", "ngd"], help="Sampling method")
    parser.add_argument("--num-sampling-steps", type=int, default=250, help="Number of sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="Classifier-free guidance scale")
    parser.add_argument("--sample-eps", type=float, default=0.0, help="Sampling epsilon")
    parser.add_argument("--mu", type=float, default=0.3, help="Momentum for NGD sampler")
    parser.add_argument("--stepsize", type=float, default=0.0017, help="Step size for GD/NGD sampler")
    
    # Transport
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    parser.add_argument("--loss-weight", type=str, default=None, choices=[None, "velocity", "likelihood"])
    parser.add_argument("--train-eps", type=float, default=0.0)
    
    # Model specific
    parser.add_argument("--uncond", type=bool, default=True, help="Unconditional model")
    parser.add_argument("--energy-head", type=str, default="implicit", help="Energy head type")
    parser.add_argument("--use-rope", action="store_true", help="Use RoPE")
    parser.add_argument("--rope-base", type=float, default=10000.0, help="RoPE base")
    parser.add_argument("--use-liere", action="store_true", help="Use LieRE")
    
    # Input
    parser.add_argument("--noised-input-path", type=str, default=None, help="Path to noised input")
    
    args = parser.parse_args()
    return args
