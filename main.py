import argparse
from omegaconf import OmegaConf
from core.trainer import Trainer
from core.inferencer import Inferencer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'])
    
    # Inference specific
    parser.add_argument('--cond_file', type=str, help='Path to condition pth')
    parser.add_argument('--ckpt_path', type=str, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='results')
    
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    
    if args.mode == 'train':
        Trainer(cfg).run()
    
    elif args.mode == 'inference':
        if args.ckpt_path:
            cfg.inference.ckpt_path = args.ckpt_path
        if not args.cond_file:
            raise ValueError("Inference requires --cond_file")
            
        Inferencer(cfg).run(args.cond_file, args.output_dir)

if __name__ == "__main__":
    main()