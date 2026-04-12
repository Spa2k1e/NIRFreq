import argparse
import yaml
import pytorch_lightning as pl
import os
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.dataset import plNBUDataset
from src.models.nir_freq_model import NIRFreqModel
from src.models.pannet_model import PanNetModel
from src.models.gppnn_model import GPPNNModel
from src.models.pnn_model import PNNModel
from src.models.lgteun_model import LGTEUNModel
from src.models.fame_net_model import FAMENetModel
from src.models.p2sharpen_model import P2SharpenModel
from src.models.dpfn_model import DPFNModel
from src.models.faformer_model import FAFormerModel
from src.models.premix_model import PreMixModel
from src.models.s3fnet_model import S3FNetModel
from src.models.ssamrn_model import SSAMRNModel
from src.utils.common import check_and_make


def get_args_parser():
    parser = argparse.ArgumentParser('Training', add_help=False)

    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to the YAML configuration file.')

    parser.add_argument('--model_name', type=str, default='nirfreq',
                        choices=['nirfreq', 'pannet', 'gppnn', 'pnn', 'lgteun', 'fame-net', 'p2sharpen', 'dpfn',
                                 'faformer', 'premix', 's3fnet', 'ssamrn'], help='Name of the model to train.')

    # 基础参数
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=400, help="Total number of training epochs.")
    parser.add_argument('--ms_chans', type=int, default=4, help="Number of multispectral channels.")
    parser.add_argument('--rgb_c', type=str, default='2,1,0', help="Comma-separated string of RGB channel indices.")
    parser.add_argument('--data_dir', type=str, help="Directory containing the dataset.")
    parser.add_argument('--sensor', type=str, default='gf', help="Sensor type (e.g., 'wv2', 'gf', 'ikonos').")
    parser.add_argument('--test_freq', type=int, default=10, help="Frequency of validation (epochs).")
    parser.add_argument('--device', type=int, default=0, help="GPU device ID to use.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help="Learning rate.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for the dataloader.")
    parser.add_argument('--pin_mem', action=argparse.BooleanOptionalAction, default=True,
                        help="Pin memory for faster data transfer.")

    # 模型特有参数
    parser.add_argument('--num_layers', type=int, default=9, help="Number of layers for NIRFreqModel.")
    parser.add_argument('--embed_dim', type=int, default=32, help="Embedding dimension.")
    parser.add_argument('--enable_GCFM', action='store_true', dest='enable_GCFM')
    parser.add_argument('--disable_GCFM', action='store_false', dest='enable_GCFM')
    parser.add_argument('--enable_HFA', action='store_true', dest='enable_HFA')
    parser.add_argument('--disable_HFA', action='store_false', dest='enable_HFA')
    parser.add_argument('--enable_corr_map', action='store_true', dest='enable_corr_map', default=True)
    parser.add_argument('--disable_corr_map', action='store_false', dest='enable_corr_map')

    # 新增 alpha 消融控制参数
    parser.add_argument('--fixed_alpha', type=float, default=None,
                        help="Fix alpha to a specific value for ablation studies (e.g., 0.0, 0.5, 1.0). If None, alpha is learnable.")

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Arguments:\n" + "\n".join(f"  {k}: {v}" for k, v in vars(args).items()))

    if args.model_name == 'nirfreq':
        alpha_str = f"_alpha={args.fixed_alpha}" if args.fixed_alpha is not None else "_alpha=learnable"
        model_name_str = (
            f"NIRFreqModel_s={args.sensor}_l={args.num_layers}_d={args.embed_dim}"
            f"_GCFM={args.enable_GCFM}_HFA={args.enable_HFA}_CorrMap={args.enable_corr_map}{alpha_str}"
        )
    else:
        model_name_str = f"{args.model_name.upper()}_s={args.sensor}"

    output_dir = f"log_m={model_name_str}"
    check_and_make(output_dir)
    seed_everything(args.seed)

    dataset = plNBUDataset(args.data_dir,
                           args.sensor,
                           args.batch_size,
                           args.num_workers,
                           args.pin_mem,
                           )

    # 实例化模型
    if args.model_name == 'nirfreq':
        model = NIRFreqModel(**vars(args))
    elif args.model_name == 'pannet':
        model = PanNetModel(**vars(args))
    elif args.model_name == 'gppnn':
        model = GPPNNModel(**vars(args))
    elif args.model_name == 'pnn':
        model = PNNModel(**vars(args))
    elif args.model_name == 'lgteun':
        model = LGTEUNModel(**vars(args))
    elif args.model_name == 'fame-net':
        model = FAMENetModel(**vars(args))
    elif args.model_name == 'p2sharpen':
        model = P2SharpenModel(**vars(args))
    elif args.model_name == 'dpfn':
        model = DPFNModel(**vars(args))
    elif args.model_name == 'faformer':
        model = FAFormerModel(**vars(args))
    elif args.model_name == 'premix':
        model = PreMixModel(**vars(args))
    elif args.model_name == 's3fnet':
        model = S3FNetModel(**vars(args))
    elif args.model_name == 'ssamrn':
        model = SSAMRNModel(**vars(args))
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    csv_logger = CSVLogger(
        save_dir=output_dir,
        name="metrics",
        version=None,
        flush_logs_every_n_steps=10
    )

    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       monitor='val/PSNR_mean',
                                       mode="max",
                                       save_top_k=3,
                                       auto_insert_metric_name=False,
                                       filename=f'{args.model_name}_ep={{epoch}}_PSNR={{val/PSNR_mean:.4f}}',
                                       every_n_epochs=args.test_freq
                                       )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         accelerator="auto",
                         devices=[args.device],
                         logger=csv_logger,
                         check_val_every_n_epoch=args.test_freq,
                         callbacks=[model_checkpoint],
                         log_every_n_steps=10,
                         )

    trainer.fit(model, datamodule=dataset)


if __name__ == '__main__':
    parser = get_args_parser()
    args, unknown = parser.parse_known_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config = {k: v for k, v in config.items() if v is not None}
        parser.set_defaults(**config)

    args = parser.parse_args()

    if not args.data_dir:
        parser.error("--data_dir is required.")

    main(args)