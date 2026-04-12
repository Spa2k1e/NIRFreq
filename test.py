import argparse
import os
import torch
import pytorch_lightning as pl
import torch._dynamo

torch._dynamo.disable()

from pytorch_lightning import seed_everything

from src.data.dataset import plNBUDataset
from src.models.nir_freq_model import NIRFreqModel
from src.models.gsa_model import GSAModel
from src.models.sfim_model import SFIMModel
from src.models.wavelet_model import WaveletModel
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


def main(args):
    print(f"Starting simplified test with seed {args.seed}")
    seed_everything(args.seed)
    check_and_make(args.output_dir)

    dataset = plNBUDataset(
        data_dir=args.data_dir,
        sensor=args.sensor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        test_mode=args.test_mode,
    )

    if args.model_name == 'nirfreq':
        if not args.ckpt:
            raise ValueError("Checkpoint path '--ckpt' is required for 'nirfreq' model.")
        print(f"Loading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu')
        hparams = checkpoint.get('hyper_parameters', {})

        sensor_defaults = {
            'gf': {'ms_chans': 4, 'rgb_c': '2,1,0'},
            'wv2': {'ms_chans': 8, 'rgb_c': '4,2,1'},
            'ikonos': {'ms_chans': 4, 'rgb_c': '2,1,0'},
            'quickbird': {'ms_chans': 4, 'rgb_c': '2,1,0'},
            'wv3': {'ms_chans': 8, 'rgb_c': '4,2,1'},
        }
        sensor_key = args.sensor.lower()
        defaults = sensor_defaults.get(sensor_key, {'ms_chans': 4, 'rgb_c': '2,1,0'})

        model_params = {
            'lr': hparams.get('lr', 0),
            'epochs': hparams.get('epochs', 0),
            'ms_chans': hparams.get('ms_chans', defaults['ms_chans']),
            'rgb_c': hparams.get('rgb_c', defaults['rgb_c']),
            'sensor': args.sensor,
            'embed_dim': hparams.get('embed_dim', 64),
            'num_layers': hparams.get('num_layers', 9),
            'test_mode': args.test_mode,
            'output_dir': args.output_dir,
            'enable_GCFM': hparams.get('enable_GCFM', True),
            'enable_HFA': hparams.get('enable_HFA', True),
            'enable_corr_map': hparams.get('enable_corr_map', True),
            'fixed_alpha': args.fixed_alpha
        }

        model = NIRFreqModel(**model_params)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("NIRFreqModel loaded successfully from checkpoint.")

    elif args.model_name == 'pannet':
        print("Initializing PanNet model.")
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = PanNetModel(lr=0, epochs=0, ms_chans=hparams.get('ms_chans', 4), sensor=args.sensor,
                                test_mode=args.test_mode, output_dir=args.output_dir)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = PanNetModel(lr=0, epochs=0, ms_chans=4, sensor=args.sensor, test_mode=args.test_mode,
                                output_dir=args.output_dir)

    elif args.model_name == 'gppnn':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = GPPNNModel(lr=0, epochs=0, ms_chans=hparams.get('ms_chans', 4), sensor=args.sensor,
                               test_mode=args.test_mode, output_dir=args.output_dir)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = GPPNNModel(lr=0, epochs=0, ms_chans=4, sensor=args.sensor, test_mode=args.test_mode,
                               output_dir=args.output_dir)

    elif args.model_name == 'pnn':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = PNNModel(lr=0, epochs=0, ms_chans=hparams.get('ms_chans', 4), sensor=args.sensor,
                             test_mode=args.test_mode, output_dir=args.output_dir)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = PNNModel(lr=0, epochs=0, ms_chans=4, sensor=args.sensor, test_mode=args.test_mode,
                             output_dir=args.output_dir)

    elif args.model_name == 'lgteun':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = LGTEUNModel(epochs=0, ms_chans=hparams.get('ms_chans', 8), sensor=args.sensor,
                                test_mode=args.test_mode, output_dir=args.output_dir)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = LGTEUNModel(epochs=0, ms_chans=8, sensor=args.sensor, test_mode=args.test_mode,
                                output_dir=args.output_dir)

    elif args.model_name == 'fame-net':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = FAMENetModel(
                ms_chans=hparams.get('ms_chans', 4),
                base_filter=hparams.get('base_filter', 32),
                sensor=args.sensor,
                test_mode=args.test_mode,
                output_dir=args.output_dir
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = FAMENetModel(ms_chans=4, base_filter=32, sensor=args.sensor, test_mode=args.test_mode,
                                 output_dir=args.output_dir)

    elif args.model_name == 'p2sharpen':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = P2SharpenModel(
                ms_chans=hparams.get('ms_chans', 4),
                ratio=hparams.get('ratio', 4),
                sensor=args.sensor,
                test_mode=args.test_mode,
                output_dir=args.output_dir
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = P2SharpenModel(ms_chans=4, ratio=4, sensor=args.sensor, test_mode=args.test_mode,
                                   output_dir=args.output_dir)

    elif args.model_name == 'dpfn':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hyper_parameters = checkpoint.get('hyper_parameters', {})
            hyper_parameters['test_mode'] = args.test_mode
            hyper_parameters['output_dir'] = args.output_dir
            hyper_parameters['sensor'] = args.sensor
            model = DPFNModel(**hyper_parameters)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = DPFNModel(ms_chans=4, sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    elif args.model_name == 'faformer':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = FAFormerModel(epochs=0, ms_chans=hparams.get('ms_chans', 4), sensor=args.sensor,
                                  test_mode=args.test_mode, output_dir=args.output_dir)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = FAFormerModel(epochs=0, ms_chans=4, sensor=args.sensor, test_mode=args.test_mode,
                                  output_dir=args.output_dir)

    elif args.model_name == 'premix':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = PreMixModel(
                ms_chans=hparams.get('ms_chans', 4),
                embed_dim=hparams.get('embed_dim', 32),
                sensor=args.sensor,
                test_mode=args.test_mode,
                output_dir=args.output_dir
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = PreMixModel(ms_chans=4, sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    elif args.model_name == 's3fnet':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = S3FNetModel(
                ms_chans=hparams.get('ms_chans', 4),
                embed_dim=hparams.get('embed_dim', 32),
                sensor=args.sensor,
                test_mode=args.test_mode,
                output_dir=args.output_dir
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = S3FNetModel(ms_chans=4, sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    elif args.model_name == 'ssamrn':
        if args.ckpt:
            checkpoint = torch.load(args.ckpt, map_location='cpu')
            hparams = checkpoint.get('hyper_parameters', {})
            model = SSAMRNModel(
                ms_chans=hparams.get('ms_chans', 4),
                sensor=args.sensor,
                test_mode=args.test_mode,
                output_dir=args.output_dir
            )
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model = SSAMRNModel(ms_chans=4, sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    elif args.model_name == 'gsa':
        model = GSAModel(sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    elif args.model_name == 'sfim':
        model = SFIMModel(sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    elif args.model_name == 'wavelet':
        model = WaveletModel(sensor=args.sensor, test_mode=args.test_mode, output_dir=args.output_dir)

    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    trainer = pl.Trainer(
        accelerator="auto",
        devices=[args.device],
        logger=False,
        callbacks=[],
    )

    print(f"Running test for '{args.model_name}' in '{args.test_mode}' mode...")
    trainer.test(model, datamodule=dataset)
    print(f"Test complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Comparison Testing')

    parser.add_argument('--model_name', type=str, default='nirfreq',
                        choices=['nirfreq', 'gsa', 'sfim', 'wavelet', 'pannet', 'gppnn', 'pnn', 'lgteun',
                                 'fame-net', 'p2sharpen', 'dpfn', 'faformer', 'premix', 's3fnet', 'ssamrn'])
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the dataset.")
    parser.add_argument('--sensor', type=str, required=True, help="Sensor type (e.g., 'wv2', 'gf', 'ikonos').")
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test_mode', type=str, default='full', choices=['reduced', 'full'], help="Test mode.")
    parser.add_argument('--output_dir', type=str, default='test_results', help="Directory for results.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for testing.")
    parser.add_argument('--device', type=int, default=0, help="GPU device ID.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")

    # 新增 alpha 消融控制参数
    parser.add_argument('--fixed_alpha', type=float, default=None,
                        help="Fix alpha to a specific value for ablation testing.")

    args = parser.parse_args()
    main(args)