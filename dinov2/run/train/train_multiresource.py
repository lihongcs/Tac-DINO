"""Distributed GeoTAC pretraining entry point."""
from dinov2.train import train as trainer
from dinov2.train.multiresource import MultiResourceSSLMetaArch, architecture_contract
from dinov2.data.multiresource import MultiResourceAugmentation, collate_multiresource
from collections import defaultdict
from dinov2.logging import MetricLogger
from dinov2.logging.helpers import SmoothedValue
from dinov2.tacdino.file_records import digest
from dinov2.logging import setup_logging

class RawStepLogger(MetricLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meters = defaultdict(lambda: SmoothedValue(window_size=1))

    def log_every(self, iterable, print_freq, header=None, n_iterations=None, start_iteration=0):
        return super().log_every(iterable, 1, header, n_iterations, start_iteration)


_original = trainer._checkpoint_provenance
_export = trainer.do_test


def export_checkpoint(cfg, model, iteration, checkpoint_iteration=None):
    # Initialization is already reproducible from the two official weight
    # hashes; do not write two multi-GB untrained duplicate exports.
    if iteration == 'training_-1':
        return
    return _export(cfg, model, iteration, checkpoint_iteration)


def provenance(cfg, *args, **kwargs):
    value = _original(cfg, *args, **kwargs)
    architecture = architecture_contract(cfg)
    value.update(architecture=architecture, architecture_sha256=digest(architecture))
    return value


def main():
    trainer.SSLMetaArch = MultiResourceSSLMetaArch
    trainer.TacDataAugmentationDINO = MultiResourceAugmentation
    trainer.collate_data_and_cast = collate_multiresource
    trainer._checkpoint_provenance = provenance
    trainer.do_test = export_checkpoint
    trainer.MetricLogger = RawStepLogger
    setup_logging()
    args = trainer.get_args_parser(add_help=True).parse_args()
    from dinov2.train.ssl_meta_arch import fmha
    if fmha is None:
        raise RuntimeError("GeoTAC CUDA pretraining requires xFormers built for the installed PyTorch")
    trainer.main(args)


if __name__ == "__main__":
    main()
