import sys
import argparse

import lightning.pytorch.loggers

from pathlib import Path

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

src_dir = Path.cwd().parent
sys.path.append(str(src_dir))
from model import RNAProteinInterAct, RNAProteinInterActSE, ModelWrapper, BaseCNN, SimpleRPI_FFN
from dataloader import get_dataloader
import numpy as np

def main(args):

    # Dim lookup — add this at the top of main()
    PROTEIN_DIMS = {"esm": 640, "esmc": 1152, "lcplm": 1536, "vesm": 640}
    RNA_DIMS     = {"lamar": 768, "rna_fm": 640, "rnaelectra": 512}

    # Append FM subfolder to the base paths
    protein_embeddings_path = Path(args.protein_embeddings_path) / args.protein_fm
    rna_embeddings_path     = Path(args.rna_embeddings_path)     / args.rna_fm
    
    # Choose model based on embedding strategy
    if args.baseline:
        rpi_model = BaseCNN(
            device="cuda" if args.accelerator == "gpu" else "cpu",
        )
    else:
        if args.one_hot_encoding:
            model = RNAProteinInterActSE
        else:
            # Comented to try the version of training avoiding the transformers
            model = RNAProteinInterAct
            # model = SimpleRPI_FFN  
        
        protein_embed_dim = PROTEIN_DIMS[args.protein_fm]
        rna_embed_dim     = RNA_DIMS[args.rna_fm]

        # Initialize model
        # rpi_model = model() 
        rpi_model = model( 
            batch_first=True,
            protein_embed_dim=protein_embed_dim,
            rna_embed_dim=rna_embed_dim,
            d_model=args.d_model,
            num_encoder_layers=args.num_encoder_layers,
            nhead=args.n_head,
            dim_feedforward=args.dim_feedforward,
            key_padding_mask=args.key_padding_mask,
            norm_first=True,
            dropout=args.dropout
        )
    
    # Wrap model in a LightningModule
    lightning_module = ModelWrapper(
        rpi_model,
        lr_init=args.lr_init,
        weight_decay=args.weight_decay,
        t_max=args.max_epochs,
        warmup_steps = args.warmup_steps,
        seed=args.seed,
        # cpr=args.cpr
    )
    
    full_exp_name = f"{args.wandb_run_name}, lr: {args.lr_init}, wd: {args.weight_decay}, dr: {args.dropout}, seed: {args.seed}"
    
    # Initialize logger
    if args.wandb:
        logger = lightning.pytorch.loggers.WandbLogger(
            project="thesis_writing_reproducibility",
            name=full_exp_name,
            group=args.wandb_group
        )
    else:
        logger = True

    # Original Checkpoint Callback, disabled only to try early stopping for the one below
    # Initialize checkpoint callback to save best models        
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.checkpoints_dir,  # Custom path for saving checkpoints
    #     filename='rpi-{epoch:02d}-{val_BinaryAUROC:.3f}-{valid_loss_epoch:.3f}-' + full_exp_name,  # Filename format
    #     every_n_epochs=10,
    #     save_top_k=-1,
    #     monitor='val_BinaryAUROC',
    #     mode='max',  # Mode for the monitored metric, 'min' for minimization
    #     save_last=True,  # Save the last checkpoint in addition to the best ones
    # )

    # 2. Update Checkpoint Callback
    # It's better to monitor 'valid_loss_epoch' or 'val_BinaryAUROC' rather than train_loss
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=args.checkpoints_dir,
    #     filename='best-{epoch:02d}-{valid_loss_epoch:.3f}',
    #     monitor='valid_loss_epoch',
    #     mode='min',
    #     save_top_k=1, 
    #     save_last=True,
    # )





    best_checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename='best-{epoch:02d}-{valid_loss_epoch:.3f}',
        monitor='valid_loss_epoch',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    all_checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        filename='epoch-{epoch:02d}-{valid_loss_epoch:.3f}',
        save_top_k=-1,
        every_n_epochs=1,
        save_on_train_epoch_end=False
    )






    best_checkpoint_callback.CHECKPOINT_NAME_LAST = ("last-" + args.wandb_group + "_" + full_exp_name)


    # checkpoint_callback.CHECKPOINT_NAME_LAST = "last-" + args.wandb_group + "_" + full_exp_name

    lr_monitor = LearningRateMonitor(logging_interval='step')


    # 1. Initialize Early Stopping Callback
    # We monitor 'valid_loss_epoch' which you already log in ModelWrapper
    early_stop_callback = EarlyStopping(
        monitor="valid_loss_epoch",
        patience=100,             # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode="min"
    )


    # Initialize trainer 
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        logger=logger,
        log_every_n_steps=1,
        # callbacks=[lr_monitor, checkpoint_callback], 
        # callbacks=[lr_monitor, checkpoint_callback, early_stop_callback], 
        callbacks=[lr_monitor, best_checkpoint_callback, all_checkpoint_callback, early_stop_callback], 
    )
    
    # Get train dataloader (val is redundant outside of HPO experiments)
    train_dataloader = get_dataloader(  
        loader_type=args.loader_type,
        dataset_path=args.train_set_path,
        rna_embeddings_path=rna_embeddings_path,
        protein_embeddings_path=protein_embeddings_path,
        use_individual_files=args.use_individual_files,  # ← NUEVO
        seed=args.seed,
        num_workers=args.num_dataloader_workers,
        batch_size=args.batch_size,
    )

    # 5. Create Validation Dataloader
    # Note: Use a separate path for the validation set (args.val_set_path)
    val_dataloader = get_dataloader(
        loader_type=args.loader_type,
        dataset_path=args.val_set_path, # Add this to your args!
        rna_embeddings_path=rna_embeddings_path,
        protein_embeddings_path=protein_embeddings_path,
        use_individual_files=args.use_individual_files,  # ← NUEVO
        seed=args.seed,
        num_workers=args.num_dataloader_workers,
        batch_size=args.batch_size,
        shuffle=False, # Don't shuffle validation
    )


    # Train model
    trainer.fit(model=lightning_module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Command line options for the script")

    parser.add_argument("--accelerator", default='cuda', help="Type of accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--wandb", action='store_true', default=False, help="Enables logging via wandb")
    parser.add_argument("--wandb_run_name", default="default", help="Name of the wandb run")
    parser.add_argument("--wandb_group", default="default", help="Name of the wandb group")

    parser.add_argument("--baseline", action='store_true', default=False, help="Runs baseline CNN model")
    parser.add_argument("--one_hot_encoding", action='store_true', default=False, help="Enables one-hot encoding")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="Number of encoder layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--d_model", type=int, default=256, help="Dimension of model")
    parser.add_argument("--n_head", type=int, default=2, help="Number of heads")
    parser.add_argument("--dim_feedforward", type=int, default=20, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--key_padding_mask", action='store_true', default=False, help="Enables key padding mask")
    parser.add_argument("--lr_init", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--loader_type", default="RPIDataset", help="Type of dataloader")
    
    parser.add_argument("--cpr", action='store_true', default=False, help="Sets AdamCPR as optimizer") 
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--max_epochs", type=int, default=1, required=True, help="Maximum number of epochs")
    
    parser.add_argument("--use_individual_files", action='store_true', default=False, help="Load embeddings from individual files instead of concatenated array")

    parser.add_argument("--num_dataloader_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--protein_embeddings_path", default="data/embeddings/eCLIP_2/RNAInterAct", help="Path to protein embeddings")
    parser.add_argument("--rna_embeddings_path", default="data/embeddings/eCLIP_2/RNAInterAct", help="Path to RNA embeddings")
    parser.add_argument("--train_set_path", default="data/interactions/eCLIP2/train_set_withValSet.parquet", help="Path to the train set file")
    # New extra argument for validation set path
    parser.add_argument("--val_set_path", default="data/interactions/eCLIP2/validation_set_withValSet.parquet", help="Path to the val set file")
    
    parser.add_argument("--seed", type=int, default=6844, help="Seed for reproducibility")
    parser.add_argument("--checkpoints_dir", default="checkpoints", help="Path to the checkpoints")

    parser.add_argument("--protein_fm", default="esm", choices=["esm", "esmc", "lcplm", "vesm"], help="Protein FM name")
    parser.add_argument("--rna_fm", default="rna_fm", choices=["lamar", "rna_fm", "rnaelectra"], help="RNA FM name")

    args = parser.parse_args()

    main(args)