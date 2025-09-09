import argparse
import os
import torch

from lightning import Trainer

from model import ModelWrapper
from dataloader import get_dataloader

###########
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report
import pandas as pd

def main(args):

    test_results_dir = os.path.join(os.getcwd(), 'test_results')
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    
    model = ModelWrapper.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        map_location=torch.device('cpu') if args.device == 'cpu' else torch.device('cuda'),
        cpr=False,
    )
    
    model.eval()
    
    test_dataloader = get_dataloader(
        loader_type=args.loader_type,
        dataset_path=args.test_set_path,
        rna_embeddings_path=args.rna_embeddings_path,
        protein_embeddings_path=args.protein_embeddings_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    trainer = Trainer(accelerator=args.device)
    trainer.test(model, dataloaders=test_dataloader)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.eval()

    # y_true_list, y_pred_list = [], []

    # with torch.no_grad():
    #     for rna_embed, protein_embed, y, _ in test_dataloader:
    #         rna_embed = rna_embed.to(device)
    #         protein_embed = protein_embed.to(device)
    #         y = y.to(device)
    #         probs = model(rna_embed, protein_embed)   # probs shape (B,1) or (B,)
    #         probs = probs.view(-1).cpu().numpy()
    #         y = y.view(-1).cpu().numpy()
    #         y_true_list.append(y)
    #         y_pred_list.append(probs)

    # y_true = np.concatenate(y_true_list)
    # y_pred = np.concatenate(y_pred_list)

#     print("shape", y_true.shape, y_pred.shape)
#     print("sklearn AUROC:", roc_auc_score(y_true, y_pred))
#     print("avg precision:", average_precision_score(y_true, y_pred))

#     # threshold 0.5
#     y_hat = (y_pred >= 0.5).astype(int)
#     print("Confusion matrix:\n", confusion_matrix(y_true, y_hat))
#     print(classification_report(y_true, y_hat, digits=4))

# ##################
#     pos_probs = y_pred[y_true == 1]
#     neg_probs = y_pred[y_true == 0]

#     def quick_stats(a):
#         return {
#             "count": len(a),
#             "mean": float(np.mean(a)),
#             "std": float(np.std(a)),
#             "median": float(np.median(a)),
#             "p10": float(np.percentile(a, 10)),
#             "p90": float(np.percentile(a, 90))
#         }

#     print("Pos stats:", quick_stats(pos_probs))
#     print("Neg stats:", quick_stats(neg_probs))

#     # show overlap: fraction of negatives above 0.5, fraction of positives below 0.5
#     print("neg_above_0.5:", np.mean(neg_probs > 0.5))
#     print("pos_below_0.5:", np.mean(pos_probs < 0.5))

##################################3

#     prec, rec, thr = precision_recall_curve(y_true, y_pred)
#     # Print some thresholds and (prec,rec) pairs
#     for i in np.linspace(0, len(thr)-1, 10, dtype=int):
#         print("thr {:.3f} -> prec {:.3f}, rec {:.3f}".format(thr[i], prec[i], rec[i]))

# ######################


#     train_df = pd.read_parquet("data/interactions/train_val_set.parquet")
#     test_df  = pd.read_parquet("data/interactions/test_set.parquet")

#     # use the ID columns your pipeline uses, e.g. Sequence_1_emb_ID and Sequence_2_emb_ID
#     train_pairs = set(zip(train_df['Sequence_1_emb_ID'], train_df['Sequence_2_emb_ID']))
#     test_pairs  = set(zip(test_df['Sequence_1_emb_ID'], test_df['Sequence_2_emb_ID']))
#     overlap = train_pairs.intersection(test_pairs)
#     print("pair overlap count:", len(overlap))




# first test

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # model.eval()

    # # --- Collect predictions ---
    # y_true_list, y_pred_list = [], []

    # with torch.no_grad():
    #     for rna_embed, protein_embed, y, _ in test_dataloader:
    #         rna_embed = rna_embed.to(device)
    #         protein_embed = protein_embed.to(device)
    #         y = y.to(device)

    #         probs = model(rna_embed, protein_embed)  # Already probabilities
    #         y_true_list.append(y.cpu().numpy())
    #         y_pred_list.append(probs.cpu().numpy())

    # # After loop
    # y_true = np.concatenate(y_true_list)
    # y_pred = np.concatenate(y_pred_list)

    # # Debug dataset balance
    # print("Original y_true distribution:", np.unique(y_true, return_counts=True))

    # # --- Permutation sanity test ---
    # shuffled_y = np.random.permutation(y_true)
    # print("Shuffled y_true distribution:", np.unique(shuffled_y, return_counts=True))

    # if len(np.unique(shuffled_y)) == 2:
    #     auroc_shuffled = roc_auc_score(shuffled_y, y_pred)
    #     print("Shuffled AUROC:", auroc_shuffled)
    # else:
    #     print("Skipping AUROC — only one class present in shuffled_y")
    
    # for i, (rna_embed, protein_embed, y_true_batch, _) in enumerate(test_dataloader):
    #     with torch.no_grad():
    #         probs = model(rna_embed.to(device), protein_embed.to(device)).cpu().numpy()
    #     print(f"\nBatch {i}")
    #     print("y_true:", y_true_batch.numpy()[:10])
    #     print("probs:", probs[:10].flatten())
    #     if i == 9:  # stop after 5 batches
    #         break

    # for i, (rna_embed, protein_embed, y_true_batch, _) in enumerate(test_dataloader):
    #     if 0.0 in y_true_batch.numpy():
    #         with torch.no_grad():
    #             probs = model(rna_embed.to(device), protein_embed.to(device)).cpu().numpy()
    #         print(f"\nBatch {i}")
    #         print("y_true:", y_true_batch.numpy())
    #         print("probs:", probs.flatten())
    #         break


    model_name = args.checkpoint_path.split('/')[-1].split('.ckpt')[0]
    test_name = args.test_set_path.split('/')[-1].split('.parquet')[0]

    with open(os.path.join(test_results_dir, f'{model_name} on {test_name}.txt'), 'w') as file:
        for key, value in trainer.logged_metrics.items():
            file.write(f'{key}: {value}\n')
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Command line options for the script")
    
    parser.add_argument("--loader_type", type=str, default="RPIDataset", help="Type of dataloader")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--rna_embeddings_path", type=str, default="data/embeddings/rpi2825/rna_embeddings.npy", help="Path to all RNA embeddings")
    parser.add_argument("--protein_embeddings_path", type=str, default="data/embeddings/rpi2825/protein_embeddings.npy", help="Path to all protein embeddings")
    parser.add_argument("--test_set_path", type=str, default="data/interactions/rpi2825_test_set.parquet", help="Path to the test set file")

    # parser.add_argument("--rna_embeddings_path", type=str, default="data/embeddings/rna_embeddings.npy", help="Path to all RNA embeddings")
    # parser.add_argument("--protein_embeddings_path", type=str, default="data/embeddings/protein_embeddings.npy", help="Path to all protein embeddings")
    # parser.add_argument("--test_set_path", type=str, default="data/interactions/test_set.parquet", help="Path to the test set file")

    parser.add_argument("--checkpoint_path", type=str, default="/gpfs/bwfor/work/ws/fr_jg590-fr_jg590-restored/retrain_tr_s159072_25e/last-reproducing11_retrain_tr_s159072_25e, lr: 0.00039457092606005325,             wd: 0.0005081310266379466, dr: 0.16244020564524297, seed: 159072.ckpt", help="Path to model's checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")

    args = parser.parse_args()

    main(args)