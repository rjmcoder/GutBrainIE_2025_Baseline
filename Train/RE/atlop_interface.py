import argparse
import os
import datetime

import numpy as np
import torch
#from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, official_evaluate
from evaluation_relationwise import official_evaluate_per_rel, official_evaluate_long_tail, official_evaluate_sklearn
# import wandb
from tqdm import tqdm
import pandas as pd
from Logger import Logger


def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        scaler = GradScaler()
        args.logger.logger.info("Total steps: {}".format(total_steps))
        args.logger.logger.info("Warmup steps: {}".format(warmup_steps))
        for epoch in tqdm(train_iterator, desc="Train epoch"):
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaler.scale(loss).backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                # wandb.log({"loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    args.logger.logger.info(
                        f"Starting evaluation step ...")
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")

                    ckpt_file = os.path.join(args.save_path, "checkpoint.ckpt")
                    args.logger.logger.info(f"[CHECKPOINT] Saving model checkpoint at epoch {epoch} into {ckpt_file} ...")
                    torch.save(model.state_dict(), ckpt_file)
                    # wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        ckpt_file = os.path.join(args.save_path, "best.ckpt")
                        args.logger.logger.info(f"[BEST] Saving best model (epoch: {epoch}) into {ckpt_file} ...")
                        torch.save(model.state_dict(), ckpt_file)
                        with open(f"{args.save_path}/result.json", "w") as fh:
                            json.dump(pred, fh)
                        """if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)"""
        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    args.logger.logger.info(f"Length preds: {len(preds)}")
    ans = to_official(args, preds, features)
    args.logger.logger.info(f"Length answer: {len(ans)}")
    if len(ans) > 0:
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, tag="train")
    else:
        #raise ValueError("Answer length is zero!")
        print("Answer length is zero!")
        best_f1 = 0
        best_f1_ign = 0
    output = {
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
    }
    return best_f1, output


def evaluate_micro(args, model, features, Logger, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        print(f'\n\n ====== INPUTS ======\n{inputs.items()}\n\n')

        with torch.no_grad():
            try:
                pred, *_ = model(**inputs)
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
                preds.append(pred)
            except:
                print(f'\n\n ====== INPUTS ======\n{inputs.items()}\n\n')
                raise Exception("Error in getting or retrieving predictions!")

    preds = np.concatenate(preds, axis=0).astype(np.float32)

    official_results = to_official(args, preds, features)

    Logger.logger.info(f"Length of official results: {len(official_results)}")

    if len(official_results) > 0:
        if tag == "dev":
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir)
        else:
            best_re, best_evi, best_re_ign, _ = official_evaluate(official_results, args.data_dir)
    else:
        best_re = best_evi = best_re_ign = [-1, -1, -1]
    Logger.logger.info(f"best_re: {best_re}, best_evi: {best_evi}, best_re_ign: {best_re_ign}")
    output = {
        tag + "_rel": [i * 100 for i in best_re],
        tag + "_rel_ign": [i * 100 for i in best_re_ign],
        tag + "_evi": [i * 100 for i in best_evi],
    }
    scores = {"dev_F1": best_re[-1] * 100, "dev_evi_F1": best_evi[-1] * 100, "dev_F1_ign": best_re_ign[-1] * 100}

    return scores, output, official_results


def evaluate_per_rel(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        args.logger.logger.info(f"[LABELS] length of labels: {len(batch[2])}")
        args.logger.logger.info(f"[LABELS]length of first element labels: {len(batch[2][0])}")
        tot_pairs = sum(len(batch[2][i]) for i in range(len(batch[2])))
        args.logger.logger.info(f"[LABELS]Total number of pairs: {tot_pairs}")
        args.logger.logger.info(f"[LABELS] length of first element of first element labels: {len(batch[2][0][0])}")
        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            args.logger.logger.info(f"[PREDS] Length pred: {len(pred)}")
            args.logger.logger.info(f"[PREDS] Length first element preds: {len(pred[0])}")
            # args.logger.logger.info(f"preds: {pred[0]}")
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    args.logger.logger.info(f"Length preds: {len(preds)}")
    args.logger.logger.info(f"Length first element preds: {len(preds[0])}")
    args.logger.logger.info(f"preds: {preds[0]}")

    official_results = to_official(args, preds, features)

    if len(official_results) > 0:
        if args.eval_mode == "per-relation":
            return official_evaluate_per_rel(args.logger, official_results, args.data_dir, args.train_file,
                                             args.test_file)
        else:
            return official_evaluate_long_tail(args.logger, official_results, args.data_dir, args.train_file,
                                               args.test_file)
    else:
        return {}

def evaluate_sklearn(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False)
    preds = []
    y_true = []
    y_pred = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }
        # labels
        for i in range(len(batch[2])):
            for j in range(len(batch[2][i])):
                y_true.append(batch[2][i][j])
        args.logger.logger.info(f"[LABELS] length of labels: {len(batch[2])}")
        args.logger.logger.info(f"[LABELS]length of first element labels: {len(batch[2][0])}")
        tot_pairs = sum(len(batch[2][i]) for i in range(len(batch[2])))
        args.logger.logger.info(f"[LABELS]Total number of pairs: {tot_pairs}")
        args.logger.logger.info(f"[LABELS] length of first element of first element labels: {len(batch[2][0][0])}")
        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            args.logger.logger.info(f"[PREDS] Length pred: {len(pred)}")
            # args.logger.logger.info(f"[PREDS] Length first element preds: {len(pred[0])}")
            # for pi in range(len(pred)):
                # y_pred.append(pred[pi])
            # args.logger.logger.info(f"preds: {pred[0]}")
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    for i in range(len(preds)):
        y_pred.append(preds[i])
    args.logger.logger.info(f"Length preds: {len(preds)}")
    args.logger.logger.info(f"Length first element preds: {len(preds[0])}")
    args.logger.logger.info(f"preds: {preds[0]}")

    # official_results = to_official(args, preds, features)

    if len(y_true) == len(y_pred):
        return official_evaluate_sklearn(args.logger, y_true, y_pred, args.data_dir)
    else:
        raise ValueError(f"Lengths of y_true and y_pred are not equal! y_true: {len(y_true)}; y_pred: {len(y_pred)}")


def report(args, model, features):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(args, preds, features)
    return preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="dev.json", type=str)
    parser.add_argument("--pred_file", default="results.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--load_checkpoint", default="best.ckpt", type=str)
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--eval_mode", default="micro", type=str,
                        choices=["micro", "per-relation", "long_tail", "sklearn"])

    args = parser.parse_args()
    # wandb.init(project="DocRED")

    # create directory to save checkpoints and predicted files
    time = str(datetime.datetime.now()).replace(' ', '_')
    save_path_ = os.path.join(args.save_path, f"{time}")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logger_name = args.save_path + "/" + "atlop_run" + ".log"
    my_logger = Logger(logger_name)
    args.logger = my_logger

    # args.save_path = save_path_

    args.logger.logger.info(f"Number of devices available: {torch.cuda.device_count()}")
    args.logger.logger.info(f"Cuda current device: {torch.cuda.current_device()}")
    device = torch.device(torch.cuda.current_device())
    # my_logger.logger.info(f"*** Model will be loaded to device: {device} ***")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, args.data_dir, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, args.data_dir, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, args.data_dir, tokenizer, max_seq_length=args.max_seq_length)

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels)
    model.to(args.device)

    if args.load_path == "":  # Training
        train(args, model, train_features, dev_features, test_features)
    else:  # Testing
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        """model.load_state_dict(torch.load(args.load_path, map_location=args.device))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        pred = report(args, model, test_features)
        with open("result.json", "w") as fh:
            json.dump(pred, fh)"""
        model.load_state_dict(torch.load(os.path.join(args.load_path, args.load_checkpoint), map_location=args.device))
        basename = os.path.splitext(args.test_file)[0]

        if args.eval_mode == "per-relation":
            args.logger.logger.info(f"Evaluating per-relation ..")
            output_dict = evaluate_per_rel(args, model, test_features, tag="test")

            score_path = os.path.join(args.load_path, f"{basename}_scores_relations.csv")
            my_logger.logger.info(f"saving evaluations into {score_path} ...")
            headers = ["precision", "recall", "F1", "prec_ign", "rec_ign", "F1_ign", "prec_evi", "rec_evi",
                       "F1_evi"]
            scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
            scores_pd.to_csv(score_path, sep='\t')

        elif args.eval_mode == "long_tail":
            args.logger.logger.info(f"Evaluating micro and macro ..")
            output_dict = evaluate_per_rel(args, model, test_features, tag="test")

            score_path = os.path.join(args.load_path, f"{basename}_scores_detailed.csv")
            my_logger.logger.info(f"saving evaluations into {score_path} ...")
            headers = ["F1", "precision", "recall"]
            scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
            scores_pd.to_csv(score_path, sep='\t')

        elif args.eval_mode == "sklearn":
            args.logger.logger.info(f"Evaluating micro and macro using sklearn ..")
            output_dict, classification_report = evaluate_sklearn(args, model, test_features, tag="test")

            report_path = os.path.join(args.load_path, f"{basename}_classification_report.csv")
            my_logger.logger.info(f"saving classification report into {report_path} ...")
            classification_report.to_csv(report_path, sep='\t')
            score_path = os.path.join(args.load_path, f"{basename}_scores_detailed_sklearn.csv")
            my_logger.logger.info(f"saving evaluations into {score_path} ...")
            headers = ["F1", "precision", "recall"]
            scores_pd = pd.DataFrame.from_dict(output_dict, orient="index", columns=headers)
            scores_pd.to_csv(score_path, sep='\t')
        else:
            args.logger.logger.info(f"Evaluating micro ..")
            test_scores, test_output, official_results = evaluate_micro(args, model, test_features, my_logger,
                                                                        tag="test")
            # wandb.log(test_scores)

            offi_path = os.path.join(args.load_path, args.pred_file)
            score_path = os.path.join(args.load_path, f"{basename}_scores.csv")

            args.logger.logger.info(f"saving official predictions into {offi_path} ...")
            json.dump(official_results, open(offi_path, "w"))

            args.logger.logger.info(f"saving evaluations into {score_path} ...")
            headers = ["precision", "recall", "F1"]
            scores_pd = pd.DataFrame.from_dict(test_output, orient="index", columns=headers)
            args.logger.logger.info(scores_pd)
            scores_pd.to_csv(score_path, sep='\t')


if __name__ == "__main__":
    main()
