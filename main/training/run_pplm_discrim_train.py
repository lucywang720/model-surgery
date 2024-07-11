#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import argparse
import csv
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertModel, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from pplm_classification_head import ClassificationHead

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size=None,
            pretrained_model="gpt2-medium",
            classifier_head=None,
            cached_mode=False,
            p=0,
            layer=31
            device='cpu',
    ):
        super(Discriminator, self).__init__()
        # if pretrained_model.startswith("gpt2"):
        if 'gpt2' in pretrained_model:
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
            self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.transformer.config.hidden_size
        elif pretrained_model.startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.encoder = BertModel.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.config.hidden_size
        else:
            self.encoder = AutoModelForCausalLM.from_pretrained(pretrained_model, output_hidden_states=True)
            # print(self.encoder)

            # def hook_fn(module, input, output):
            #     module.down_proj_output = output
            # for layer in self.encoder.model.layers:
            #     layer.mlp.down_proj.register_forward_hook(hook_fn)

            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.config.hidden_size
        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            print("building classification")
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                embed_size=self.embed_size,
                p=p
            )
        self.cached_mode = cached_mode
        self.layer = layer
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        # print(mask.shape)
        # print(x.ne(0))
        # mask = mask[:,-1,:]
        # print(x.shape, mask.shape)
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden = self.encoder.transformer(x)[1]
            hidden = torch.stack(hidden)
            print(hidden.shape)
        elif hasattr(self.encoder, 'bert'):
            # for bert
            hidden = self.encoder(x)[2][0]
        else:
            hidden = self.encoder(x)[2][self.layer]

        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden
        # return hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            # print(x)
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs
        # return logits

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        if self.cached_mode:
            input_t = self.avg_representation(input_t)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def train_epoch(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), loss.item()
                )
            )


def evaluate_performance(data_loader, discriminator, device='cpu', train_type='False'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    c1_cnt, c0_cnt, c1_sum, c0_sum = 0, 0, 0, 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            # print(c0_sum, c1_sum)
            input_t, target_t = input_t.to(device), target_t.to(device)
            # print(input_t[0], target_t[0])
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

            pred_t = output_t.argmax(dim=1, keepdim=True)

            total_samples = len(target_t)
            c0_cnt += pred_t[target_t == 0].eq(target_t.view_as(pred_t)[target_t == 0]).sum().item()
            c0_sum += (target_t == 0).sum().item()

            c1_cnt += pred_t[target_t == 1].eq(target_t.view_as(pred_t)[target_t == 1]).sum().item()
            c1_sum += (target_t == 1).sum().item()

            # if c0_sum + c1_sum == 16*64:
            #     break


    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * accuracy
        )
    )
    if train_type:
        print("Train: ", "c0", c0_cnt / c0_sum, "c1", c1_cnt / c1_sum)
    else:
        print("Test: ", "c0", c0_cnt / c0_sum, "c1", c1_cnt / c1_sum)

    return test_loss, accuracy


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader


def get_idx2class(dataset_fp):
    classes = set()
    with open(dataset_fp) as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in tqdm(csv_reader, ascii=True):
            if row:
                classes.add(row[0])

    return sorted(classes)


def get_generic_dataset(dataset_fp, tokenizer, device,
                        idx2class=None, add_eos_token=False):
    if not idx2class:
        idx2class = get_idx2class(dataset_fp)
    class2idx = {c: i for i, c in enumerate(idx2class)}

    x = []
    y = []
    with open(dataset_fp) as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(tqdm(csv_reader, ascii=True)):
            if row:
                label = row[0]
                text = row[1]

                try:
                    seq = tokenizer.encode(text)
                    if (len(seq) < max_length_seq):
                        if add_eos_token:
                            seq = [50256] + seq
                        seq = torch.tensor(
                            seq,
                            device=device,
                            dtype=torch.long
                        )

                    else:
                        print(
                            "Line {} is longer than maximum length {}".format(
                                i, max_length_seq
                            ))
                        continue

                    x.append(seq)
                    y.append(class2idx[label])

                except:
                    print("Error tokenizing line {}, skipping it".format(i))
                    pass

    return Dataset(x, y)


def train_discriminator(
        dataset,
        dataset_fp=None,
        pretrained_model="gpt2-medium",
        epochs=10,
        learning_rate=0.0001,
        batch_size=64,
        log_interval=10,
        save_model=False,
        cached=False,
        no_cuda=False,
        output_fp='.',
        p=0.0,
        layer=layer
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    add_eos_token = pretrained_model.startswith("gpt2")

    if save_model:
        if not os.path.exists(output_fp):
            os.makedirs(output_fp)
    classifier_head_meta_fp = os.path.join(
        output_fp, "{}_classifier_head_meta.json".format(dataset)
    )
    classifier_head_fp_pattern = os.path.join(
        output_fp, "32_toxic_{}_classifier_head_epoch".format(dataset) + "_{}.pt"
    )

    print("Preprocessing {} dataset...".format(dataset))
    start = time.time()

    if dataset == "emotion":
        idx2class = ["non_negative", "negative"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            layer=layer,
            device=device
        ).to(device)

        x = []
        y = []
        import random
        import ast
        with open(args.dataset_fp) as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                d = ast.literal_eval(line)
                seq = discriminator.tokenizer.encode(d["text"])
                seq = torch.tensor(
                    seq, device=device, dtype=torch.long
                )
                print(d)
                if int(d["label"]) == 0:
                    x.append(seq)
                    y.append(int(d["label"]))
                else:
                    x.append(seq)
                    y.append(int(d["label"]))

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }

    elif dataset == "toxic":
        idx2class = ["non_toxic", "toxic"]
        class2idx = {c: i for i, c in enumerate(idx2class)}

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            p=p,
            layer=layer,
            device=device
        ).to(device)

        x = []
        y = []
        import random
        with open("datasets/toxic/toxic_train.txt") as f:
            for i, line in enumerate(tqdm(f, ascii=True)):
                try:
                    d = eval(line)
                    seq = discriminator.tokenizer.encode(d["text"])
                    seq = torch.tensor(
                        seq, device=device, dtype=torch.long
                    )
                    x.append(seq)
                    y.append(int(np.sum(d["label"]) > 0))
                except:
                    print("Error evaluating / tokenizing"
                          " line {}, skipping it".format(i))
                    pass

        full_dataset = Dataset(x, y)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": class2idx,
            "default_class": 0,
        }


    else: 

        if dataset_fp is None:
            raise ValueError("When generic dataset is selected, "
                             "dataset_fp needs to be specified aswell.")

        idx2class = get_idx2class(dataset_fp)

        discriminator = Discriminator(
            class_size=len(idx2class),
            pretrained_model=pretrained_model,
            cached_mode=cached,
            device=device
        ).to(device)

        full_dataset = get_generic_dataset(
            dataset_fp, discriminator.tokenizer, device,
            idx2class=idx2class, add_eos_token=add_eos_token
        )
        train_size = int(0.99 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size]
        )

        discriminator_meta = {
            "class_size": len(idx2class),
            "embed_size": discriminator.embed_size,
            "pretrained_model": pretrained_model,
            "class_vocab": {c: i for i, c in enumerate(idx2class)},
            "default_class": 0,
        }

    end = time.time()
    print("Preprocessed {} data points".format(
        len(train_dataset) + len(test_dataset))
    )
    print("Data preprocessing took: {:.3f}s".format(end - start))

    if cached:
        print("Building representation cache...")

        start = time.time()

        train_loader = get_cached_data_loader(
            train_dataset, batch_size, discriminator,
            shuffle=True, device=device
        )

        test_loader = get_cached_data_loader(
            test_dataset, batch_size, discriminator, device=device
        )

        end = time.time()
        print("Building representation cache took: {:.3f}s".format(end - start))

    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  collate_fn=collate_fn)

    if save_model:
        with open(classifier_head_meta_fp, "w") as meta_file:
            json.dump(discriminator_meta, meta_file)

    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.9, 0.99))

    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(
            discriminator=discriminator,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            device=device
        )
        # discriminator.classifier_head.load_state_dict(torch.load('/cluster/home2/whq/PPLM_llama/toxic_classifier_head_epoch_10.pt'))
        test_loss, test_accuracy = evaluate_performance(
            data_loader=test_loader,
            discriminator=discriminator,
            device=device
        )

        # train_loss, train_accuracy = evaluate_performance(
        #     data_loader=train_loader,
        #     discriminator=discriminator,
        #     device=device
        # )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        # train_losses.append(train_loss)
        # train_accuracies.append(train_accuracy)

        print("\nExample prediction")
        predict(example_sentence, discriminator, idx2class,
                cached=cached, device=device)

        if save_model:
            # torch.save(discriminator.state_dict(),
            #           "{}_discriminator_{}.pt".format(
            #               args.dataset, epoch + 1
            #               ))
            torch.save(discriminator.get_classifier().state_dict(),
                       classifier_head_fp_pattern.format(epoch + 1))

    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    print("Test performance per epoch")
    print("epoch\tloss\tacc")
    for e, (loss, acc) in enumerate(zip(test_losses, test_accuracies)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))


    # min_loss = float("inf")
    # min_loss_epoch = 0
    # max_acc = 0.0
    # max_acc_epoch = 0
    # print("Train performance per epoch")
    # print("epoch\tloss\tacc")
    # for e, (loss, acc) in enumerate(zip(train_losses, train_accuracies)):
    #     print("{}\t{}\t{}".format(e + 1, loss, acc))
    #     if loss < min_loss:
    #         min_loss = loss
    #         min_loss_epoch = e + 1
    #     if acc > max_acc:
    #         max_acc = acc
    #         max_acc_epoch = e + 1
    # print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    # print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))

    return discriminator, discriminator_meta


def load_classifier_head(weights_path, meta_path, p, device='cpu'):
    with open(meta_path, 'r', encoding="utf8") as f:
        meta_params = json.load(f)
    classifier_head = ClassificationHead(
        class_size=meta_params['class_size'],
        embed_size=meta_params['embed_size'],
        p=p
    ).to(device)
    classifier_head.load_state_dict(
        torch.load(weights_path, map_location=device))
    classifier_head.eval()
    return classifier_head, meta_params


def load_discriminator(weights_path, meta_path, p, device='cpu'):
    classifier_head, meta_param = load_classifier_head(
        weights_path, meta_path, p, device
    )
    discriminator =  Discriminator(
        pretrained_model=meta_param['pretrained_model'],
        classifier_head=classifier_head,
        cached_mode=False,
        device=device
    )
    return discriminator, meta_param


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a discriminator on top of GPT-2 representations")
    parser.add_argument("--dataset", type=str, default="SST",
                        choices=("SST", "clickbait", "toxic", "generic", "emotion"),
                        help="dataset to train the discriminator on."
                             "In case of generic, the dataset is expected"
                             "to be a TSBV file with structure: class \\t text")
    parser.add_argument("--dataset_fp", type=str, default="",
                        help="File path of the dataset to use. "
                             "Needed only in case of generic datadset")
    parser.add_argument("--pretrained_model", type=str, default="gpt2-medium",
                        help="Pretrained model to use as encoder")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learnign rate")
    parser.add_argument("--batch_size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--save_model", action="store_true",
                        help="whether to save the model")
    parser.add_argument("--cached", action="store_true",
                        help="whether to cache the input representations")
    parser.add_argument("--no_cuda", action="store_true",
                        help="use to turn off cuda")
    parser.add_argument("--output_fp", default=".",
                        help="path to save the output to")
    parser.add_argument("--p", type=float, default=0.0,
                        help="path to save the output to")
    parser.add_argument("--layer", type=int, default=32,
                        help="layer index for training")
    args = parser.parse_args()

    train_discriminator(**(vars(args)))
