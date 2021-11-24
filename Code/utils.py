import random
from functools import partial

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, DistributedBatchSampler, BatchSampler
from paddlenlp.data import Pad


def print_args(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')

def set_seed(seed):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(seed)
    np.random.seed(seed)
    # Maybe different op seeds(for dropout) for different procs is better.
    paddle.seed(seed + dist.get_rank())

def post_process_response(token_ids, tokenizer):
    """Post-process the decoded sequence. Truncate from the first <eos>."""
    eos_pos = len(token_ids)
    for i, tok_id in enumerate(token_ids):
        if tok_id == tokenizer.sep_token_id:
            eos_pos = i
            break
    token_ids = token_ids[:eos_pos]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = tokenizer.merge_subword(tokens)
    return token_ids, tokens

def get_in_turn_repetition(pred, is_cn=False):
    """Get in-turn repetition."""
    if len(pred) == 0:
        return 1.0
    if isinstance(pred[0], str):
        pred = [tok.lower() for tok in pred]
        if is_cn:
            pred = "".join(pred)
    tri_grams = set()
    for i in range(len(pred) - 2):
        tri_gram = tuple(pred[i:i + 3])
        if tri_gram in tri_grams:
            return True
        tri_grams.add(tri_gram)
    return False


def select_response(ids,
                    scores,
                    tokenizer,
                    max_dec_len=None,
                    num_return_sequences=1,
                    keep_space=True):
    ids = ids.numpy().tolist()
    scores = scores.numpy()

    if len(ids) != len(scores) or (len(ids) % num_return_sequences) != 0:
        raise ValueError(
            "the length of `ids` is {}, but the `num_return_sequences` is {}".
            format(len(ids), num_return_sequences))

    group = []
    tmp = []
    for pred, score in zip(ids, scores):
        pred_token_ids, pred_tokens = post_process_response(pred, tokenizer)
        num_token = len(pred_token_ids)
        if keep_space:
            response = " ".join(pred_tokens)
        else:
            response = "".join(pred_tokens)

        in_turn_repetition = get_in_turn_repetition(
            pred_tokens, True) or get_in_turn_repetition(pred_token_ids)
        # not ending
        if max_dec_len is not None and num_token >= max_dec_len:
            score -= 1e3
        elif in_turn_repetition:
            score -= 1e3

        tmp.append([response, score])
        if len(tmp) == num_return_sequences:
            group.append(tmp)
            tmp = []

    results = []
    for preds in group:
        preds = sorted(preds, key=lambda x: -x[1])
        results.append(preds[0][0])
    return results