# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.distributions as dis
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.sequence_generator import EnsembleModel
from numpy.random import uniform


def smrt_cross_entropy_loss(lprobs, orig_lprobs, target, original_target, epsilon, ignore_index=None,
                            reduce=True, paraphraser_probs=None):
    """
    epsilon is labelsmooth amount
    returns loss, nll loss where:
    nll loss remains the standard nll loss against the original target
    (mirrors label_smoothed xent, where the non labelsmooth loss is also returned)

    if paraphraser_probs are passed in: loss is the  distribution loss against the paraphraser (no label smoothing).
    if not: loss is computed against the target (with label smoothing)
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    if paraphraser_probs is not None:
        paraphraser_probs = paraphraser_probs.reshape(-1, paraphraser_probs.shape[-1])
        xent_loss = - lprobs * paraphraser_probs
        xent_loss = xent_loss.sum(1).unsqueeze(1)
    else:
        xent_loss = -lprobs.gather(dim=-1, index=target)
    orig_nll_loss = -orig_lprobs.gather(dim=-1, index=original_target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        orig_pad_mask = original_target.eq(ignore_index)
        if pad_mask.any():
            xent_loss.masked_fill_(pad_mask, 0.)
            orig_nll_loss.masked_fill_(orig_pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        xent_loss = xent_loss.squeeze(-1)
        orig_nll_loss = orig_nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        xent_loss = xent_loss.sum()
        orig_nll_loss = orig_nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * xent_loss + eps_i * smooth_loss
    return loss, orig_nll_loss


@register_criterion('smrt_cross_entropy')
class SMRTCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

        self.paraphraser_model = args.paraphraser_model
        paraphraser_data_dir = args.paraphraser_data_dir

        arg_overrides = dict(data=paraphraser_data_dir,
                             source_lang=args.target_lang,
                             target_lang=args.target_lang)
        from fairseq import checkpoint_utils

        self.paraphraser_model, _paraphraser_model_args = \
            checkpoint_utils.load_model_ensemble(filenames=[self.paraphraser_model, ],
                                                 arg_overrides=arg_overrides, task=None)

        # we only handle the case of a single paraphraser model
        self.paraphraser_model = self.paraphraser_model[0]
        self.distribution_loss = True
        self.prob_use_smrt = args.prob_use_smrt
        self.sample_topN = args.paraphraser_sample_topN
        self.paraphraser_lang_prefix = args.paraphraser_lang_prefix

        self.paraphraser_temperature = 1.0

        self.task = task

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing. '
                                 'Not applied when applying a distribution loss towards the paraphraser distribution')
        parser.add_argument('--paraphraser-model', type=str, help='model of the paraphraser')
        parser.add_argument('--paraphraser-data-dir', type=str, help='data dir of paraphrase (location of dictionary)')
        parser.add_argument('--paraphraser-lang-prefix', type=str, default="",
                            help='token to force decode in the first step of the paraphraser decode. '
                                 'Used to specify language prefix for a multilingual paraphraser.')
        parser.add_argument('--prob-use-smrt', type=float, default=0.5,
                            help='use paraphraser simulated multi-reference training with probability prob_use_smrt, '
                                 'label smoothed cross entroy with probability (1.0 - prob_use_smrt). default=0.5')
        parser.add_argument('--paraphraser-sample-topN', type=int, default=100,
                            help='sample from top paraphraser_sample_topN tokens '
                                 'from paraphraser softmax output. default=100')
        # fmt: on

    def paraphrase_sample(self, model, sample):
        previous_seq, paraphraser_predictions, paraphraser_n_tokens, paraphraser_probs = \
            self._paraphrase_sample(model, sample, sample_topN=self.sample_topN)

        # contiguous() avoids the error discussed here: https://github.com/agrimgupta92/sgan/issues/22
        previous_seq = previous_seq.contiguous()
        paraphraser_predictions = paraphraser_predictions.contiguous()
        new_sample = dict(id=sample['id'],
                          nsentences=sample['nsentences'],
                          ntokens=paraphraser_n_tokens,
                          net_input=dict(src_tokens=sample['net_input']['src_tokens'],
                                         src_lengths=sample['net_input']['src_lengths'],
                                         prev_output_tokens=previous_seq),
                          target=paraphraser_predictions, )
        return new_sample, paraphraser_probs

    @torch.no_grad()
    def _paraphrase_sample(self, model, sample, sample_topN):
        """
        model: MT model being trained
        sample: fairseq data structure for training batch
        sample_topN: number of top candidates to sample from in paraphraser output softmax
        """
        # disable training model dropout
        model.eval()

        # disable paraphraser dropout
        # train() on the paraphraser model automatically gets when train() is called on the criteron,
        # we need to set it back to eval mode
        self.paraphraser_model.eval()  # this should disable dropout
        self.paraphraser_model.training = False  # not sure if this does anything

        pad = self.task.target_dictionary.pad()
        eos = self.task.target_dictionary.eos()
        bos = self.task.target_dictionary.bos()

        assert pad == self.task.source_dictionary.pad()
        assert eos == self.task.source_dictionary.eos()
        assert bos == self.task.source_dictionary.bos()

        # we don't know how long the paraphrase will be, so we take the target length and increase it a bit.
        target_length = sample['target'].shape[1]
        max_paraphrase_length = int(2 * target_length) + 3

        batch_size = sample['net_input']['prev_output_tokens'].shape[0]

        combined_tokens = sample['net_input']['prev_output_tokens'][:, :1]
        combined_tokens[:, :] = eos  # eos to match 'bug' in fairseq ("should" be bos)

        # make the target look like a source, to feed it into the paraphraser encoder
        paraphraser_src_lengths = torch.ones(batch_size, dtype=torch.int)
        paraphraser_source = sample['target'].new_zeros(tuple(sample['target'].shape)) + pad
        for i in range(batch_size):
            n_pad = (sample['target'][i] == pad).sum()
            paraphraser_src_lengths[i] = target_length - n_pad
            paraphraser_source[i, n_pad:target_length] = sample['target'][i, :target_length - n_pad]

        paraphraser_prediction_tokens_list = []
        paraphraser_probs_list = []

        paraphraser = EnsembleModel([self.paraphraser_model, ])

        paraphraser_encoder_out = paraphraser.forward_encoder(dict(src_tokens=paraphraser_source,
                                                                   src_lengths=paraphraser_src_lengths))

        if self.paraphraser_lang_prefix:
            # take one step update the state of the paraphraser, so that the "first" time step
            #    in the loop below will pass in the language prefix
            paraphraser_probs, _ = paraphraser.forward_decoder(tokens=combined_tokens,
                                                               encoder_outs=paraphraser_encoder_out,
                                                               temperature=self.paraphraser_temperature,
                                                               use_log_probs=False)

            prefixed_combined_tokens = sample['net_input']['prev_output_tokens'][:, :2]
            prefixed_combined_tokens[:, 0] = eos  # eos to match bug in fairseq ("should" be bos)
            prefixed_combined_tokens[:, 1] = self.task.target_dictionary.index(self.paraphraser_lang_prefix)
        else:
            prefixed_combined_tokens = None

        done = [False, ] * batch_size
        for ii in range(max_paraphrase_length + 1):
            # paraphraser prefix may or may not have the language tag prepended (after the go symbol) to input
            if prefixed_combined_tokens is None:
                paraphraser_combined_tokens = combined_tokens
            else:
                paraphraser_combined_tokens = prefixed_combined_tokens

            # this is used to compute the loss
            paraphraser_probs, _ = paraphraser.forward_decoder(tokens=paraphraser_combined_tokens,
                                                               encoder_outs=paraphraser_encoder_out,
                                                               temperature=self.paraphraser_temperature,
                                                               use_log_probs=False)

            # this is used to generate the previous context word
            paraphraser_probs_context = paraphraser_probs

            # save the paraphraser predictions to train toward (if we don't have a distribution loss)
            _, paraphraser_predictions = torch.max(paraphraser_probs, 1)
            if self.distribution_loss:
                paraphraser_probs_list.append(paraphraser_probs.unsqueeze(1))

            # paraphraser predictions are simply the most likely next word, according to the paraphraser
            paraphraser_prediction_tokens_list.append(paraphraser_predictions.reshape((-1, 1)))

            combined_probs = paraphraser_probs_context
            # disallow length=0 paraphrases
            if ii == 0:
                combined_probs[:, eos] = 0.0
            # disallow other undefined behavior
            combined_probs[:, pad] = 0.0
            combined_probs[:, bos] = 0.0

            if ii == max_paraphrase_length or all(done):
                break

            # sample from top N of paraphraser distribution
            if sample_topN == 1:
                _, combined_predictions = torch.max(combined_probs, 1)
                combined_predictions = combined_predictions.reshape((-1, 1))
            else:
                topk_val, topk_ind = torch.topk(combined_probs, sample_topN)
                # re-normalize top values
                topk_val2 = topk_val / topk_val.sum(dim=1).reshape((-1, 1))
                # make distribution from normalized topk values
                mm = dis.Categorical(topk_val2)  # this will take un-normalized
                # sample indexes into topk
                topk_idx_idx = mm.sample().reshape((-1, 1))
                # convert topk indexes back into vocab indexes
                combined_predictions = torch.cat([v[i] for i, v in zip(topk_idx_idx, topk_ind)]).reshape((-1, 1))

            for jj in range(batch_size):
                if combined_predictions[jj, 0] == eos:
                    done[jj] = True

            # append output tokens to input for next time step
            combined_tokens = torch.cat((combined_tokens, combined_predictions), 1)
            if prefixed_combined_tokens is not None:
                prefixed_combined_tokens = torch.cat((prefixed_combined_tokens, combined_predictions), 1)

        paraphraser_prediction_tokens = torch.cat(paraphraser_prediction_tokens_list, 1)
        if self.distribution_loss:
            paraphraser_probs_tokens = torch.cat(paraphraser_probs_list, 1)
        else:
            paraphraser_probs_tokens = None

        model.train()  # re-enable dropout

        # compute length of valid output for each sentence
        n_tokens = 0
        for i in range(batch_size):
            for j in range(paraphraser_prediction_tokens.shape[1]):
                if paraphraser_prediction_tokens[i, j] == eos:
                    n_tokens += j  # TODO should this include EOS? HK
                    # set anything after EOS to PAD
                    paraphraser_prediction_tokens[i, j + 1:paraphraser_prediction_tokens.shape[1]] = pad
                    break

        return combined_tokens, paraphraser_prediction_tokens, n_tokens, paraphraser_probs_tokens

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # only use paraphraser if model is being trained, not for validation
        # and also if we have the right probabilty
        if model.training and self.prob_use_smrt > uniform():
            new_sample, paraphraser_probs = self.paraphrase_sample(model=model, sample=sample)
        else:
            new_sample = sample
            paraphraser_probs = None

        new_net_output = model(**new_sample['net_input'])
        original_net_output = model(**sample['net_input'])

        loss, nll_loss = self.compute_loss(model, net_output=new_net_output, sample=new_sample,
                                           original_net_output=original_net_output,
                                           original_sample=sample,
                                           paraphraser_probs=paraphraser_probs, reduce=reduce)

        sample_size = new_sample['target'].size(0) if self.args.sentence_avg else new_sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    # sample and net_output are what we are currently training towards.
    # original_net_output and original_sample, are always the original;
    #   they are needed to compute original nll loss
    def compute_loss(self, model, net_output, sample, original_net_output,
                     original_sample, paraphraser_probs=None, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        orig_lprobs = model.get_normalized_probs(original_net_output, log_probs=True)
        orig_lprobs = orig_lprobs.view(-1, orig_lprobs.size(-1))

        orig_target = model.get_targets(original_sample, original_net_output).view(-1, 1)
        target = model.get_targets(sample, net_output).view(-1, 1)

        epsilon = self.eps if paraphraser_probs is None else 0.0
        loss, nll_loss = smrt_cross_entropy_loss(
            lprobs=lprobs, orig_lprobs=orig_lprobs, target=target, original_target=orig_target,
            paraphraser_probs=paraphraser_probs, epsilon=epsilon, ignore_index=self.padding_idx, reduce=reduce
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
