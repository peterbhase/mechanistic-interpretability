import fairseq
import transformers
import torch
import numpy as np
import pandas as pd
import argparse
import utils
import metrics
import time
import os
import jsonlines
from utils import str2bool
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from transformers import AdamW, get_scheduler
from torch.optim import SGD, RMSprop
from models.probe import Probe
from models.learned_optimizer import ModelWithLearnedOptimizer
from copy import deepcopy

def sentence_to_prompt(args, sentence):
    if args.sentence_to_prompt == 'first_three_words':
        sentence =  sentence.split()[:3]
    if args.sentence_to_prompt == 'random_cut':
        sentence = sentence.split()
        floor = min(5, len(sentence))
        upper_bound = .8 * len(sentence)
        num_keep = max(floor, np.random.randint(upper_bound))
        sentence = sentence[:num_keep]
    if args.sentence_to_prompt == 'after_first_capitalized_entity':
        sentence = sentence.split()
        seen_capitalized=False
        first_capitalized_word_idx = None
        last_first_capitalized_word_idx = None
        for i in range(1, len(sentence)):
            if sentence[i][0].isupper() and seen_capitalized is False:
                first_capitalized_word_idx = i
                seen_capitalized=True
            if not sentence[i][0].isupper() and seen_capitalized is True:
                last_first_capitalized_word_idx = i-1
                break
            # if at the end of the question without breaking
            if sentence[i][0].isupper() and i == len(sentence)-1:
                if first_capitalized_word_idx is not None:
                    last_first_capitalized_word_idx = first_capitalized_word_idx-1
        if seen_capitalized is False:
            last_first_capitalized_word_idx = max(min(4, len(sentence)), np.random.randint(.8 * len(sentence)-1))
        sentence = sentence[:(last_first_capitalized_word_idx+1)]
    return ' '.join(sentence)

def write_continuations(args, model, data_loader, tokenizer, nli_model, nli_tokenizer):
    '''
    write continuations of data points
    '''
    # local variables
    n_batches = len(data_loader)
    start_time = time.time()
    split_name = data_loader.dataset.split_name
    data_names = data_loader.dataset.data_names
    all_propositions = []
    data_id_to_continuations = {}
    save_data_dict = {}
    
    model.eval()
    for batch_num, batch in enumerate(data_loader):
        running_time = (time.time()-start_time)
        est_epoch_run_time = (running_time/(batch_num+1)*n_batches)
        batch_size = batch['main_input_ids'].size(0)
        print(f" {split_name.capitalize():5s} | Batch: {batch_num+1}/{n_batches} | Runtime: {running_time/60:.1f} min. / {est_epoch_run_time/60:.1f} min.", end='\r')
        
        text_data = batch['text_data']
        data_ids = [item['id'] for item in text_data]
        statement_name = 'implicit_rule' if args.dataset == 'LeapOfThought' else 'proposition'
        propositions = [item[statement_name] for item in text_data]
        contexts = [sentence_to_prompt(args, proposition) for proposition in propositions]
        input_kwargs = tokenizer(contexts, return_tensors='pt', padding=True, max_length=args.max_seq_len, truncation=True)
        all_propositions.extend(propositions)
        
        utils.move_kwargs_to_gpu(input_kwargs)
        # sample until valid continuations are found or max_iters reached
        valid_continuations = {id : [] for id in data_ids}
        iters, max_iters = 0, args.max_filter_iters
        while not all( len(continuations) >= args.min_samples_per_prompt for continuations in valid_continuations.values() ):
            with torch.no_grad():
                # sample new batch
                preds = model.generate(**input_kwargs, max_length=args.gen_max_seq_len, num_beams=args.beam_search_size, 
                                    num_return_sequences=args.num_samples, do_sample=args.do_sample, temperature=args.temperature, p=.9)
                preds = preds.reshape(batch_size, args.beam_search_size, args.gen_max_seq_len)
                pred_strs = [utils.decode_until_eos_token(tokenizer, point_samples) for point_samples in preds]
                pred_strs = utils.first_sentences_from_samples(pred_strs) # sample up until line breaks and periods

                # get valid continuations for continuation in num_return_samples
                for sample_num in range(args.num_samples):
                    # encode for nli model
                    samples = [continuations[sample_num] for continuations in pred_strs]
                    nli_inputs = nli_tokenizer( 
                        propositions, 
                        samples, 
                        return_tensors='pt',
                        padding=True,
                        max_length=args.max_seq_len, 
                        return_token_type_ids=True, 
                        truncation=True)
                    utils.move_kwargs_to_gpu(nli_inputs)
                    outputs = nli_model(**nli_inputs)
                    pred_probs = torch.softmax(outputs[0], dim=1)
                    if args.neutral_prob_threshold > 0:
                        neutral_pred_probs = pred_probs[:,1] # 1 is neutral class in ynie model
                        is_valid = neutral_pred_probs > args.neutral_prob_threshold
                    else:
                        is_valid = (pred_probs.argmax(-1) == 1) # is neutral pred
                    # check the prompt negations as well
                    if args.use_negated_propositions:
                        negated_propositions = [utils.negate_sentence(proposition) for proposition in propositions]
                        negated_nli_inputs = nli_tokenizer( 
                            negated_propositions, 
                            samples, 
                            padding=True,
                            return_tensors='pt',
                            max_length=args.max_seq_len, 
                            return_token_type_ids=True, 
                            truncation=True)
                        utils.move_kwargs_to_gpu(negated_nli_inputs)
                        outputs = nli_model(**negated_nli_inputs)
                        pred_probs = torch.softmax(outputs[0], dim=1)
                        if args.neutral_prob_threshold > 0:
                            neutral_pred_probs = pred_probs[:,1] # 1 is neutral class in ynie model
                            is_valid = is_valid * (neutral_pred_probs > args.neutral_prob_threshold)
                        else:
                            is_valid = is_valid * (pred_probs.argmax(-1) == 1) # is neutral pred
                    # filter for question mark in ZSRE
                    if args.dataset == 'ZSRE':
                        has_question_mark = torch.tensor(['?' in sample for sample in samples]).to(args.device)
                        is_valid = is_valid * has_question_mark
                    where_valid = is_valid.nonzero().reshape(-1).tolist()
                    for valid_idx in where_valid:
                        point_id = data_ids[valid_idx]
                        sample = pred_strs[valid_idx][sample_num]
                        if sample not in valid_continuations[point_id]:
                            valid_continuations[point_id].append(sample)
                iters += 1
                if iters >= max_iters:
                    break

        # additional filtering of valid_continuations
        max_samples_per_item = 5
        for k in list(valid_continuations.keys()):
            samples = valid_continuations[k]
            num_samples = len(samples)
            if num_samples > max_samples_per_item:
                if args.likelihood_filtering:
                    inputs = tokenizer( 
                            samples, 
                            return_tensors='pt',
                            padding=True,
                            max_length=args.max_seq_len, 
                            return_token_type_ids=True, 
                            truncation=True)
                    utils.move_kwargs_to_gpu(inputs)
                    where_padding = (inputs['input_ids'] == tokenizer.pad_token_id)
                    labels = inputs['input_ids'].masked_fill(where_padding, -100)
                    outputs = model(**inputs)
                    preds = outputs[0]
                    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
                    per_token_loss = loss_f(preds.reshape(-1, preds.size(-1)), labels.reshape(-1))
                    per_token_loss = per_token_loss.reshape(num_samples, -1)
                    num_tokens_per_sample = args.gen_max_seq_len - where_padding.sum(-1)
                    per_sample_loss = per_token_loss.sum(-1) / num_tokens_per_sample
                    best_idx = torch.argsort(per_sample_loss, descending=False)[:max_samples_per_item]
                    selected_samples = np.array(samples)[best_idx.cpu().numpy()].tolist()
                    valid_continuations[k] = selected_samples
                else:
                    valid_continuations[k] = np.random.choice(samples, size=max_samples_per_item, replace=False).tolist()

        # add continuations
        data_id_to_continuations.update(valid_continuations)
        save_data_dict.update({k : {
                'id' : k,
                'proposition' : all_propositions[k],
                'samples' : '\n'.join(v),
            } for k,v in valid_continuations.items()
        })
            
        # print last batch
        if args.print:
            print("-"*20 + f"\nPrinting {split_name.lower()} data:")
            print_data_names = [name for name in data_names if name not in ['main']]
            for i in range(min(args.num_print, batch_size)):
                text_data = batch['text_data']
                print(f" Model input {i}: {contexts[i]}")
                break_separated_samples = '\n\t' + '\n\t'.join([f"{j} : {pred_strs[i][j]}" for j in range(args.num_samples)])
                print(f"  Preds: {break_separated_samples}")
                valid_conts = valid_continuations[data_ids[i]]
                if len(valid_conts) > 0:
                    break_separated_conts = '\n\t' + '\n\t'.join([f"{j} : {valid_conts[j]}" for j in range(len(valid_conts))])
                    print(f"  Conts: {break_separated_conts}")
                else:
                    print(f"  No valid continuations")
                for data_name in print_data_names:
                    print(f"  {data_name.capitalize():12s}: {text_data[i][data_name]}")
            print("-"*20)
            # import pdb; pdb.set_trace()

    # print % with each at least n continuations
    print("Proportion with at least n continuations")
    for i in range(1, args.min_samples_per_prompt+1):
        prop = np.mean([len(conts) >= i for conts in data_id_to_continuations.values()])
        print(f" {i} : {100*prop:.2f}")

    # write continuations to file
    if split_name == 'dev':
        data = pd.DataFrame.from_dict(save_data_dict, "index")
        data.to_csv(f'data_utils/{args.dataset}_dev_neutral_samples.csv', index=False)
    # data_path = getattr(args, f"{split_name}_path")
    # new_data = []
    # with jsonlines.open(data_path) as file:
    #     for data_num, datapoint in enumerate(file):
    #         datapoint.update({
    #             f'independent_statements' : data_id_to_continuations[data_num]
    #         })
    #         new_data.append(datapoint)
    # with jsonlines.open(data_path, mode='w') as file:
    #     for point in new_data:
    #         file.write(point)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc. & debugging
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu id to use')
    parser.add_argument("--seed", default=0, type=int, help='')
    parser.add_argument("--fp16", default = False, type=str2bool, help = 'use fp16 as described here https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--small_data", '-s', action='store_true')
    parser.add_argument("--update_small_data", '-us', action='store_true')
    parser.add_argument("--print", default = True, type=str2bool, help = 'flag for printing things helpful for debugging / seeing whats happening')
    parser.add_argument("--num_print", '-np', default = 1, type=int, help = 'number of points to print in train_or_test.py')
    # model + model paths
    parser.add_argument("--model", default='gpt2', type=str, help='name of pretrained model')
    # paths/directories for data, cache, etc.
    parser.add_argument("--dataset", default='FEVER', type=str, help='')
    parser.add_argument("--data_dir", default='/checkpoint/peterhase/data', type=str, help='')
    parser.add_argument("--save_dir", default='/checkpoint/peterhase/saved_models', type=str, help='')
    parser.add_argument("--cache_dir", default='/checkpoint/peterhase/cached_models', type=str, help='')
    # gen args
    parser.add_argument("--sentence_to_prompt", default='first_three_words', type=str, help='how to get prompt from proposition')
    parser.add_argument("--gen_max_seq_len", default=30, type=int, help='')
    parser.add_argument("--temperature", default=.8, type=float, help='')
    parser.add_argument("--num_samples", default=5, type=int, help='beam search size for generative tasks')
    parser.add_argument("--beam_search_size", default=5, type=int, help='beam search size for generative tasks')
    parser.add_argument("--do_sample", default = True, type=str2bool, help = '')
    # filtering args
    parser.add_argument("--use_negated_propositions", default = True, type=str2bool, help = '')
    parser.add_argument("--neutral_prob_threshold", default=.9, type=float, help='')
    parser.add_argument("--max_filter_iters", default=100, type=int, help='')
    parser.add_argument("--min_samples_per_prompt", default=3, type=int, help='')
    parser.add_argument("--likelihood_filtering", default = True, type=str2bool, help = '')
    # still need these args for compatability with utils.load_data
    parser.add_argument("--max_seq_len", default=200, type=int, help='')
    parser.add_argument("--num_train_points", "-n", default = -1, type=int, help='if set to >0, use this many points for training')
    parser.add_argument("--num_eval_points", "-ne", default = -1, type=int, help='if set to >0, use this many points for training')
    parser.add_argument("--fit_to_alt_labels", default = False, type=str2bool, help = 'if true, use alt-labels for every data point to learn optimizer, not only using the true labels')
    parser.add_argument("--probing_style", default='model', choices=['model', 'cloze', 'seq2seq'], help='')
    parser.add_argument("--preprocess_data_when_loading", '-p', default = False, type=str2bool, help = '')
    parser.add_argument("--fit_to_dev_data", default = False, type=str2bool, help = 'used with --use_learned_optimizer')
    parser.add_argument("--train_batch_size", default=50, type=int, help='')
    parser.add_argument("--test_batch_size", default=50, type=int, help='')
    parser.add_argument("--use_learned_optimizer", default = False, type=str2bool, help = 'learn an optimizer for the problem')
    parser.add_argument("--update_beliefs", default = False, type=str2bool, help = 'use update_model function to update model beliefs (no learned opt)')
    parser.add_argument("--paraphrases_to_unique_points", default = False, type=str2bool, help = 'if data point has e.g. three paraphrases, convert it to three unique data points')
    parser.add_argument("--leapofthought_add_context", default = 'never', choices=['never', 'always', 'eval_only'], help = 'see utils.load_data')
    parser.add_argument("--leapofthought_main", default = 'implicit_rule', choices=['property', 'implicit_rule', 'both', 'hypothesis'], help = 'see utils.load_data')
    parser.add_argument("--update_eval_truthfully", default = False, type=str2bool, help = 'forces fit_to_wrong_points to true and fit_to_alt_labels to false on eval splits')
   
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    
    # parse + experiment checks
    args = parser.parse_args()

    # add dataset-specific parameters to args
    utils.add_dataset_config_to_args(args, args.dataset)
    args.base_experiment_name = ""
    args.num_samples = min(args.beam_search_size, args.num_samples) if not args.do_sample else args.num_samples

    # GPU + SEED setup
    n_gpu = torch.cuda.device_count()
    multi_gpu = (n_gpu > 1 and args.gpu == -1) # i.e. multiple gpus available and gpu choice not specified
    if multi_gpu: 
        device = torch.device("cuda") if args.gpu == -1 else torch.device(f'cuda:{args.gpu}')
        assert args.train_batch_size % n_gpu == 0, f"Train batch size will need to be allocated equally across {n_gpu} gpus, but {args.train_batch_size} cannot be"
        assert args.test_batch_size % n_gpu == 0, f"Eval batch size will need to be allocated equally across {n_gpu} gpus, but {args.test_batch_size} cannot be"
    else:
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(device)
    args.device = device
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load generative model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = model.to(args.device)
    num_opt_params = np.sum([np.prod(params.size()) for n, params in model.named_parameters() if params.requires_grad])
    print(f"Num model parameters: {num_opt_params/1e6:.2f}m")

    # add pad token if tokenizer does not have one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side='left'

    # load nli model
    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    nli_tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)
    nli_model = nli_model.to(args.device).eval()
    nli_special_tokens = [nli_tokenizer.bos_token_id, nli_tokenizer.sep_token_id, nli_tokenizer.eos_token_id, nli_tokenizer.pad_token_id] 

    # load data, optimizer, scheduler, and scaler
    print("Loading data...")
    start = time.time()
    train_dataloader, dev_dataloader, test_dataloader = utils.load_data(args, tokenizer, shuffle_loaders=False)
    print(f"\nTrain size: {len(train_dataloader.dataset)} | Dev size: {len(dev_dataloader.dataset)} | Test size: {len(test_dataloader.dataset)} ")
    for data_loader in [dev_dataloader]:
    # for data_loader in [train_dataloader, dev_dataloader, test_dataloader]:
        _ = write_continuations(args,
                    model=model, 
                    data_loader=data_loader, 
                    tokenizer=tokenizer,
                    nli_model=nli_model,
                    nli_tokenizer=nli_tokenizer)
    time_msg = utils.format_training_time(start=start, end=time.time())
    print(time_msg)