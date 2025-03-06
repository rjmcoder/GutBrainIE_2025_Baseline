import os
import os.path
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support, classification_report
import pandas as pd

# rel2id = json.load(open('meta/rel2id.json', 'r'))
# id2rel = {value: key for key, value in rel2id.items()}

def get_title2pred(pred: list) -> dict:
    '''
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']  
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    '''
    
    title2pred = {}

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])
        
        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(p["score"], title2pred[p["title"]][curr])
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            title2pred[p["title"]] = {curr: p["score"]}
    return title2pred


"""def get_title2gt(features: dict) -> dict:
    '''
    Convert ground-truth labels to dictionary.
    Input:
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :title2gt: dictionary with (key, value) = (title, [gold_triples])
    '''
    title2gt = {}
    for f in features:
        title = f["title"]
        title2gt[title] = []
        for idx, p in enumerate(f["hts"]): 
            h,t = p
            label = np.array(f['labels'][idx])
            rs = np.nonzero(label[1:])[0] + 1 # + 1 for no-label
            title2gt[title].extend([(h,t,id2rel[r]) for r in rs])
            
    return title2gt"""

def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    '''
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand. 
    '''
    
    sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
    precs, recalls = [], []
    
    for pred in sorted_pred:     
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred)  # Precision
        recalls.append(correct / num_gt)  # Recall                             

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print('Best thresh', thresh, '\tbest F1', f1)
    return thresh, sorted_pred[:f1_pos + 1]

"""
def merge_results(pred: list, pred_pseudo: list, features: list, thresh: float = None):
    '''
    Merge relation predictions from the original document and psuedo documents.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :pred_pseudo: list of dictionaries, each dictionary entry is a predicted relation triple from pseudo documents. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :thresh: threshold for selecting predictions.
    Output:
        :merged_res: list of merged relation predictions. Each relation prediction is a dictionay with keys (title, h_idx, t_idx, r).
        :thresh: threshold of selecting relation predictions.
    '''
    
    title2pred = get_title2pred(pred)
    title2pred_pseudo = get_title2pred(pred_pseudo)

    title2gt = get_title2gt(features)
    num_gt = sum([len(title2gt[t]) for t in title2gt])

    titles = list(title2pred.keys())
    cand = []
    merged_res = []
    correct, num_pred = 0, 0

    for t in titles:
        rels = title2pred[t]
        rels_pseudo = title2pred_pseudo[t] if t in title2pred_pseudo else {}

        union = set(rels.keys()) | set(rels_pseudo.keys())
        for r in union:
            if r in rels and r in rels_pseudo: # add those into predictions
                if rels[r] > 0 and rels_pseudo[r] > 0:
                    merged_res.append({'title':t, 'h_idx':r[0], 't_idx':r[1], 'r': r[2]})
                    num_pred += 1
                    correct += r in title2gt[t]
                    continue
                score = rels[r] + rels_pseudo[r]
            elif r in rels: # -10 for penalty
                score = rels[r] - 10
            elif r in rels_pseudo:
                score = rels_pseudo[r] - 10
            cand.append((r in title2gt[t], score, t, r[0], r[1], r[2]))
    
    if thresh != None:
        sorted_pred = sorted(cand, key=lambda x:x[1], reverse=True)
        last = min(filter(lambda x: x[1] > thresh, sorted_pred))
        until = sorted_pred.index(last)
        cand = sorted_pred[:until + 1]
        merged_res.extend([{'title':r[2], 'h_idx':r[3], 't_idx':r[4], 'r': r[5]} for r in cand])
        return merged_res, thresh

    if cand != []:
        thresh, cand = select_thresh(cand, num_gt, correct, num_pred)
        merged_res.extend([{'title':r[2], 'h_idx':r[3], 't_idx':r[4], 'r': r[5]} for r in cand])

    return merged_res, thresh"""


def extract_relative_score(scores: list, topks: list) -> list:
    '''
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    '''
    
    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks==0)].item()     
    
    scores -= na_score

    return scores
"""
def to_official(preds: list, features: list, evi_preds: list = [], scores: list = [], topks: list = []):
    '''
    Convert the predictions to official format for evaluating.
    Input:
        :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :evi_preds: list of the evidence prediction corresponding to each relation triple prediction.
        :scores: list of scores of topk relation labels for each entity pair.
        :topks: list of topk relation labels for each entity pair.
    Output:
        :official_res: official results used for evaluation.
        :res: topk results to be dumped into file, which can be further used during fushion.
    '''
    
    
    h_idx, t_idx, title, sents = [], [], [], []

    for f in features:
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]

        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        sents += [len(f["sent_pos"])] * len(hts)

    official_res = []
    res = []

    for i in tqdm(range(preds.shape[0]), desc="preds"): # for each entity pair
        if scores != []:
            score = extract_relative_score(scores[i], topks[i]) 
            pred = topks[i]
        else:
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()
        
        for p in pred: # for each predicted relation label (topk)
            curr_result = {
                    'title': title[i],
                    'h_idx': h_idx[i],
                    't_idx': t_idx[i],
                    'r': id2rel[p],
                }
            if evi_preds != []:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist() 
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if scores != []:
                curr_result["score"] = score[np.where(topks[i] == p)].item()
            if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                official_res.append(curr_result)
            res.append(curr_result)

    return official_res, res"""


def gen_train_facts(data_file_name, truth_dir):
    
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate_per_rel(logger, tmp, path, train_file="train_annotated.json", dev_file="dev.json"):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))
        
    std = {}
    parsed_relations_truth = []
    tot_evidences = {}
    tot_relations = {}
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        if 'labels' not in x:   # official test set from DocRED
            continue
        
        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            if r in parsed_relations_truth:
                tot_evidences[r] += len(label['evidence'])
                tot_relations[r] += 1
            else:
                parsed_relations_truth.append(r)
                tot_evidences[r] = len(label['evidence'])
                tot_relations[r] = 1

            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            # tot_evidences += len(label['evidence'])


    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    num_predicted_relations = {tmp[0]["r"]: 1}

    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
            try:
                num_predicted_relations[tmp[i]["r"]] += 1
            except KeyError:
                num_predicted_relations[tmp[i]["r"]] = 1

    correct_re = {}
    correct_evidence = {}
    pred_evi = {}

    correct_in_train_annotated = {}
    correct_in_train_distant = {}

    parsed_relations = []

    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']

        if r not in parsed_relations:
            parsed_relations.append(r)
            correct_re[r] = 0
            correct_evidence[r] = 0
            pred_evi[r] = 0
            correct_in_train_annotated[r] = 0
            correct_in_train_distant[r] = 0

        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x : #and (title, h_idx, t_idx) in std:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi[r] += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re[r] += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence[r] += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated[r] += 1
            if in_train_distant:
                correct_in_train_distant[r] += 1

    output_dict = {}
    for r in parsed_relations_truth:
        if r not in parsed_relations:
            correct_re[r] = 0
            correct_evidence[r] = 0
            pred_evi[r] = 0
            correct_in_train_annotated[r] = 0
            correct_in_train_distant[r] = 0
            num_predicted_relations[r] = 0

        re_p = 1.0 * correct_re[r] / num_predicted_relations[r] if num_predicted_relations[r] != 0 else 0
        re_r = 1.0 * correct_re[r] / tot_relations[r] if tot_relations[r] != 0 else 0
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence[r] / pred_evi[r] if pred_evi[r] > 0 else 0
        evi_r = 1.0 * correct_evidence[r] / tot_evidences[r] if tot_evidences[r] > 0 else 0

        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re[r] - correct_in_train_annotated[r]) / (num_predicted_relations[r] - correct_in_train_annotated[r] + 1e-5)
        re_p_ignore_train = 1.0 * (correct_re[r] - correct_in_train_distant[r]) / (num_predicted_relations[r] - correct_in_train_distant[r] + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        output_dict[r] = [re_p, re_r, re_f1, re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated, evi_p, evi_r, evi_f1]

    return output_dict
    # return [re_p, re_r, re_f1], [evi_p, evi_r, evi_f1], [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], [re_p_ignore_train, re_r, re_f1_ignore_train]


def official_evaluate_long_tail(logger, tmp, path, train_file="train_annotated.json", dev_file="dev.json"):
    '''
        Evaluation of long-tailed relations (macro@K, with K=100; 200; 500; all)
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    # fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    # fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    parsed_relations_truth = []
    tot_rel = {"100": 0, "200": 0, "500": 0}
    tot_relations = {}
    titleset = set([])

    title2vectexSet = {}

    rels_freq = json.load(open(os.path.join(path, "meta/relations_frequency.json")))

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        if 'labels' not in x:  # official test set from DocRED
            continue

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            if r in parsed_relations_truth:
                # tot_evidences[r] += len(label['evidence'])
                tot_relations[r] += 1
            else:
                parsed_relations_truth.append(r)
                # tot_evidences[r] = len(label['evidence'])
                tot_relations[r] = 1
            if r in rels_freq["100"]:
                tot_rel["100"] += 1
            if r in rels_freq["200"]:
                tot_rel["200"] += 1
            if r in rels_freq["500"]:
                tot_rel["500"] += 1

            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            # tot_evidences += len(label['evidence'])

    tot_rel["ALL"] = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    num_predicted_relations = {tmp[0]["r"]: 1}
    num_predicted = {"100": 0, "200": 0, "500": 0}

    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
            try:
                num_predicted_relations[tmp[i]["r"]] += 1
            except KeyError:
                num_predicted_relations[tmp[i]["r"]] = 1
            if tmp[i]["r"] in rels_freq["100"]:
                num_predicted["100"] += 1
            if tmp[i]["r"] in rels_freq["200"]:
                num_predicted["200"] += 1
            if tmp[i]["r"] in rels_freq["500"]:
                num_predicted["500"] += 1

    num_predicted["ALL"] = len(submission_answer)
    correct_re = {}
    corrected = {"ALL": 0, "100": 0, "200": 0, "500": 0}
    correct_evidence = {}
    pred_evi = {}

    correct_in_train_annotated = {}
    correct_in_train_distant = {}

    parsed_relations = []

    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']

        if r not in parsed_relations:
            parsed_relations.append(r)
            correct_re[r] = 0

        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        """if 'evidence' in x:  # and (title, h_idx, t_idx) in std:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi[r] += len(evi)"""

        if (title, r, h_idx, t_idx) in std:
            correct_re[r] += 1
            corrected["ALL"] += 1
            if r in rels_freq["100"]:
                corrected["100"] += 1
            if r in rels_freq["200"]:
                corrected["200"] += 1
            if r in rels_freq["500"]:
                corrected["500"] += 1


    f1_scores, prec_scores, rec_scores = {}, {}, {}
    macro_f1, prec, rec = 0, 0, 0
    # logger.logger.info(f"parsed_relations_truth (n. {len(parsed_relations_truth)}): {parsed_relations_truth}")
    for r in parsed_relations_truth:
        if r not in parsed_relations:
            correct_re[r] = 0
            num_predicted_relations[r] = 0

        re_p = 1.0 * correct_re[r] / num_predicted_relations[r] if num_predicted_relations[r] != 0 else 0
        re_r = 1.0 * correct_re[r] / tot_relations[r] if tot_relations[r] != 0 else 0
        prec += re_p
        prec_scores[r] = re_p
        rec += re_r
        rec_scores[r] = re_r
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        f1_scores[r] = re_f1
        macro_f1 += re_f1

    output_dict = {}

    # micro-averaged
    """micro_prec = 1.0 * corrected["all"] / len(submission_answer)
    micro_rec = 1.0 * corrected["all"] / tot_rel if tot_rel != 0 else 0
    if micro_prec + micro_rec == 0:
        micro_f1 = 0
    else:
        micro_f1 = 2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
    output_dict["micro@ALL"] = [micro_f1, micro_prec, micro_rec]"""

    for k in corrected.keys():
        micro_prec = 1.0 * corrected[k] / num_predicted[k] if num_predicted[k] != 0 else 0
        micro_rec = 1.0 * corrected[k] / tot_rel[k] if tot_rel[k] != 0 else 0
        if micro_prec + micro_rec == 0:
            micro_f1 = 0
        else:
            micro_f1 = 2.0 * micro_prec * micro_rec / (micro_prec + micro_rec)
        output_dict[f"micro@{k}"] = [micro_f1, micro_prec, micro_rec]

    # macro-averaged
    macro_f1 = macro_f1/len(parsed_relations)
    macro_prec = prec/len(parsed_relations)
    macro_rec = rec/len(parsed_relations)
    output_dict["macro@ALL"] = [macro_f1, macro_prec, macro_rec]

    f1_100, f1_200, f1_500 = 0, 0, 0
    prec_100, prec_200, prec_500 = 0, 0, 0
    rec_100, rec_200, rec_500 = 0, 0, 0

    logger.logger.info(f"*** NUMBER OF CONSIDERED RELATIONS: {len(f1_scores.keys())} ***")

    for rel in rels_freq["100"]:
        # TODO: remove if when performing full experiment
        if rel in f1_scores.keys():
            f1_100 += f1_scores[rel]
            rec_100 += rec_scores[rel]
            prec_100 += prec_scores[rel]
    output_dict["macro@100"] = [f1_100/len(rels_freq["100"]), prec_100/len(rels_freq["100"]), rec_100/len(rels_freq["100"])]

    for rel in rels_freq["200"]:
        # TODO: remove if when performing full experiment
        if rel in f1_scores.keys():
            f1_200 += f1_scores[rel]
            rec_200 += rec_scores[rel]
            prec_200 += prec_scores[rel]
    output_dict["macro@200"] = [f1_200 / len(rels_freq["200"]), prec_200 / len(rels_freq["200"]),
                          rec_200 / len(rels_freq["200"])]

    for rel in rels_freq["500"]:
        # TODO: remove if when performing full experiment
        if rel in f1_scores.keys():
            f1_500 += f1_scores[rel]
            rec_500 += rec_scores[rel]
            prec_500 += prec_scores[rel]
    output_dict["macro@500"] = [f1_500 / len(rels_freq["500"]), prec_500 / len(rels_freq["500"]),
                          rec_500 / len(rels_freq["500"])]


    return output_dict
    # return [re_p, re_r, re_f1], [evi_p, evi_r, evi_f1], [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], [re_p_ignore_train, re_r, re_f1_ignore_train]


def official_evaluate_sklearn(logger, y_true, y_pred, path):
    '''
        Evaluation using sklearn
    '''

    rel2id = json.load(open(os.path.join(path, "meta/rel2id.json")))
    id2rel = {value: key for key, value in rel2id.items()}
    rels_freq = json.load(open(os.path.join(path, "meta/relations_frequency.json")))

    y_true_500 = []
    y_pred_500 = []
    y_true_200 = []
    y_pred_200 = []
    y_true_100 = []
    y_pred_100 = []

    logger.logger.info(f"length of preds: {len(y_pred)}")
    logger.logger.info(f"length of true: {len(y_true)}")

    # Apply condition to filter rows: first column == 1 (NA relation) and sum of the row == 1 (only NA relation selected)
    # y_pred = [row for row in y_pred if not (row[0] == 1 and sum(row) == 1)]
    # y_true = [row for row in y_true if not (row[0] == 1 and sum(row) == 1)]
    """y_pred_filtered = []
    y_true_filtered = []
    for i in range(len(y_pred)):
        if y_pred[i][0] == 1 and sum(y_pred[i] == 1):
            # skip row
            continue
        else:
            y_pred_filtered.append(y_pred[i])
            y_true_filtered.append(y_true[i])

    y_pred = y_pred_filtered
    y_true = y_true_filtered

    logger.logger.info(f"length of preds: {len(y_pred)}")
    logger.logger.info(f"length of true: {len(y_true)}")"""
    y_true_filtered = []
    y_pred_filtered = []
    """
    REMOVING NA rows does not work
    if eval_mode == "sklearn-test":
        for i in range(len(y_true)):
            if y_true[i][0] == 1 and sum(y_true[i]) == 1:
                continue
            else:
                y_true_filtered.append(y_true[i])
                y_pred_filtered.append(y_pred[i])
        y_pred = y_pred_filtered
        y_true = y_true_filtered
    """

    for j in range(len(y_pred)):
        pred_pair_100 = []
        pred_pair_200 = []
        pred_pair_500 = []
        true_pair_100 = []
        true_pair_200 = []
        true_pair_500 = []
        for i in range(len(y_pred[j])):
            if id2rel[i] in rels_freq["100"]:
                pred_pair_100.append(y_pred[j][i])
                true_pair_100.append(y_true[j][i])
            if id2rel[i] in rels_freq["200"]:
                pred_pair_200.append(y_pred[j][i])
                true_pair_200.append(y_true[j][i])
            if id2rel[i] in rels_freq["500"]:
                pred_pair_500.append(y_pred[j][i])
                true_pair_500.append(y_true[j][i])

        if len(pred_pair_100) > 0 and len(pred_pair_100) == len(true_pair_100):
            y_pred_100.append(pred_pair_100)
            y_true_100.append(true_pair_100)
        elif len(pred_pair_100) != len(true_pair_100):
            raise ValueError(f"Length of true and predicted pair @100 does not match! pred_pair: {len(pred_pair_100)}; true_pair: {len(true_pair_100)}")

        if len(pred_pair_200) > 0 and len(pred_pair_200) == len(true_pair_200):
            y_pred_200.append(pred_pair_200)
            y_true_200.append(true_pair_200)
        elif len(pred_pair_200) != len(true_pair_200):
            raise ValueError(f"Length of true and predicted pair @200 does not match! pred_pair: {len(pred_pair_200)}; true_pair: {len(true_pair_200)}")

        if len(pred_pair_500) > 0 and len(pred_pair_500) == len(true_pair_500):
            y_pred_500.append(pred_pair_500)
            y_true_500.append(true_pair_500)
        elif len(pred_pair_500) != len(true_pair_500):
            raise ValueError(f"Length of true and predicted pair @500 does not match! pred_pair: {len(pred_pair_500)}; true_pair: {len(true_pair_500)}")

    # Exclude NA column
    y_true = np.delete(y_true, 0, axis=1)
    y_pred = np.delete(y_pred, 0, axis=1)

    output_dict = {}
    p_r_f_s = precision_recall_fscore_support(y_true, y_pred, average='micro')
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    logger.logger.info(f"micro {p_r_f_s}")
    logger.logger.info(f"Type: {type(p_r_f_s)}")
    logger.logger.info(f"Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")
    precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    logger.logger.info(f"Weighted Precision: {precision}, Recall: {recall}, F-score: {fscore}, Support: {support}")
    pred2cutoff = {"ALL": [y_true, y_pred], "500": [y_true_500, y_pred_500], "200": [y_true_200, y_pred_200],
                   "100": [y_true_100, y_pred_100]}
    for avg in ["micro", "macro", "weighted"]:
        for cutoff in ["ALL", "500", "200", "100"]:
            y = pred2cutoff[cutoff]
            output_dict[f"{avg}@{cutoff}"] = [f1_score(y[0], y[1], average=avg, zero_division=np.nan),
                                        precision_score(y[0], y[1], average=avg, zero_division=np.nan),
                                        recall_score(y[0], y[1], average=avg, zero_division=np.nan)]
    # rel2id = json.load(open(os.path.join(path, "meta/rel2id.json")))
    logger.logger.info(f"Number of labels: {len(id2rel.keys())}")
    logger.logger.info(f"Labels: {id2rel.keys()}")
    new_labels = list(id2rel.keys())
    new_labels.remove(0)
    new_labels = sorted(new_labels)
    logger.logger.info(f"Labels without 0: {new_labels}")
    logger.logger.info(f"Number of labels: {len(new_labels)}")
    logger.logger.info(f"length of preds: {len(y_pred[0])}")
    logger.logger.info(f"length of true: {len(y_true[0])}")
    labels = [id2rel[i] for i in new_labels]
    logger.logger.info(f"Number of labels: {len(labels)}")
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    df = pd.DataFrame(report).transpose()

    return output_dict, df