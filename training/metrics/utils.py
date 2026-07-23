from sklearn import metrics
import numpy as np


def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # low-FPR operating points (fake = positive): TPR while wrongly flagging
    # at most 1% / 5% of real samples, the real-side view (fraction of reals
    # kept when 95% of fakes must be caught), and the standardized partial
    # AUC of the FPR<=5% regime. roc_curve returns fpr sorted ascending with
    # fpr[0]=0 and tpr[-1]=1, so the index lookups below are always valid.
    def _tpr_at_fpr(target):
        return float(tpr[np.searchsorted(fpr, target, side='right') - 1])
    tpr_fpr1 = _tpr_at_fpr(0.01)
    tpr_fpr5 = _tpr_at_fpr(0.05)
    tnr_tpr95 = float(1.0 - fpr[np.searchsorted(tpr, 0.95, side='left')])
    try:
        pauc5 = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.05)
    except ValueError:
        pauc5 = float('nan')
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        # img_names may be longer than y_pred when the test loader uses
        # drop_last=True (e.g. DeepFakeDetection); the loader is unshuffled,
        # so the first len(y_pred) names align with the predictions.
        img_names = img_names[:len(y_pred)]
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap,
            'tpr@1%fpr': tpr_fpr1, 'tpr@5%fpr': tpr_fpr5,
            'tnr@95%tpr': tnr_tpr95, 'pauc@5%fpr': pauc5,
            'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
