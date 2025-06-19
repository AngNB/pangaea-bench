import logging
import os
import time
from pathlib import Path
import math
import numpy as np
import sklearn.metrics
import wandb

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# added for visualization when regression
import random
import csv



class Evaluator:
    """
    Evaluator class for evaluating the models.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for experiment outputs.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate if Weights and Biases (wandb) is used for logging.
        logger (logging.Logger): Logger for logging information.
        classes (list): List of class names in the dataset.
        split (str): Dataset split (e.g., 'train', 'val', 'test').
        ignore_index (int): Index to ignore in the dataset.
        num_classes (int): Number of classes in the dataset.
        max_name_len (int): Maximum length of class names.
        wandb (module): Weights and Biases module for logging (if use_wandb is True).
    Methods:
        __init__(val_loader: DataLoader, exp_dir: str | Path, device: torch.device, use_wandb: bool) -> None:
            Initializes the Evaluator with the given parameters.
        evaluate(model: torch.nn.Module, model_name: str, model_ckpt_path: str | Path | None = None) -> None:
            Evaluates the given model. This method should be implemented by subclasses.
        __call__(model: torch.nn.Module) -> None:
            Calls the evaluator on the given model.
        compute_metrics() -> None:
            Computes evaluation metrics. This method should be implemented by subclasses.
        log_metrics(metrics: dict) -> None:
            Logs the computed metrics. This method should be implemented by subclasses.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
    ) -> None:
        self.rank = int(os.environ["RANK"])
        self.val_loader = val_loader
        self.logger = logging.getLogger()
        self.exp_dir = exp_dir
        self.device = device
        self.inference_mode = inference_mode
        self.sliding_inference_batch = sliding_inference_batch
        self.classes = self.val_loader.dataset.classes
        self.split = self.val_loader.dataset.split
        self.ignore_index = self.val_loader.dataset.ignore_index
        self.num_classes = len(self.classes)
        self.max_name_len = max([len(name) for name in self.classes])
        self.use_wandb = use_wandb
        
        # Compute valid class indices (excluding ignore index)
        self.valid_class_indices = [
            i for i in range(self.num_classes) if i != self.ignore_index
        ]
        self.valid_classes = [self.classes[i] for i in self.valid_class_indices]

    def evaluate(
            self,
            model: torch.nn.Module,
            model_name: str,
            model_ckpt_path: str | Path | None = None,
    ) -> None:
        raise NotImplementedError

    def __call__(self, model):
        pass

    def compute_metrics(self):
        pass

    def log_metrics(self, metrics):
        pass

    @staticmethod
    def sliding_inference(model, img, input_size, output_shape=None, stride=None, max_batch=None):
        b, c, t, height, width = img[list(img.keys())[0]].shape

        if stride is None:
            h = int(math.ceil(height / input_size))
            w = int(math.ceil(width / input_size))
        else:
            h = math.ceil((height - input_size) / stride) + 1
            w = math.ceil((width - input_size) / stride) + 1

        h_grid = torch.linspace(0, height - input_size, h).round().long()
        w_grid = torch.linspace(0, width - input_size, w).round().long()
        num_crops_per_img = h * w

        for k, v in img.items():
            img_crops = []
            for i in range(h):
                for j in range(w):
                    img_crops.append(v[:, :, :, h_grid[i]:h_grid[i] + input_size, w_grid[j]:w_grid[j] + input_size])
            img[k] = torch.cat(img_crops, dim=0)

        pred = []
        max_batch = max_batch if max_batch is not None else b * num_crops_per_img
        batch_num = int(math.ceil(b * num_crops_per_img / max_batch))
        for i in range(batch_num):
            img_ = {k: v[max_batch * i: min(max_batch * i + max_batch, b * num_crops_per_img)] for k, v in img.items()}
            pred_ = model.forward(img_, output_shape=(input_size, input_size))
            pred.append(pred_)
        pred = torch.cat(pred, dim=0)
        pred = pred.view(num_crops_per_img, b, -1, input_size, input_size).transpose(0, 1)

        merged_pred = torch.zeros((b, pred.shape[2], height, width), device=pred.device)
        pred_count = torch.zeros((b, height, width), dtype=torch.long, device=pred.device)
        for i in range(h):
            for j in range(w):
                merged_pred[:, :, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += pred[:, h * i + j]
                pred_count[:, h_grid[i]:h_grid[i] + input_size,
                w_grid[j]:w_grid[j] + input_size] += 1

        merged_pred = merged_pred / pred_count.unsqueeze(1)
        if output_shape is not None:
            merged_pred = F.interpolate(merged_pred, size=output_shape, mode="bilinear")

        return merged_pred


class SegEvaluator(Evaluator):
    """
    SegEvaluator is a class for evaluating segmentation models. It extends the Evaluator class and provides methods
    to evaluate a model, compute metrics, and log the results.
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on.
        use_wandb (bool): Flag to indicate whether to use Weights and Biases for logging.
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the given model on the validation dataset and computes metrics.
        __call__(model, model_name, model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        compute_metrics(confusion_matrix):
            Computes various metrics such as IoU, precision, recall, F1-score, mean IoU, mean F1-score, and mean accuracy
            from the given confusion matrix.
        log_metrics(metrics):
            Logs the computed metrics. If use_wandb is True, logs the metrics to Weights and Biases.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb)

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None):
        t = time.time()

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)
            model_name = os.path.basename(model_ckpt_path).split(".")[0]
            if "model" in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded {model_name} for evaluation")
        model.eval()

        tag = f"Evaluating {model_name} on {self.split} set"
        confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes), device=self.device
        )

        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):

            image, target = data["image"], data["target"]
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                max_batch=self.sliding_inference_batch)
            elif self.inference_mode == "whole":
                logits = model(image, output_shape=target.shape[-2:])
            else:
                raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
            if logits.shape[1] == 1:
                pred = (torch.sigmoid(logits) > 0.5).type(torch.int64).squeeze(dim=1)
            else:
                pred = torch.argmax(logits, dim=1)
            valid_mask = target != self.ignore_index
            pred, target = pred[valid_mask], target[valid_mask]
            count = torch.bincount(
                (pred * self.num_classes + target), minlength=self.num_classes ** 2
            )
            confusion_matrix += count.view(self.num_classes, self.num_classes)
            
            
            # ------------------------------ VISUALIZATION ------------------------------
            '''
            # SAVE SOME INTERMEDIATE RESULTS (ONLY PREDS THAT HAVE VALID VALUES ONLY AND IF WANDB IS ACTIVATED)
            if torch.all(torch.flatten(valid_mask)) and self.use_wandb and self.rank == 0:
                # JUST SAVE IMAGES IN BATCH=0 AND ONLY FOR THE EPOCHS: (5, 10, 30, 60) ---OR--- IMAGES AFTER EVERY 5 EPOCHS IN THE FINALIZING VALIDATION ROUND (it has to be multiple of 5, since Evaluation only for all 5 epochs)
                if (batch_idx == 0 and (model_name == "epoch 5" or model_name == "epoch 10" or model_name == "epoch 30" or model_name == "epoch 60")) or ((batch_idx == 0 or batch_idx == 5 or batch_idx == 10 or batch_idx == 15 or batch_idx == 30 ) and model_name == "checkpoint__best"):
                    # CLONE PREDICTED AND GROUND TRUTH TENSORS, SO THAT ANY CHANGES DO NOT AFFECT ORIGINAL TENSOR
                    pred_saved = pred.clone()
                    target_saved = target.clone()
                    
                    # TRANSFORM TO NUMPY
                    pred_np = pred_saved.cpu().numpy()
                    target_np = target_saved.cpu().numpy()

                    # TENSOR IS 1D, COMPUTE 2D DIM BASED ON TENSOR LENGHT
                    dim_2 = int(math.sqrt(len(pred_np)))
                    
                    # LOG INTO WANDB AFTER RESHAPING
                    img_pred = wandb.Image(pred_np.reshape((dim_2,dim_2)))
                    img_target = wandb.Image(target_np.reshape((dim_2,dim_2)))

                    # JUST SO THAT IMAGE DISPLAY ON WANDB IS SORTABLE
                    if len(str(batch_idx)) == 1:   
                        wandb.log({f"batch_00{batch_idx}_{model_name}_pred": img_pred})
                        wandb.log({f"batch_00{batch_idx}_{model_name}_target": img_target})
                    elif len(str(batch_idx)) == 2:   
                        wandb.log({f"batch_0{batch_idx}_{model_name}_pred": img_pred})
                        wandb.log({f"batch_0{batch_idx}_{model_name}_target": img_target})
                    else:   
                        wandb.log({f"batch_{batch_idx}_{model_name}_pred": img_pred})
                        wandb.log({f"batch_{batch_idx}_{model_name}_target": img_target})
            '''
            # ------------------------------ VISUALIZATION ------------------------------


        torch.distributed.all_reduce(
            confusion_matrix, op=torch.distributed.ReduceOp.SUM
        )
        metrics = self.compute_metrics(confusion_matrix.cpu())
        self.log_metrics(metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name, model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def compute_metrics(self, confusion_matrix):
        # Calculate IoU for each class
        intersection = torch.diag(confusion_matrix)
        union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
        iou = (intersection / (union + 1e-6)) * 100

        # Calculate precision and recall for each class
        precision = intersection / (confusion_matrix.sum(dim=0) + 1e-6) * 100
        recall = intersection / (confusion_matrix.sum(dim=1) + 1e-6) * 100

        # Calculate F1-score for each class
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        # Calculate mean IoU, mean F1-score, and mean Accuracy
        valid = self.valid_class_indices
            
        miou = iou[valid].mean().item() if valid else 0.0
        mf1 = f1[valid].mean().item() if valid else 0.0
        macc = (intersection.sum() / (confusion_matrix.sum() + 1e-6)).item() * 100

        # Convert metrics to CPU and to Python scalars
        iou = [iou[i].item() for i in valid]
        f1 = [f1[i].item() for i in valid]
        precision = [precision[i].item() for i in valid]
        recall = [recall[i].item() for i in valid]
        
        # Prepare the metrics dictionary
        metrics = {
            "IoU": iou,
            "mIoU": miou,
            "F1": f1,
            "mF1": mf1,
            "mAcc": macc,
            "Precision": precision,
            "Recall": recall,
        }

        return metrics

    def log_metrics(self, metrics):
        def format_metric(name, values, mean_value):
            header = f"------- {name} --------\n"
            metric_str = (
                    "\n".join(
                        c.ljust(self.max_name_len, " ") + "\t{:>7}".format("%.3f" % num)
                        for c, num in zip(self.valid_classes, values)
                    )
                    + "\n"
            )
            mean_str = (
                    "-------------------\n"
                    + "Mean".ljust(self.max_name_len, " ")
                    + "\t{:>7}".format("%.3f" % mean_value)
            )
            return header + metric_str + mean_str

        iou_str = format_metric("IoU", metrics["IoU"], metrics["mIoU"])
        f1_str = format_metric("F1-score", metrics["F1"], metrics["mF1"])

        precision_mean = sum(metrics["Precision"]) / len(metrics["Precision"]) if metrics["Precision"] else 0.0
        recall_mean = sum(metrics["Recall"]) / len(metrics["Recall"]) if metrics["Recall"] else 0.0

        precision_str = format_metric("Precision", metrics["Precision"], precision_mean)
        recall_str = format_metric("Recall", metrics["Recall"], recall_mean)

        macc_str = f"Mean Accuracy: {metrics['mAcc']:.3f} \n"

        self.logger.info(iou_str)
        self.logger.info(f1_str)
        self.logger.info(precision_str)
        self.logger.info(recall_str)
        self.logger.info(macc_str)

        if self.use_wandb and self.rank == 0:
            wandb.log(
                {
                    f"{self.split}_mIoU": metrics["mIoU"],
                    f"{self.split}_mF1": metrics["mF1"],
                    f"{self.split}_mAcc": metrics["mAcc"],
                    **{
                        f"{self.split}_IoU_{c}": v
                        for c, v in zip(self.valid_classes, metrics["IoU"])
                    },
                    **{
                        f"{self.split}_F1_{c}": v
                        for c, v in zip(self.valid_classes, metrics["F1"])
                    },
                    **{
                        f"{self.split}_Precision_{c}": v
                        for c, v in zip(self.valid_classes, metrics["Precision"])
                    },
                    **{
                        f"{self.split}_Recall_{c}": v
                        for c, v in zip(self.valid_classes, metrics["Recall"])
                    },
                }
            )


class RegEvaluator(Evaluator):
    """
    RegEvaluator is a subclass of Evaluator designed for regression tasks. It evaluates a given model on a validation dataset and computes metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
    Attributes:
        val_loader (DataLoader): DataLoader for the validation dataset.
        exp_dir (str | Path): Directory for saving experiment results.
        device (torch.device): Device to run the evaluation on (e.g., CPU or GPU).
        use_wandb (bool): Flag to indicate whether to log metrics to Weights and Biases (wandb).
    Methods:
        evaluate(model, model_name='model', model_ckpt_path=None):
            Evaluates the model on the validation dataset and computes MSE and RMSE.
        __call__(model, model_name='model', model_ckpt_path=None):
            Calls the evaluate method. This allows the object to be used as a function.
        log_metrics(metrics):
            Logs the computed metrics (MSE and RMSE) to the logger and optionally to wandb.
    """

    def __init__(
            self,
            val_loader: DataLoader,
            exp_dir: str | Path,
            device: torch.device,
            inference_mode: str = 'sliding',
            sliding_inference_batch: int = None,
            use_wandb: bool = False,
    ):
        super().__init__(val_loader, exp_dir, device, inference_mode, sliding_inference_batch, use_wandb)

    @torch.no_grad()
    def evaluate(self, model, model_name='model', model_ckpt_path=None, testing_bool=False):
        t = time.time()
        self.testing_bool=testing_bool

        if model_ckpt_path is not None:
            model_dict = torch.load(model_ckpt_path, map_location=self.device, weights_only=False)  # added 'weights_only=False'
            model_name = os.path.basename(model_ckpt_path).split('.')[0]
            if 'model' in model_dict:
                model.module.load_state_dict(model_dict["model"])
            else:
                model.module.load_state_dict(model_dict)

            self.logger.info(f"Loaded model from {model_ckpt_path} for evaluation")

        model.eval()

        tag = f'Evaluating {model_name} on {self.split} set'

        mse = torch.zeros(1, device=self.device)
        mean_absolute_error = torch.zeros(1, device=self.device)
        mean_error = torch.zeros(1, device=self.device)


        for batch_idx, data in enumerate(tqdm(self.val_loader, desc=tag)):
            image, target = data['image'], data['target']
            image = {k: v.to(self.device) for k, v in image.items()}
            target = target.to(self.device)

            if self.inference_mode == "sliding":
                input_size = model.module.encoder.input_size
                logits = self.sliding_inference(model, image, input_size, output_shape=target.shape[-2:],
                                                max_batch=self.sliding_inference_batch).squeeze(dim=1)
            elif self.inference_mode == "whole":
                logits = model(image, output_shape=target.shape[-2:]).squeeze(dim=1)
            else:
                raise NotImplementedError((f"Inference mode {self.inference_mode} is not implemented."))
            
            # for sparse mse, get central pixel assuming image height = image width and no rescaling
            pxl = int(logits.shape[-1]/2)
            
            assert logits.shape == target.shape, f"Shape error in evaluator.py: logits.shape = {logits.shape} != target.shape {target.shape}"

            mse += F.mse_loss(logits[:,pxl,pxl],target[:,pxl,pxl])

            # calculate extra metrics for logging
            mse_to_print = F.mse_loss(logits[:,pxl,pxl],target[:,pxl,pxl])

            error_per_batch = (logits[:,pxl,pxl] - target[:,pxl,pxl])
            mean_error += torch.mean(error_per_batch)

            absolute_error = (torch.abs(error_per_batch))
            mean_absolute_error += torch.mean(absolute_error)

            if self.testing_bool:
                path_to_csv = os.path.join(self.exp_dir, f"test_metrics/metrics.csv")
                with open(path_to_csv, 'a') as file:
                    writer = csv.writer(file)
                    for i in range(target.shape[0]):
                        writer.writerow([batch_idx, i, target[i,pxl,pxl].item(), logits[i,pxl,pxl].item(), mse_to_print.item()])
                    file.close()

            # ------------------------------ VISUALIZATION ------------------------------
            # save predicted values in testing phase, when best model with name "chechpoint__best" is called to evaluate on the test dataset (only when use_wandb=true)
            if self.use_wandb and self.rank == 0 and (batch_idx % 120 == 0) and (model_name == "epoch 0" or model_name == "epoch 10" or model_name == "checkpoint__best"):
                    # CLONE PREDICTED AND GROUND TRUTH TENSORS, SO THAT ANY CHANGES DO NOT AFFECT ORIGINAL TENSOR
                    pred_saved = logits.clone()
                    target_saved = target.clone()

                    # create index for random image in the batch
                    max_batch_size = logits.shape[0]
                    i = random.randint(0, max_batch_size-1)

                    # ONLY SHOW CENTRAL PIXEL -> MASK THE REST
                    logits_pxl = pred_saved[i,pxl,pxl]
                    target_pxl = target_saved[i,pxl,pxl]
                    
                    pred_saved_masked = torch.zeros((pred_saved[i,:,:].shape))
                    target_saved_masked = torch.zeros((target_saved[i,:,:].shape))

                    pred_saved_masked[pxl,pxl] = logits_pxl
                    target_saved_masked[pxl,pxl] = target_pxl

                    pred_saved_masked = pred_saved_masked[pxl-3:pxl+4,pxl-3:pxl+4]
                    target_saved_masked = target_saved_masked[pxl-3:pxl+4,pxl-3:pxl+4]

                    # TRANSFORM TO NUMPY
                    pred_np = pred_saved.cpu().numpy()
                    target_np = target_saved.cpu().numpy()
                    pred_saved_masked_np = pred_saved_masked.cpu().numpy()
                    target_saved_masked_np = target_saved_masked.cpu().numpy()
                    
                    # LOG INTO WANDB AFTER SLICING (TENSOR IS 3D, TAKE ANY IMAGE)
                    img_pred = wandb.Image(pred_np[i,:,:], caption=f"prediction (value: {logits_pxl:.3f})")
                    img_target = wandb.Image(target_np[i,:,:], caption=f"ground truth (value: {target_pxl:.3f})")

                    img_pred_masked = wandb.Image(pred_saved_masked_np, caption=f"prediction (masked, zoomed) (value: {logits_pxl:.3f})")
                    img_target_masked = wandb.Image(target_saved_masked_np, caption=f"ground truth (masked, zoomed) (value: {target_pxl:.3f})")

                    wandb.log({f"batch_{batch_idx}_{model_name}_#{i}: MSE = {mse_to_print:.3f}": (img_pred, img_target, img_pred_masked, img_target_masked)})

            # ------------------------------ VISUALIZATION ------------------------------


        torch.distributed.all_reduce(mse, op=torch.distributed.ReduceOp.SUM)
        mse = mse / len(self.val_loader)
        mean_absolute_error = mean_absolute_error/ len(self.val_loader)
        mean_error = mean_error / len(self.val_loader)

        other_metrics = {"MAE": mean_absolute_error.item(), "ME": mean_error.item()}

        metrics = {"MSE": mse.item(), "RMSE": torch.sqrt(mse).item()}
        self.log_metrics(metrics, other_metrics)

        used_time = time.time() - t

        return metrics, used_time

    @torch.no_grad()
    def __call__(self, model, model_name='model', model_ckpt_path=None):
        return self.evaluate(model, model_name, model_ckpt_path)

    def log_metrics(self, metrics, other_metrics):
        header = "------- MSE and RMSE --------\n"
        mse = "-------------------\n" + 'MSE \t{:>7}'.format('%.3f' % metrics['MSE']) + '\n'
        rmse = "-------------------\n" + 'RMSE \t{:>7}'.format('%.3f' % metrics['RMSE'])
        self.logger.info(header + mse + rmse)

        if self.use_wandb and self.rank == 0:
            wandb.log({f"{self.split}_MSE_(mean_over_test_batches)": metrics["MSE"], f"{self.split}_RMSE_(mean_over_test_batches)": metrics["RMSE"], f"{self.split}_MAE_(mean_over_test_batches)": other_metrics["MAE"], f"{self.split}_ME_(mean_over_test_batches)": other_metrics["ME"]})
