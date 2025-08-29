import os, re, torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader, TensorDataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple
from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_curve,
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score, 
    auc
)

def preprocess_text(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub(r'@\w+|\bhttps?://\S+\b', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def load_dataset(datatrain, datatest) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(datatrain)
    y = pd.read_csv(datatest)
    return X, y

def dataloader_NN(X_train, X_test, y_train, y_test) -> Tuple[DataLoader, DataLoader]:
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()
    X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_dense, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).squeeze() 
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).squeeze()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

def get_metrics(model_name: str, y_test: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    return {
        'Model': model_name,
        
        'Accuracy': round(accuracy_score(y_test, y_pred), 3),
        'Precision': round(precision_score(y_test, y_pred), 3),
        'Recall': round(recall_score(y_test, y_pred), 3),
        'F1': round(f1_score(y_test, y_pred), 3),

        'ROC AUC': round(roc_auc_score(y_test, y_proba), 3),
        'fpr': fpr,
        'tpr': tpr,

        'pr_auc': round(auc(recall, precision),3),
        'pr_prec': precision,
        'pr_rec': recall,
    }

def create_folder(folder_path) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def save_metrics(metrics: list, cfg, name_model = 'metrics') -> None:
    create_folder(cfg['metrics_folder'])
    metrics = pd.DataFrame(metrics)
    metrics.drop(columns=['fpr', 'tpr', 'pr_prec', 'pr_rec'], inplace=True)
    metrics.to_csv(f'{cfg["metrics_folder"]}/{name_model}.csv', index=False)

def save_metrics_nn(metrics: list, cfg, name_model = 'metrics_nn') -> None:
    create_folder(cfg['metrics_folder'])
    metrics = pd.DataFrame(metrics, index=[0])
    metrics.to_csv(f'{cfg["metrics_folder"]}/{name_model}.csv', index=False)

class Plotter:
    def __init__(self, cfg, model_name: str, type_metrics: str = 'test'):
        self.cfg = cfg
        self.model_name = model_name
        self.type_metrics = type_metrics
        create_folder(cfg['images_folder'])

    def plot_roc(self, metrics: list, type_metrics: str = 'test') -> Figure:
        plt.figure(figsize=(8, 8))
        plt.title('ROC AUC')
        plt.plot([0, 1], [0, 1], '--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        for model_metrics in metrics:
            model_name = model_metrics['Model']
            
            roc_auc = model_metrics['ROC AUC']
            fpr = model_metrics['fpr']
            tpr = model_metrics['tpr']

            plt.plot(fpr, tpr, label=f"{model_name} (ROC AUC = {roc_auc:.2f})")
            plt.legend()

        plt.savefig(f'{self.cfg["images_folder"]}/roc_curve_{type_metrics}.png')
        plt.close()

    def plot_pr(self, metrics: list, type_metrics: str = 'test') -> Figure:
        plt.figure(figsize=(8, 8))
        plt.title('PR AUC')
        plt.plot([0, 1], [1, 0], '--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        for model_metrics in metrics:
            model_name = model_metrics['Model']

            pr_auc = model_metrics['pr_auc']
            precision = model_metrics['pr_prec']
            recall = model_metrics['pr_rec']

            plt.plot(precision, recall, label=f"{model_name} (PR AUC = {pr_auc:.2f})")
            plt.legend()

        plt.savefig(f'{self.cfg["images_folder"]}/pr_curve_{type_metrics}.png')
        plt.close()

    def plot_cm(self, y_test: pd.Series, y_pred: pd.Series) -> Figure:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues')
        plt.title(f'{self.model_name} Confusion Matrix')
        plt.savefig(f'{self.cfg["images_folder"]}/{self.model_name}_cm.png')
        plt.close()

def plot_loss_history(loss_history: List[float], save_path: str = 'loss_history.png') -> None:

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2, label='Training Loss')
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'ro', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.annotate(f'{loss_history[-1]:.4f}', 
                xy=(len(loss_history), loss_history[-1]),
                xytext=(len(loss_history)+0.5, loss_history[-1]),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_NN(preds: torch.Tensor, labels: torch.Tensor, model_name: str = 'Model', save_path: str = 'confusion_matrix.png') -> float:

    y_true = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
    y_pred = preds.numpy() if torch.is_tensor(preds) else np.array(preds)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    return cm

def evaluate(model, test_loader, threshold: float = 0.5, model_name: str = 'Модель 1') -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch['image'] if 'image' in batch else batch[0]
                labels = batch['label'] if 'label' in batch else batch[1]
                        
            outputs = model(inputs)
            
            if outputs.dim() == 2 and outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(outputs).squeeze()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend((probs >= threshold).cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError as e:
        roc_auc = 0.0  
        print(f"Ошибка вычисления ROC-AUC: {e}")
    
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError as e:
        pr_auc = 0.0  
        print(f"Ошибка вычисления PR-AUC: {e}")
        
    metrics = {
        'Model': model_name,
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3),
        'roc_auc': round(roc_auc, 3),
        'pr_auc': round(pr_auc, 3),
    }
    
    return metrics, y_pred, y_probs, y_true

def plot_roc_curve_NN(y_probs: np.ndarray, y_true: np.ndarray, model_name: str = "Model", save_path: str ='roc_curve.png') -> plt.Figure:
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
        
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.title('PR AUC')
    plt.plot([0, 1], [1, 0], '--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
        
    return roc_auc, fpr, tpr

def plot_pr_curve_NN(y_probs: np.ndarray, y_true: np.ndarray, 
                    model_name: str = "Model", 
                    save_path: str = 'pr_curve.png') -> Tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_probs)
    
    positive_ratio = np.mean(y_true)
    random_precision = positive_ratio
    
    plt.figure(figsize=(8, 8))
    
    # Правильный порядок: recall по X, precision по Y
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'{model_name} (AP = {avg_precision:.3f}, AUC = {pr_auc:.3f})')
    
    # Линия случайного классификатора (горизонтальная)
    plt.axhline(y=random_precision, color='navy', lw=2, linestyle='--', 
                label=f'Random (AP = {random_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
    plt.legend(loc="upper right")  # Изменено на upper right для лучшей видимости
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
        
    return precision, recall, pr_auc
    