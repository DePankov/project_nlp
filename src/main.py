import utils, json, train, NNmodel, json, argparse, torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from utils import Plotter
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch.nn as nn
import torch.optim as optim

models = {
    'LogReg': LogisticRegression(),
    'RF': RandomForestClassifier(),
    'GB': GradientBoostingClassifier(),
    'LGB': LGBMClassifier(),
    'XGB': XGBClassifier()
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg.json")
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)

    X, y = utils.load_dataset(cfg['data_X'], cfg['data_y'])

    X['cleaned_text'] = X['text'].apply(utils.preprocess_text)

    X_temp, X_test, y_temp, y_test = train_test_split(X['cleaned_text'].values, y, test_size = 0.2, random_state=21, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.25, random_state=21, stratify=y_temp)
    

    vectorizer = TfidfVectorizer(max_features=500)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)   
    X_test_tfidf = vectorizer.transform(X_test) 
    
    #обучение классических моделей и вывод метрик
    metrics_test, metrics_val = [], []
    for model_name, model in models.items():
        y_pred_val, y_proba_val = train.fit_model(model, X_train_tfidf, y_train, X_val_tfidf) 
        y_pred, y_proba = model.predict(X_test_tfidf), model.predict_proba(X_test_tfidf)[:, 1]
        metrics_val.append(utils.get_metrics(model_name, y_val, y_pred_val, y_proba_val))
        metrics_test.append(utils.get_metrics(model_name, y_test, y_pred, y_proba)) 
        
        plotter = Plotter(cfg, model_name)
        plotter.plot_cm(y_test, y_pred) 
    utils.save_metrics(metrics_test, cfg, name_model='metrics_test')
    utils.save_metrics(metrics_val, cfg, name_model='metrics_val')
    plotter.plot_pr(metrics_test) 
    plotter.plot_roc(metrics_test) 
    plotter.plot_pr(metrics_val, type_metrics = 'val') 
    plotter.plot_roc(metrics_val, type_metrics = 'val') 

    # обучение нейронки
    X_train_NN, X_test_NN, y_train_NN, y_test_NN = train_test_split(X['cleaned_text'].values, y, test_size = 0.2, random_state=21)
    vectorizer = TfidfVectorizer(max_features=500)
    X_train_tfidf_NN = vectorizer.fit_transform(X_train_NN)
    X_test_tfidf_NN = vectorizer.transform(X_test_NN)

    train_loader, test_loader = utils.dataloader_NN(X_train_tfidf_NN, X_test_tfidf_NN, y_train_NN, y_test_NN)

    embedding_dim = 500
    model_NN = NNmodel.TextClassificationNN(embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_NN.parameters(), lr=0.01, weight_decay=1e-5)

    loss_history = train.fit_model_NN(model_NN, train_loader, criterion, optimizer, epochs=50)

    #метрики нейронки
    metrics_NN, y_pred, y_probs, y_true = utils.evaluate(model_NN, test_loader)
    utils.save_metrics_nn(metrics_NN, cfg)
    utils.plot_loss_history(loss_history, save_path=f'{cfg["images_folder"]}/NN_loss_history.png')
    utils.plot_confusion_matrix_NN(y_pred, y_true, save_path=f'{cfg["images_folder"]}/NN_confusion_matrix.png')
    utils.plot_roc_curve_NN(y_probs, y_true, save_path=f'{cfg["images_folder"]}/NN_roc_curve.png')
    utils.plot_pr_curve_NN(y_probs, y_true, save_path=f'{cfg["images_folder"]}/NN_pr_curve.png')
