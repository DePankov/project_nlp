from typing import Tuple
import pandas as pd

def fit_model(model, X_train, y_train, X_test) -> Tuple[pd.Series, pd.Series]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def fit_model_NN(model, dataloader, criterion, optimizer, epochs = 5) -> list:
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    loss_history = []

    for epoch in range(epochs):
        for X, y in dataloader:
            optimizer.zero_grad()  
            outputs = model(X)     
            loss = criterion(outputs, y)
            loss.backward()        
            optimizer.step()      

            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        if epoch %10 == 0:
            print(f'Epoch {epoch+10}/{epochs}, Loss: {avg_loss:.4f}')
    return loss_history