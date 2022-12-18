
from torch.utils.data import DataLoader
from torchvision.models import get_model
from torchvision.datasets import CelebA
import os
import numpy as np
import torch
from tqdm.auto import tqdm
import random
from torchvision import transforms
import mlflow
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss
from ignite.metrics.metric import Metric

def transform_attrs(attrs):
    def transform(x):
        return x[attrs].to(torch.float32)
    return transform

def get_celeba_dataloaders(attributes_to_use_idx, transforms, batch_size=32, num_workers=16, train_size=None, val_size=None):
    valid_ds = CelebA(root='/home/daniel/', split='valid', download=False, transform=transforms, target_transform=transform_attrs(attributes_to_use_idx))
    test_ds = CelebA(root='/home/daniel/', split='test', download=False, transform=transforms, target_transform=transform_attrs(attributes_to_use_idx))
    train_ds = CelebA(root='/home/daniel/', split='train', download=False, transform=transforms, target_transform=transform_attrs(attributes_to_use_idx))

    if train_size is not None:
        idxs = np.arange(0, len(train_ds))
        np.random.shuffle(idxs)
        train_ds = torch.utils.data.Subset(train_ds, idxs[:train_size])
    if val_size is not None:
        idxs = np.arange(0, len(valid_ds))
        np.random.shuffle(idxs)
        valid_ds = torch.utils.data.Subset(valid_ds, idxs[:val_size])    

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, valid_dataloader, test_dataloader


def calculate_accuracy(outs, y_true):
    preds = torch.round(torch.sigmoid(outs)).detach()
    y_true = y_true.detach()

    acc_f1 = (preds[:, 0] == y_true[:, 0]).to(torch.float32).mean()
    acc_f2 = (preds[:, 1] == y_true[:, 1]).to(torch.float32).mean()
    acc_f3 = (preds[:, 2] == y_true[:, 2]).to(torch.float32).mean()

    weighted_acc = (acc_f1*0.15 + acc_f2 * 0.15 + acc_f3 * 0.7)
    return [weighted_acc.item(), acc_f1.item(), acc_f2.item(), acc_f3.item()]


def custom_output_transform(x, y, y_pred, loss):
    return y_pred, y


class WeightedAccuracy(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._iter_accuracy = []
        super(WeightedAccuracy, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._iter_accuracy.clear()

    def update(self, output):
        y_pred, y = output
        self._iter_accuracy.append(calculate_accuracy(y_pred, y))

    def compute(self):
        return np.mean(self._iter_accuracy, axis=0)[0]


def validate_on_epoch_end(trainer, evaluator, valid_dl, best_metrics):
    t_metrics = trainer.state.metrics
    print(f"Epoch number: {trainer.state.epoch}, Loss: {t_metrics['loss']:.4f}, Accuracy: {t_metrics['accuracy']:.4f}")
    evaluator.run(valid_dl)
    metrics = evaluator.state.metrics
    print(f"Validation Results - Epoch: {trainer.state.epoch}  Avg accuracy: {metrics['accuracy']:.4f} Avg loss: {metrics['loss']:.4f}")
    best_metrics['val_acc'].append(metrics['accuracy'])
    best_metrics['val_loss'].append(metrics['loss'])
    best_metrics['train_acc'].append(t_metrics['accuracy'])
    best_metrics['train_loss'].append(t_metrics['loss'])    


def train2(model, train_dl, valid_dl, epochs=10, lr=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.to(device)
    progress_bar = tqdm(range(epochs*len(train_dl)))
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device, output_transform=custom_output_transform)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda _: progress_bar.update(1))
    
    val_metrics = {"accuracy": WeightedAccuracy(), "loss": Loss(loss_fn)}
    evaluator = create_supervised_evaluator(model, device=device, metrics=val_metrics)
    
    for name, metric in val_metrics.items():
        metric.attach(trainer, name)
    
    best_metrics = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}
    trainer.add_event_handler(Events.EPOCH_COMPLETED, validate_on_epoch_end, evaluator, valid_dl, best_metrics)

    trainer.run(train_dl, max_epochs=epochs)

    return (np.min(best_metrics['train_loss']), np.max(best_metrics['train_acc']), np.min(best_metrics['val_loss']), np.max(best_metrics['val_acc']))

        
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print('seed_everything done: ', seed)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_mlflow_experiment():
    exp_name = 'Celeba Gender/Hat/Glasses'
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        mlflow.create_experiment(
            exp_name,
            tags={"version": "v1"},
        )
    mlflow.set_experiment(exp_name)


def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

def modify_output_layer(model, num_classes):
    l = len(model.classifier)
    model.classifier.apply(weight_reset)
    model.classifier[l-1] = torch.nn.Linear(in_features=model.classifier[l-1].in_features, out_features=num_classes)


def get_base_model(name, pretrained=False, num_classes=3):
    weights = 'DEFAULT' if pretrained else None
    model = get_model(name, weights=weights)
    if name == 'googlenet' or name == 'resnet50':
        model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes)
    else:    
        modify_output_layer(model, num_classes)
    return model

def do_train(train_size, model_name, pretrained, epochs, batch_size, lr):
    seed_everything(42)
    transform = get_transforms()
    attributes_to_use_idx = [15, 35, 20] #Eyeglasses, Wearing_Hat, Male
    train_dl, valid_dl, test_dl = get_celeba_dataloaders(attributes_to_use_idx, transform, batch_size=batch_size, num_workers=16, train_size=train_size)
    model = get_base_model(model_name, pretrained=pretrained, num_classes=len(attributes_to_use_idx))
    metrics = train2(model, train_dl, valid_dl, epochs=epochs, lr=lr)
    return model, metrics


if __name__ == '__main__':    
    create_mlflow_experiment()
    hyperparams = {
        'train_size': 150000,
        'model_name': 'resnet50',
        'pretrained': True,
        'epochs': 10,
        'batch_size': 32,
        'lr': 1e-4
    }
    with mlflow.start_run():
        mlflow.log_params(hyperparams)
        model, metrics = do_train(**hyperparams)
        (best_train_loss, best_train_acc, best_val_loss, best_val_acc) = metrics
        mlflow.log_metric('train_BCE', best_train_loss)
        mlflow.log_metric('train_accuracy', best_train_acc)
        mlflow.log_metric('BCE', best_val_loss)
        mlflow.log_metric('accuracy', best_val_acc)
        mlflow.pytorch.log_model(model, 'model')


# Separar en archivos (Training y visualizacion utils) OK
# Reducir en 30% las lineas de codigo usando ignite (a 150)
# Agregar mecanismo de detencion temprana manual