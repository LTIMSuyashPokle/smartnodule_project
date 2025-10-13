from mlops.experiment_tracker import MLflowTracker
import torch

tracker = MLflowTracker(
    tracking_uri="sqlite:///mlflow.db", 
    experiment_name="SmartNodule_Production"
)

# Fake model and data
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2,1)
    def forward(self, x):
        return self.fc(x)
        
model = DummyModel()

tracker.log_training_run(
    model=model,
    train_metrics={'accuracy': 0.95, 'loss': 0.1},
    val_metrics={'accuracy': 0.93, 'loss': 0.12},
    test_metrics={'accuracy': 0.92, 'loss': 0.14},
    hyperparameters={'lr': 0.001, 'epochs': 10},
    model_path='./model.pth',
    run_name="TestRun"
)
