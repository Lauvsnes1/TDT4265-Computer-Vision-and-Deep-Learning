import pathlib
import torch
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Linear
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import time
from task2 import ExampleModel
from task3_model2 import Model2



def create_plots(trainer_examplemodel: Trainer, trainer_model2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer_examplemodel.train_history["loss"], label="Training loss - exampleModel", npoints_to_average=10)
    utils.plot_loss(trainer_examplemodel.validation_history["loss"], label="Validation loss - exampleModel")

    utils.plot_loss(trainer_model2.train_history["loss"], label="Training loss - Model2", npoints_to_average=10)
    utils.plot_loss(trainer_model2.validation_history["loss"], label="Validation loss - Model2")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    
    starttime = time.time()
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model_ex = ExampleModel(image_channels=3, num_classes=10)
    trainer_examplemodel = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_ex,
        dataloaders
    )
    trainer_examplemodel.optimizer = torch.optim.SGD(model_ex.parameters(), learning_rate)
    trainer_examplemodel.train()

    model_2 = Model2(image_channels=3, num_classes=10)
    trainer_model2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model_2,
        dataloaders
        )
    trainer_model2.optimizer = torch.optim.ASGD(model_2.parameters(), learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
    trainer_model2.train()

    endtime = time.time()
    print("Time is took to run the network: ", endtime-starttime)
    
    create_plots(trainer_examplemodel, trainer_model2, "task3d")

if __name__ == "__main__":
    main()