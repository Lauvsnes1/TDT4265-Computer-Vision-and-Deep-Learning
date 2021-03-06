import pathlib
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Linear
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import time


class Model2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            #First hidden conv layer
            nn.Conv2d(in_channels=image_channels, out_channels=num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            #Second hidden conv layer
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            #Third hidden conv layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            #Fourth hidden conv layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            #Dropout
            nn.Dropout(0.25)
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 4 * 4 * 128
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)

            #First linear layer
            nn.Linear(self.num_output_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            #Second linear layer
            nn.Linear(128,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            #Dropout
            nn.Dropout(0.25),

            #Third linear layer
            nn.Linear(128,num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out = x
        # Hidden conv layer
        out = self.feature_extractor(out)
        # Hidden linear layer
        out = self.classifier(out) 

        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    utils.plot_loss(trainer.test_history["loss"], label="Test loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.train_history["accuracy"], label="Train Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.test_history["accuracy"], label="Test Accuracy")
    
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    starttime = time.time()
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model2(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    endtime = time.time()
    print("Time is took to run the network: ", endtime-starttime)
    create_plots(trainer, "task3_model_e")

if __name__ == "__main__":
   main()