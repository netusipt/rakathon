import torchmetrics
from trainable_module import *
import numpy as np
import random
from parse_data import *
import pandas as pd
import torch.nn.init as init


from torch.utils.data import random_split


class Model(TrainableModule):
    def __init__(self, age_dim, tumour_dim, things_dim, fourth_dim, embeding_dim=4):
        super(Model, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=64),
            # torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=4),
        )

        self.embedding1 = nn.Embedding(age_dim, 1)
        self.embedding2 = nn.Embedding(tumour_dim, 1)
        self.embedding3 = nn.Embedding(things_dim, 1)
        self.embedding4 = nn.Embedding(fourth_dim, 1)

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.model:
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)  # Apply Xavier uniform
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x1, x2, x3, x4):
        x1 = x1.long()
        x2 = x2.long()
        x3 = x3.long()
        x4 = x4.long()
        x1 = self.embedding1(x1).view(-1, 1)
        x2 = self.embedding2(x2).view(-1, 1)
        x3 = self.embedding3(x3).view(-1, 1)
        x4 = self.embedding4(x4).view(-1, 1)

        x = torch.cat([x1, x2, x3, x4], dim=1)  # shape: [batch_size, 4]
        y = self.model(x)
        return y

    def embedding(self, x1, x2, x3, x4):
        x1 = x1.long()
        x2 = x2.long()
        x3 = x3.long()
        x4 = x4.long()

        x1 = self.embedding1(x1).view(-1, 1)
        x2 = self.embedding2(x2).view(-1, 1)
        x3 = self.embedding3(x3).view(-1, 1)
        x4 = self.embedding4(x4).view(-1, 1)


        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv("onco_data.csv")

    dataset = create_dataset(df)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    first_dim = 8
    second_dim = 2
    third_dim = 9
    fourth_dim = 9

    model = Model(first_dim, second_dim, third_dim, fourth_dim)

    model.configure(
        loss=torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 1, 1])),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        metrics={
            "val": torchmetrics.Accuracy(task="multiclass", num_classes=4),
        },
    )
    # dev = ...
    # test = ...

    model.fit(train_loader, dev=val_loader, epochs=6)

    evals = model.evaluate(test_loader)
    print(evals)
    # y_hat = model.predict(test_loader)
    # print(y_hat)
    # print(y_hat[0].shape)
    # print(y_hat.shape)

    x, y = next(iter(train_loader))

    res = model.embedding(x[0], x[1], x[2], x[3])
    print(res[:10])

    res = model.evaluate(test_loader)
    print(res)

    # pred = model.predict(test_loader)
    # print(pred.shape)


if __name__ == "__main__":
    main()
