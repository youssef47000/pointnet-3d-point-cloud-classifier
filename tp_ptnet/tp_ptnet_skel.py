import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Conv1d
# from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils import data
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from os import listdir
from os import makedirs
from os.path import join
from os.path import exists
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def visualize(x):
    # Creating figure
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    # Creating plot
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], color="blue")
    plt.title("Point Set")

    # show plot
    plt.show()


class DatasetFromFolder(data.Dataset):
    def __init__(self, datadir):
        super(DatasetFromFolder, self).__init__()

        datadir1 = join(datadir, '00')
        filenames1 = [join(datadir1, x) for x in listdir(datadir1)]

        datadir2 = join(datadir, '01')
        filenames2 = [join(datadir2, x) for x in listdir(datadir2)]

        datadir3 = join(datadir, '02')
        filenames3 = [join(datadir3, x) for x in listdir(datadir3)]

        self.filenames = filenames1 + filenames2 + filenames3

    def __getitem__(self, index):
        name = self.filenames[index]

        input = torch.from_numpy(np.loadtxt(name).transpose(1, 0)).float()

        # Récupère le nom du dossier parent ("00", "01", "02") pour définir la classe
        folder = os.path.basename(os.path.dirname(name))

        target = torch.zeros([1], dtype=torch.long)

        if folder == "00":
            target[0] = 0
        elif folder == "01":
            target[0] = 1
        elif folder == "02":
            target[0] = 2
        else:
            print("BUG →", name, "FOLDER =", folder)

        return input, target

    def __len__(self):
        return len(self.filenames)


# Prédit une matrice de transformation pour rendre le réseau :
# - invariant à la rotation
# - invariant à l’inclinaison
# - invariant à la permutation
# - capable d'aligner tous les objets dans un repère canonique
class MyTNet(nn.Module):
    def __init__(self, dim=3):
        super(MyTNet, self).__init__()

        self.dim = dim

        self.conv1 = Conv1d(in_channels=dim, out_channels=64, kernel_size=1)
        self.bach1 = nn.BatchNorm1d(64)

        self.conv2 = Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bach2 = nn.BatchNorm1d(128)

        self.conv3 = Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bach3 = nn.BatchNorm1d(1024)

        self.linear1 = nn.Linear(1024, 512)
        self.bach11 = nn.BatchNorm1d(512)

        self.linear2 = nn.Linear(512, 256)
        self.bach22 = nn.BatchNorm1d(256)

        self.linear3 = nn.Linear(256, dim * dim)

    def forward(self, x):
        # Entrée : tenseur (B, C, N).
        # On applique une Conv1d (kernel=1) sur chaque point 'n' en parallèle.
        # C'est comme un MLP appliqué point par point (shared weights).
        # x : (B, 3, N) -> On veut obtenir une matrice (B, 3, 3) unique pour transformer l'objet entier.
        x1 = self.conv1(x)
        x1 = F.relu(self.bach1(x1))

        x2 = self.conv2(x1)
        x2 = F.relu(self.bach2(x2))

        x3 = self.conv3(x2)
        x3 = F.relu(self.bach3(x3))

        # Extraction de la signature globale de l'objet (Global Feature).
        # On garde la valeur max pour chaque feature sur l'ensemble des points (dim=2).
        # Exemple : point 1 [1, 5...], point 2 [1, 20...] -> Feature max : 20.
        # Cela rend le réseau invariant à l'ordre des points (permutation).
        x4 = torch.max(x3, dim=2)[0]  # (B, 1024, N)  →  (B, 1024). [0] récupère les valeurs, on ignore les indices.

        # Couches pleinement connectées pour réduire la dimension et prédire la matrice
        x5 = self.linear1(x4)
        x5 = F.relu(self.bach11(x5))

        x6 = self.linear2(x5)
        x6 = F.relu(self.bach22(x6))

        # Matrice de transformation prédite (à plat, vecteur de taille dim*dim).
        x7 = self.linear3(x6)

        # Initialisation à l'identité :
        # On ajoute la matrice identité à la sortie pour stabiliser l'entraînement.
        # Sans ça, le réseau pourrait commencer avec une matrice nulle ou aléatoire
        # qui écraserait le nuage de points, rendant l'apprentissage impossible au début.
        myidentity = torch.from_numpy(np.eye(self.dim, dtype=np.float32)).view(1, self.dim * self.dim).repeat(
            x.size()[0], 1)  # (3×3) → (1×9)
        if x.is_cuda:
            myidentity = myidentity.cuda()

        x7 = x7 + myidentity

        # On remet en forme (B, dim, dim)
        x8 = x7.view(-1, self.dim, self.dim)

        return x8


# PointNet Architecture Principale
class MyPointNet(nn.Module):
    def __init__(self, dim=3, dimfeat=64, num_class=3):
        super(MyPointNet, self).__init__()

        self.tnet1 = MyTNet(dim)

        self.fc1 = Conv1d(in_channels=dim, out_channels=64, kernel_size=1)
        self.bach1 = nn.BatchNorm1d(64)

        self.fc2 = Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.bach2 = nn.BatchNorm1d(64)

        self.tnet2 = MyTNet(dimfeat)

        self.fc3 = Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bach3 = nn.BatchNorm1d(128)

        self.fc4 = Conv1d(in_channels=128, out_channels=1024, kernel_size=1)
        self.bach4 = nn.BatchNorm1d(1024)

        self.fc5 = nn.Linear(1024, 512)
        self.bach5 = nn.BatchNorm1d(512)

        self.fc6 = nn.Linear(512, 256)
        self.bach6 = nn.BatchNorm1d(256)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc7 = nn.Linear(256, num_class)

    def forward(self, x):
        # x : (B, 3, N)
        # TNet1 sort un tenseur (B, 3, 3) : c'est la matrice d'alignement spatial (Input Transform)
        tn1 = self.tnet1(x)

        # On applique la transformation pour aligner le nuage de points
        alignement3 = torch.bmm(tn1, x)  # (B, 3, N)

        x1 = F.relu(self.bach1(self.fc1(alignement3)))
        x2 = F.relu(self.bach2(self.fc2(x1)))

        # x2 : (B, 64, N)
        # TNet2 sort un tenseur (B, 64, 64) : alignement dans l'espace des features (Feature Transform)
        tn2 = self.tnet2(x2)
        alignement64 = torch.bmm(tn2, x2)  # (B, 64, N)

        x3 = F.relu(self.bach3(self.fc3(alignement64)))
        x4 = F.relu(self.bach4(self.fc4(x3)))

        # Vecteur caractéristique global (Signature de l'objet)
        # On prend le max sur la dimension des points pour avoir un descripteur unique
        max_global = torch.max(x4, dim=2).values

        x5 = F.relu(self.bach5(self.fc5(max_global)))
        x6 = F.relu(self.bach6(self.fc6(x5)))

        x7 = (self.logsoftmax(self.fc7(x6)))

        return x7, tn1, tn2


if __name__ == "__main__":

    myptnet = MyPointNet().to(device)


    def tnet_regularization(matrix):
        # matrix : (B, k, k)
        # Force la matrice de transformation à être proche d'une matrice orthogonale
        # Cela évite les déformations bizarres (étirements infinis) et stabilise l'alignement.
        I = torch.eye(matrix.size(1)).to(matrix.device)
        loss = torch.norm(torch.bmm(matrix, matrix.transpose(2, 1)) - I, dim=(1, 2)).mean()
        return loss


    '''
    Note sur le papier PointNet :
     - Le modèle est sensible au bruit additif gaussien.
     - Il est surtout robuste aux transformations géométriques (rotation, translation).
     - Il résiste mal au bruit fort → c'est pour ça que PointNet++ a été créé.

     L'ajout de bruit pendant l'entraînement (Data Augmentation) :
     - Oblige le réseau à apprendre les formes globales.
     - Évite qu’il ne se focalise trop sur des points précis (Overfitting).
     - Améliore la généralisation et la robustesse au test.
     '''


    # Crée un tenseur de même taille que les points,
    # rempli avec du bruit aléatoire (distribution normale)
    def bruit(points, sigma):
        # Notes perso sur l'impact du bruit (sigma) sur l'accuracy :
        # 5% de bruit -> perte de 10% d'accuracy environ.
        # 0.01 -> impact faible (kif-kif).
        # 10% -> Accuracy chute à 42.77%.
        # 20% -> 41.44%.
        # 50% -> 33% (le réseau est perdu).
        bruit = torch.randn_like(points) * sigma
        return points + bruit


    if not exists('mypointnet_bruit.pt'):
        myptnet.to(device)

        num_epochs = 100
        num_w = 4
        batch_s = 32

        # Loss and optimizer
        optimizer = optim.SGD(myptnet.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.NLLLoss()

        # Loading data
        trainset = DatasetFromFolder("data/train/", )
        trainloader = DataLoader(trainset, num_workers=num_w, batch_size=batch_s, shuffle=True)
        losslog = []
        max_sigma = 0.10

        # Training loop
        for epoch in range(0, num_epochs):

            for i, data in enumerate(trainloader, 0):
                points, target = data
                points, target = points.to(device), target.to(device)

                # Injection de bruit aléatoire pour robustifier le modèle
                sigma = torch.rand(1).item() * max_sigma
                points = bruit(points, sigma)

                optimizer.zero_grad()
                output, tn1, tn2 = myptnet(points)
                target = target.view(-1)

                reg1 = tnet_regularization(tn1)
                reg2 = tnet_regularization(tn2)

                # Loss totale = Classification + Régularisation des matrices de transformation
                loss = criterion(output, target) + 0.001 * (reg1 + reg2)
                loss.backward()
                optimizer.step()
                losslog.append(loss.item())

            # if (epoch-1)%1 == 0 :
            print(f"{epoch} Epoch - training loss:{losslog[-1]:.4f}")

        # Display the result
        plt.figure(figsize=(6, 4))
        plt.yscale('log')
        plt.plot(losslog, label='loss ({:.4f})'.format(losslog[-1]))
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
        plt.close()

        torch.save(myptnet.state_dict(), 'mypointnet_bruit.pt')

    else:
        # Read the saved model
        myptnet.load_state_dict(torch.load('mypointnet_bruit.pt'))
        myptnet.eval()

    testset = DatasetFromFolder("data/test")
    testloader = DataLoader(testset, num_workers=4, batch_size=1, shuffle=False)

    max_sigma = 0.10
    gtlabels = []
    predlabels = []
    criterion = nn.NLLLoss()
    losslog = []
    correct = 0
    total = 0
    myptnet.eval()

    # Test avec le réseau entraîné et ajout de bruit gaussien (10% max)
    # pour prouver que le réseau généralise bien et ne perd que peu d'accuracy (env. 5% de perte -> 90.55%)
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            points, target = data
            points, target = points.to(device), target.to(device)

            sigma = torch.rand(1).item() * max_sigma
            points = bruit(points, sigma)

            output, _, _ = myptnet(points)
            loss = criterion(output, target.view(-1))

            pred = torch.argmax(output, dim=1)
            correct += (pred == target.view(-1)).sum().item()

            losslog.append(loss.item())
            total += target.view(-1).size(0)

            gtlabels.append(target.view(-1).item())
            predlabels.append(pred.view(-1).item())

        accuracy = 100 * correct / total
        print("accuracy : ", accuracy)

    cm = confusion_matrix(gtlabels, predlabels)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()