import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # pass
        dropout_rate = 0.5
        self.conv_as_mlp1 = nn.Sequential(
            nn.Conv1d(3,64,1,padding='valid'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,1,padding='valid'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_as_mlp2 = nn.Sequential(
            nn.Conv1d(64,64,1,padding='valid'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,1,padding='valid'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1024,1,padding='valid'),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        # self.mlp1 =nn.Sequential(
        #     nn.Linear(3,64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Linear(64,64),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(64),
        # )
        # self.mlp2 = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 1024),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(1024),
        # )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.fc =  nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points.transpose(2,1) #(B, 3, N)
        x = self.conv_as_mlp1(x)
        x = self.conv_as_mlp2(x)
        x = self.maxpool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.conv_as_mlp1 = nn.Sequential(
            nn.Conv1d(3,64,1,padding='valid'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,64,1,padding='valid'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_as_mlp2 = nn.Sequential(
            nn.Conv1d(64,64,1,padding='valid'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,128,1,padding='valid'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1024,1,padding='valid'),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.maxpool =nn.AdaptiveMaxPool1d(1)

        self.conv_as_mlp3 = nn.Sequential(
            nn.Conv1d(1088,512,1,padding='valid'),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512,256,1,padding='valid'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1,padding='valid'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.conv_as_mlp4 = nn.Sequential(
            nn.Conv1d(128,128,1,padding='valid'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,num_seg_classes,1,padding='valid'),
        )
    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, point_dim = points.size()
        points = points.transpose(2,1)  # (B, 3, N)
        local_embeddings = self.conv_as_mlp1(points) # (B, 64, N)
        global_embeddings = self.maxpool(self.conv_as_mlp2(local_embeddings)) # (B, 1024)
        global_embeddings = global_embeddings.expand(-1, -1, N)
        x = torch.cat([local_embeddings, global_embeddings], dim=1) #B,1088,N
        x = self.conv_as_mlp3(x)
        x = self.conv_as_mlp4(x)#B,num_seg_classes,N,1
        x = x.transpose(2,1)
        return x
