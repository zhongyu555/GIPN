
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import numpy as np
import random

class MapMaker(nn.Module):

    def __init__(self,image_size):

        super(MapMaker, self).__init__()
        self.image_size = image_size
        print('MapMaker initialized')


    def forward(self, vision_adapter_features,vision_gip_features,propmt_adapter_features):
        anomaly_maps_gip=[]
        anomaly_maps=[]

        for i,vision_feature in enumerate(vision_adapter_features):
            vision_feature = vision_feature.permute(0, 2, 3,1)
            B, H, W, C = vision_feature.shape
            anomaly_map = (vision_feature.view((B, H * W, C)) @ propmt_adapter_features).contiguous().view(
                (B, H, W, -1)).permute(0, 3, 1, 2)

            anomaly_maps.append(anomaly_map)
        anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)

        anomaly_map = F.interpolate(anomaly_map, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        for i,vision_gip_feature in enumerate(vision_gip_features):
            vision_gip_feature = vision_gip_feature.permute(0, 2, 3,1)
            B, H, W, C = vision_gip_feature.shape
            anomaly_map_gip = (vision_gip_feature.view((B, H * W, C)) @ propmt_adapter_features).contiguous().view(
                (B, H, W, -1)).permute(0, 3, 1, 2)

            anomaly_maps_gip.append(anomaly_map_gip)
        anomaly_map_gip = torch.stack(anomaly_maps_gip, dim=0).mean(dim=0)

        anomaly_map_gip = F.interpolate(anomaly_map_gip, (self.image_size, self.image_size), mode='bilinear', align_corners=True)

        return torch.softmax(anomaly_map, dim=1), torch.softmax(anomaly_map_gip, dim=1)
