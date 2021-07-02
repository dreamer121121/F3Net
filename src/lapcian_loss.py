def gaussian(ori_image, down_times=5):
    # 1：添加第一个图像为原始图像
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        # 2：连续存储5次下采样，这样高斯金字塔就有6层
        temp_gau = cv2.pyrDown(temp_gau)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid


def laplacian(gaussian_pyramid, up_times=5):
    laplacian_pyramid = [gaussian_pyramid[-1]]

    for i in range(up_times, 0, -1):
        # i的取值为5,4,3,2,1,0也就是拉普拉斯金字塔有6层
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        temp_lap = cv2.subtract(gaussian_pyramid[i - 1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid


def Lapyramid_loss(pred, target):
    lap_pyramid_pred = laplacian(gaussian(pred))
    lap_pyramid_target = laplacian(gaussian(target))

    W, H, C = pred.shape
    new_lap_pyramid_pred = [cv2.resize(item, (W, H)) for item in lap_pyramid_pred]
    new_lap_pyramid_target = [cv2.resize(item, (W, H)) for item in lap_pyramid_target]
    tmp = np.zeros((W, H))
    for i in range(1, 6):
        tmp += abs(new_lap_pyramid_pred[i] - new_lap_pyramid_target[i])
    return tmp[:, :, np.newaxis]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

