import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torchvision.models as models
import utils as ksd

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_adv_img_tensor(img_teonsor, device=None):
    adv_image_tensor = Variable(img_teonsor.clone().detach().to(device), requires_grad=True)
    return adv_image_tensor


def fgsm(model, img_tensor, e=0.001, max_iterations=500, target_label=1, device=None):
    """
    fgsm 目标攻击算法
    :param model: 要攻击的模型
    :param img_tensor: 传入的图片的tensor，后面会发生改变，所以直接传攻击结果向量即可
    :param e: 超参数
    :param max_iterations: 最大迭代次数
    :param target_label:  目标类别
    :return: True or False 攻击是否成功
    """

    adv_img_tensor = get_adv_img_tensor(img_tensor, device=device)
    # 模型的参数不需要发生变化
    for param in model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.Adam([adv_img_tensor])
    target = Variable(torch.Tensor([float(target_label)]).to(device).long())
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(max_iterations):
        # forward + backward
        output = model(torch.unsqueeze(adv_img_tensor, 0))
        loss = loss_func(output, target)
        label = np.argmax(output.data.cpu().numpy())
        # 如果定向攻击成功
        if label == target:
            return adv_img_tensor
        # 梯度清零
        optimizer.zero_grad()
        # 反向传递 计算梯度
        loss.backward()
        adv_img_tensor.data = adv_img_tensor.data - e * torch.sign(adv_img_tensor.grad.data)
    else:
        return None


def deepfool_untarget(model, img_tensor, original_label, max_iterations=100, num_classes=1000, overshoot=0.02, device=None):
    adv_img_tensor = get_adv_img_tensor(img_tensor)
    input_shape = adv_img_tensor.cpu().detach().numpy().shape
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)
    output = model(adv_img_tensor)
    for epoch in range(max_iterations):
        scores = model(adv_img_tensor).data.cpu().numpy()[0]
        adv_label = np.argmax(scores)
        print("epoch={} label={} score={}".format(epoch, adv_label, scores[label]))
        # 如果无定向攻击成功
        if adv_label != original_label:
            return adv_img_tensor
        pert = np.inf
        output[0, original_label].backward(retain_graph=True)
        grad_orig = adv_img_tensor.grad.data.cpu().numpy().copy()
        for k in range(1, num_classes):
            if k == original_label:
                continue
            # 梯度清零
            adv_img_tensor.grad.zero_()
            output[0, k].backward(retain_graph=True)
            cur_grad = adv_img_tensor.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (output[0, k] - output[0, original_label]).data.cpu().numpy()

            pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

            # 选择pert最小值
            if pert_k < pert:
                pert = pert_k
                w = w_k
        # 计算 r_i 和 r_tot
        r_i = (pert + 1e-8) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        adv_img_tensor.data = adv_img_tensor.data + (1 + overshoot) * torch.from_numpy(r_tot).to(device)


def deepfool_target(model, img_tensor, target_label,
                    max_iterations=100, overshoot=0.02):
    adv_img_tensor = get_adv_img_tensor(img_tensor)
    for param in model.parameters():
        param.requires_grad = False
    input_shape = adv_img_tensor.cpu().detach().numpy().shape
    target = Variable(torch.Tensor([float(target_label)]).to(device).long())
    r_tot = np.zeros(input_shape)
    loss_func = torch.nn.CrossEntropyLoss()
    output = model(adv_img_tensor)
    output[0, target].backward(retain_graph=True)
    for epoch in range(max_iterations):
        output = model(adv_img_tensor)
        label = np.argmax(output.data.cpu().numpy())
        loss = loss_func(output, target)
        # print("epoch={} label={} loss={}".format(epoch, label, loss))
        # 如果定向攻击成功
        if label == target_label:
            return adv_img_tensor
        # 梯度清零
        adv_img_tensor.grad.zero_()
        output[0, target_label].backward(retain_graph=True)
        w = adv_img_tensor.grad.data.cpu().numpy().copy()
        f = output[0, target_label].data.cpu().numpy()
        pert = abs(f) / np.linalg.norm(w.flatten())
        # 计算 r_i 和 r_tot
        r_i = (pert + 1e-8) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)
        adv_img_tensor.data = adv_img_tensor.data + (1 + overshoot) * torch.Tensor(r_tot).to(device)


def cw(model, img_tensor, max_iterations=1000, learning_rate=0.01, binary_search_steps=10,
       confidence=1e2, k=40, box_area=(-3.0, 3.0), num_labels=1000, target_label=288):
    """
    cw攻击算法 ，有目标攻击
    :param model:   攻击的分类器
    :param img_tensor: 输入图像的tensor
    :param max_iterations: 最大迭代次数，论文中设置的是10000次，但是1000次已经可以完成95%的优化工作
    :param learning_rate: adam learning rate
    :param binary_search_steps: 二分查找次数
    :param confidence:   c 的初始值
    :param k:   k 值
    :param box_area: 像素区间值，(-3.0,3.0)
    :param target_label: 攻击目标
    :param num_labels:
    :return: 返回值是最佳攻击图像
    """

    tlab = Variable(torch.from_numpy(np.eye(num_labels)[target_label]).to(device).float())
    shape = (1, 3, 224, 224)
    # c的初始化边界
    lower_bound = 0
    c = confidence
    upper_bound = 1e10

    # the best l2, score, and image attack
    o_bestl2 = 1e10
    o_bestscore = -1
    o_bestattack = [np.zeros(shape)]

    # the resulting image, tanh'd to keep bounded from boxmin to boxmax
    boxmul = (box_area[1] - box_area[0]) / 2.
    boxplus = sum(box_area) / 2.
    # 设置为不保存梯度值 自然也无法修改
    for param in model.parameters():
        param.requires_grad = False
    for outer_step in range(binary_search_steps):
        print("o_bestl2={} confidence={}".format(o_bestl2, confidence))
        # 把原始图像转换成图像数据和扰动的形态
        timg = Variable(
            torch.from_numpy(np.arctanh((img_tensor.numpy() - boxplus) / boxmul * 0.999999)).to(device).float())
        modifier = Variable(torch.zeros_like(timg).to(device).float())
        # 图像数据的扰动量梯度可以获取
        modifier.requires_grad = True
        # 定义优化器 仅优化modifier
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)
        for iteration in range(1, max_iterations + 1):
            optimizer.zero_grad()
            # 定义新输入
            newimg = torch.tanh(modifier + timg) * boxmul + boxplus
            output = model(newimg)
            loss2 = torch.dist(newimg, (torch.tanh(timg) * boxmul + boxplus), p=2)
            """
            # compute the probability of the label class versus the maximum other
                real = tf.reduce_sum((tlab)*output,1)
                # 论文中的开源实现 other = tf.reduce_max((1-tlab)*output - (tlab*10000),1)
                other = tf.reduce_max((1-tlab)*output)
                loss1 = tf.maximum(0.0, other-real+k)
                loss1 = tf.reduce_sum(const*loss1)
            """
            real = torch.max(output * tlab)
            other = torch.max((1 - tlab) * output)
            loss1 = other - real + k
            loss1 = torch.clamp(loss1, min=0)
            loss1 = confidence * loss1
            loss = loss1 + loss2
            loss.backward(retain_graph=True)
            optimizer.step()
            l2 = loss2
            sc = output.data.cpu().numpy()
            # print out the losses every 10%
            if iteration % (max_iterations // 10) == 0:
                print("iteration={} loss={} loss1={} loss2={}".format(iteration, loss, loss1, loss2))
            if (l2 < o_bestl2) and (np.argmax(sc) == target_label):
                print("attack success l2={} target_label={}".format(l2, target_label))
                o_bestl2 = l2
                o_bestscore = np.argmax(sc)
                o_bestattack = newimg.data.cpu().numpy()
        confidence_old = -1
        if (o_bestscore == target_label) and o_bestscore != -1:
            # 攻击成功 减小c
            upper_bound = min(upper_bound, confidence)
            if upper_bound < 1e9:
                print()
                confidence_old = confidence
                confidence = (lower_bound + upper_bound) / 2
        else:
            lower_bound = max(lower_bound, confidence)
            confidence_old = confidence
            if upper_bound < 1e9:
                confidence = (lower_bound + upper_bound) / 2
            else:
                confidence *= 10
        print("outer_step={} confidence {}->{}".format(outer_step, confidence_old, confidence))
    return torch.Tensor(o_bestattack)


if __name__ == "__main__":
    device = ksd.try_gpu()
    import classifiers as clfs
    # original_image_path = "./pics/cropped_panda.jpg"
    # ToTensor_transform = transforms.Compose([transforms.Resize((224, 224)),
    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #
    # original_image_tensor = ksd.image_preprocessing(original_image_path, ToTensor_transform, is_unsqueeze=True)
    #
    # adv_image_tensor = Variable(original_image_tensor.clone().detach().to(device))
    # adv_image_tensor.requires_grad = True

    model = clfs.Classifer_MNIST()
    model.load_state_dict(torch.load('./models/classifiers/classifier_mnist.pth'))
    model.eval()
    model.to(device)
    test_data_set = ksd.load_data_mnist_test(100)
    for X_iter, y_iter in test_data_set:
        for i in range(100):
            print(X_iter[i].shape)
            X_adv = fgsm(model, X_iter[i], e=0.01, device=device, target_label=1)
            print(X_adv.shape)
            break
        break


    # output = torch.softmax(output, 1, dtype=torch.float32)
    # score, label = torch.max(output, 1)
    # score = score / torch.sum(output, dtype=torch.float32)
    # print("score={},label={}".format(score.data, label.data))
    # # adv_image_tensor = deepfool_target(model, original_image_tensor, target_label=288)
    # adv_image_tensor = fgsm(model, original_image_tensor, e=0.001, device=ksd.try_gpu())
    # # adv_image_tensor = deepfool_untarget(model, original_image_tensor, original_label=label)
    # # adv_image_tensor = cw(model, original_image_tensor, target_label=288)
    # ksd.transform_to_img_from_dataset(adv_image_tensor.cpu().clone().detach(), ToTensor_transform, "pics/test111_adv.png")
    # ksd.show_images_diff(original_image_tensor, adv_image_tensor.cpu().clone().detach(), ToTensor_transform, label, 288)
