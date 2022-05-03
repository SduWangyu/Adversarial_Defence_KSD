import torch
from torch.autograd import Variable
import torch.utils.data.dataset
from torch import nn
from utils import misc


def get_adv_img_tensor(img_teonsor, device=None):
    adv_image_tensor = Variable(img_teonsor.clone().detach().to(device), requires_grad=True)
    return adv_image_tensor


def fgsm_i(net, x_input, y_input, target=False, eps=0.1, alpha=1, iteration=1000,
           x_val_min=-1, x_val_max=1, device=None):
    x_adv = Variable(misc.copy_tensor(x_input).to(device), requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    for i in range(iteration):
        h_adv = net(x_adv)
        adv_label = h_adv.argmax(1)
        if target:
            loss = loss_fn(h_adv, y_input)
            if adv_label == y_input:
                return x_adv, adv_label
        else:
            loss = -loss_fn(h_adv, y_input)
            if adv_label != y_input:
                return x_adv, adv_label
        net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        loss.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - alpha * x_adv.grad
        x_adv = misc.where(x_adv > x_input + eps, x_input + eps, x_adv)
        x_adv = misc.where(x_adv < x_input - eps, x_input - eps, x_adv)
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
        x_adv = Variable(x_adv.data, requires_grad=True)
    adv_label = h_adv.argmax(1)
    return x_adv, adv_label


def deepfool(net, x_input, y_input, target_label=None, max_iterations=100, num_labels=10, overshoot=0.02, device=None):
    x_adv = Variable(misc.copy_tensor(x_input).to(device), requires_grad=True)
    for param in net.parameters():
        param.requires_grad = False
    input_shape = x_adv.shape
    w = torch.zeros(input_shape)
    w_norm = torch.inf
    pert = torch.inf
    if target_label:
        for epoch in range(max_iterations):
            h_adv = net(x_adv)
            adv_label = h_adv.argmax(1)
            print(f'epoch: {epoch}, label:{adv_label}, score:{h_adv[0, adv_label]}')
            if adv_label == target_label:
                return x_adv, adv_label
            h_adv[0, y_input].backward(retain_graph=True)
            org_label_grad = misc.copy_tensor(x_adv.grad).to(device)
            x_adv.grad.zero_()
            h_adv[0, target_label].backward(retain_graph=True)
            w = x_adv.grad - org_label_grad
            f = h_adv[0, y_input] - h_adv[0, target_label]
            # + 1e-8是为了保证结果不为0
            w_norm = torch.norm(w.flatten()) + 1e-8
            pert = (abs(f) + 1e-8) / w_norm
            r_i = pert * w / w_norm
            x_adv.data = x_adv.data + (1 + overshoot) * r_i
    else:
        for epoch in range(max_iterations):
            h_adv = net(x_adv)
            adv_label = h_adv.argmax(1)
            # print(h_adv[0, y_input])
            # print(f'epoch: {epoch}, label:{adv_label}, score:{h_adv[0, adv_label]}')
            if adv_label != y_input:
                print("success")
                return x_adv, adv_label
            h_adv[0, y_input].backward(retain_graph=True)
            org_label_grad = misc.copy_tensor(x_adv.grad.data).to(device)
            for k in range(0, num_labels):
                if k == y_input:
                    continue
                # 梯度清零
                x_adv.grad.zero_()
                h_adv[0, k].backward(retain_graph=True)
                w_k = x_adv.grad.data - org_label_grad.data
                f_k = h_adv[0, k] - h_adv[0, y_input]
                # + 1e-8是为了保证结果不为0
                w_k_norm = torch.norm(w_k.flatten()) + 1e-8
                pert_k = (abs(f_k) + 1e-8) / w_k_norm
                # 选择pert最小值
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
                    w_norm = w_k_norm
            r_i = pert * w / w_norm
            x_adv.data = x_adv.data + (1 + overshoot) * r_i

    print("faild")
    return x_adv, adv_label


def cw_l2(net, x_input, y_input, max_iterations=1000, learning_rate=0.01, binary_search_steps=10,
          confidence=1e2, k=0, box_min=-3.0, box_max=3.0, num_labels=10, target_label=None, device=None):
    """
    cw攻击算法 l2 ，有目标攻击
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
    for param in net.parameters():
        param.requires_grad = False

    # c的初始化边界
    lower_bound = 0
    upper_bound = 1e10

    # the best l2, score, and image attack
    o_bestl2 = 1e10
    o_bestscore = -1
    o_bestattack = torch.zeros(x_input.shape).to(device)

    # the resulting image, tanh'd to keep bounded from boxmin to boxmax
    boxmul = (box_max - box_min) / 2.
    boxplus = (box_max + box_min) / 2.

    # 设置为不保存梯度值 自然也无法修改
    if target_label:
        tlb = torch.eye(num_labels)[target_label].float().to(device)
    else:
        tlb = torch.eye(num_labels)[y_input].float().to(device)

    for outer_step in range(binary_search_steps):
        # print("o_bestl2={} confidence={}".format(o_bestl2, confidence))
        # 把原始图像转换成图像数据和扰动的形态
        timg = Variable(torch.arctanh((x_input.data - boxplus) / boxmul * 0.999999).to(device).float())
        modifier = Variable(torch.zeros_like(timg).to(device).float())
        # 图像数据的扰动量梯度可以获取
        modifier.requires_grad = True
        # 定义优化器 仅优化modifier
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)
        for iteration in range(1, max_iterations + 1):
            optimizer.zero_grad()
            # 定义新输入
            newimg = torch.tanh(modifier + timg) * boxmul + boxplus
            output = net(newimg)
            loss2 = torch.dist(newimg, (torch.tanh(timg) * boxmul + boxplus), p=2)
            """
            # compute the probability of the label class versus the maximum other
                real = tf.reduce_sum((tlab)*output,1)
                # 论文中的开源实现 other = tf.reduce_max((1-tlab)*output - (tlab*10000),1)
                other = tf.reduce_max((1-tlab)*output)
                loss1 = tf.maximum(0.0, other-real+k)
                loss1 = tf.reduce_sum(const*loss1)
            """
            real = torch.max(output * tlb)
            other = torch.max((1 - tlb) * output)
            if target_label:
                loss1 = other - real + k
            else:
                loss1 = -other + real + k
            loss1 = torch.clamp(loss1, min=0)
            loss1 = confidence * loss1
            loss = loss1 + loss2
            loss.backward(retain_graph=True)
            optimizer.step()
            l2 = loss2
            sc = output.data
            # print out the losses every 10%
            pred = sc.argmax()
            # if iteration % (max_iterations // 10) == 0:
                # print("iteration={} loss={} loss1={} loss2={} pred={}".format(iteration, loss, loss1, loss2, pred))

            if target_label:

                if (l2 < o_bestl2) and (sc.argmax(axis=1) == target_label):
                    # print("attack success l2={} target_label={}".format(l2, target_label))
                    o_bestl2 = l2
                    o_bestscore = sc.argmax(axis=1)
                    o_bestattack = newimg.data
            else:
                if (l2 < o_bestl2) and (pred != y_input):
                    # print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                    # print("attack success l2={} label={}".format(l2,pred))
                    # print("attack success l2={} label={}".format(l2, pred))
                    o_bestl2 = l2
                    o_bestscore = pred
                    o_bestattack = newimg.data
        confidence_old = -1
        if target_label:
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
        else:
            if (o_bestscore != y_input) and (o_bestscore != -1):
                # 攻击成功 减小c
                upper_bound = min(upper_bound, confidence)
                if upper_bound < 1e9:
                    confidence_old = confidence
                    confidence = (lower_bound + upper_bound) / 2
            else:
                lower_bound = max(lower_bound, confidence)
                confidence_old = confidence
                if upper_bound < 1e9:
                    confidence = (lower_bound + upper_bound) / 2
                else:
                    confidence *= 10
        # print("outer_step={} confidence {}->{}".format(outer_step, confidence_old, confidence))
    return o_bestattack, o_bestscore






    # original_image_path = "./pics/cropped_panda.jpg"
    # ToTensor_transform = transforms.Compose([transforms.Resize((224, 224)),
    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #
    # original_image_tensor = misc.image_preprocessing(original_image_path, ToTensor_transform, is_unsqueeze=True)
    #
    # adv_image_tensor = Variable(original_image_tensor.clone().detach().to(device))
    # adv_image_tensor.requires_grad = True

    # test_dataloader = misc.load_data_mnist_test(16)
    # for X, y in test_dataloader:
    #     X, y = X.to(device), y.to(device)
    #     y_hat_ori = misc.argmax(model(X), 1)
    #     print(y_hat_ori)
    #     print(y)
    #     target_label = (y_hat_ori + 1) % 10
    #     X_adv, y_hat_adv = fgsm(model, X, device=device, target_label=target_label)
    #     print(y_hat_adv)
    #     break
    # model.load_state_dict(torch.load('./models/classifiers/classifier_mnist.pth'))
    # model.eval()
    # model.to(device)
    # test_data_set = misc.load_data_mnist_test(100)
    # for X_iter, y_iter in test_data_set:
    #     for i in range(100):
    #         print(X_iter[i].shape)
    #         X_adv = fgsm(model, X_iter[i], e=0.01, device=device, target_label=1)
    #         print(X_adv.shape)
    #         break
    #     break

    # output = torch.softmax(output, 1, dtype=torch.float32)
    # score, label = torch.max(output, 1)
    # score = score / torch.sum(output, dtype=torch.float32)
    # print("score={},label={}".format(score.data, label.data))
    # # adv_image_tensor = deepfool_target(model, original_image_tensor, target_label=288)
    # adv_image_tensor = fgsm(model, original_image_tensor, e=0.001, device=misc.try_gpu())
    # # adv_image_tensor = deepfool_untarget(model, original_image_tensor, original_label=label)
    # # adv_image_tensor = cw(model, original_image_tensor, target_label=288)
    # misc.transform_to_img_from_dataset(adv_image_tensor.cpu().clone().detach(), ToTensor_transform, "pics/test111_adv.png")
    # misc.show_images_diff(original_image_tensor, adv_image_tensor.cpu().clone().detach(), ToTensor_transform, label, 288)
