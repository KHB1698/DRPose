import torch
import logging
import numpy as np
import hashlib
from torch.autograd import Variable
import os
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # 如果target为Nan则改为0
    if torch.isnan(target).any():
        target[torch.isnan(target)] = 0
    # assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum


def test_calculation_diffu(predicted, target, action, error_sum, data_type, subject,is_train,time_step=-1,mode='p_avg',reproject_2d=None,input_2D=None):
    
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum,mode=mode,reproject_2d=reproject_2d,input_2D=input_2D)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum,mode=mode,reproject_2d=reproject_2d,input_2D=input_2D)
    

    return error_sum

def mpjpe_p_best(predicted, target):
    b = predicted.shape[0]
    t = predicted.shape[1]
    h = predicted.shape[2]
    # print(predicted.shape)
    # print(target.shape)
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)
    from einops import rearrange
    # errors = rearrange(errors, 'b t h f n  -> t h b f n', ).reshape(t, h, -1)
    errors = rearrange(errors, 'b t h f n  -> t b h f n', ).reshape(t, b, h, -1)
    errors = torch.mean(errors, dim=-1, keepdim=False)
    min_errors = torch.min(errors, dim=-1, keepdim=False).values
    return min_errors

def mpjpe_p_avg(predicted, target):
    b = predicted.shape[0]
    t = predicted.shape[1]
    h = predicted.shape[2]
    mean_pose = torch.mean(predicted, dim=2, keepdim=False)
    target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
    errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
    from einops import rearrange
    errors = rearrange(errors, 'b t f n  -> t b f n', )
    errors = errors.reshape(t,b, -1)
    errors = torch.mean(errors, dim=-1, keepdim=False)
    return errors

def mpjpe_j_best(predicted, target):
    b = predicted.shape[0]
    t = predicted.shape[1]
    h = predicted.shape[2]
    # print(predicted.shape)
    # print(target.shape)
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)
    from einops import rearrange
    #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
    errors = rearrange(errors, 'b t h f n  -> t h b f n', )
    min_errors = torch.min(errors, dim=1, keepdim=False).values
    min_errors = min_errors.reshape(t,b, -1)
    min_errors = torch.mean(min_errors, dim=-1, keepdim=False)
    return min_errors

def mpjpe_j_avg(predicted, target, reproj_2d, target_2d):
    b = predicted.shape[0]
    t = predicted.shape[1]
    h = predicted.shape[2]
    # print(predicted.shape)
    # print(target.shape)
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)  # b,t,h,f,n
    errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1)
    from einops import rearrange
    #errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
    select_ind = torch.min(errors_2d, dim=2, keepdim=True).indices# b,t,1,f,n
    errors_select = torch.gather(errors, 2, select_ind)# b,t,1,f,n
    errors_select = rearrange(errors_select, 'b t h f n  -> t h b f n', )
    errors_select = errors_select.reshape(t,b, -1)
    errors_select = torch.mean(errors_select, dim=-1, keepdim=False)
    return errors_select
    
def p_mpjpe_p_avg(predicted, target):

    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    predicted = torch.mean(predicted, dim=2, keepdim=False)
    target = target.unsqueeze(1).repeat(1, t_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    
    target = target.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
    predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
    errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    # from einops import rearrange
    # errors = rearrange(errors, 'b t f n  -> t b f n', )
    errors = errors.transpose(1, 0, 2, 3)
    errors = errors.reshape(t_sz,b_sz, -1)
    errors = np.mean(errors, axis=-1, keepdims=False)
    return errors

def p_mpjpe_p_best(predicted, target):

    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    # from einops import rearrange
    # # errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
    # errors = rearrange(errors, 'b t h f n  -> t h b f n', )
    # errors = errors.transpose(1, 2, 0, 3, 4).reshape(t_sz, h_sz, -1) # t, h, b, f, n
    errors = errors.transpose(1, 0, 2, 3, 4).reshape(t_sz, b_sz, h_sz, -1) # t, b, h, f, n
    errors = np.mean(errors, axis=-1, keepdims=False)
    min_errors = np.min(errors, axis=-1, keepdims=False)
    return min_errors

def p_mpjpe_j_best(predicted, target):
    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    # from einops import rearrange
    # # errors = rearrange(errors, 't b h f n  -> t h b f n', ).reshape(t, h, -1)
    # errors = rearrange(errors, 'b t h f n  -> t h b f n', )
    errors = errors.transpose(1, 2, 0, 3, 4) # t, h, b, f, n
    min_errors = np.min(errors, axis=1, keepdims=False)
    min_errors = min_errors.reshape(t_sz,b_sz, -1)
    min_errors = np.mean(min_errors, axis=-1, keepdims=False)
    return min_errors

def p_mpjpe_j_avg(predicted, target, reproj_2d, target_2d):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    #assert predicted.shape == target.shape

    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape

    target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    target_2d = target_2d.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    errors_2d = torch.norm(reproj_2d - target_2d, dim=len(target_2d.shape) - 1) # b, t, h, f, n
    selec_ind = torch.min(errors_2d, dim=2, keepdims=True).indices # b, t, 1, f, n


    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t


    target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
    errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    errors = torch.from_numpy(errors).cuda()
    errors_select = torch.gather(errors, 2, selec_ind) #b, t, 1, f, n
    from einops import rearrange
    errors_select = rearrange(errors_select, 'b t h f n  -> t h b f n', )
    errors_select = errors_select.reshape(t_sz,b_sz, -1)
    errors_select = torch.mean(errors_select, dim=-1, keepdim=False)
    #errors = errors.transpose(1, 2, 0, 3, 4)  # t, h, b, f, n
    errors_select = errors_select.cpu().numpy()

    return errors_select


def mpjpe_by_action_p1(predicted, target, action, action_error_sum, mode='p_avg',reproject_2d=None,input_2D=None):
    # assert predicted.shape == target.shape
    # num = predicted.size(0)
    # dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)
    
    num = target.shape[0]
    if mode == 'p_avg':
        dists = mpjpe_p_avg(predicted, target)
    elif mode == 'p_best':
        dists = mpjpe_p_best(predicted, target)
    elif mode == 'j_best':
        dists = mpjpe_j_best(predicted, target)
    elif mode == 'j_avg':
        dists = mpjpe_j_avg(predicted, target,reproject_2d,input_2D)
    
    # dist = dists[0]

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        # action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
        # 对dists的每个元素进行更新
        action_error_sum[action_name]['p1'].update(torch.mul(torch.mean(dists,dim=-1),num), num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            # action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
            action_error_sum[action_name]['p1'].update(dists[:,i], 1)
            
    return action_error_sum

def mpjpe_by_action_p2(predicted, target, action, action_error_sum,mode='p_avg',reproject_2d=None,input_2D=None):
    # assert predicted.shape == target.shape
    # num = predicted.size(0)
    # pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    # gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    # dist = p_mpjpe(pred, gt)
    
    num = target.shape[0]
    if mode == 'p_avg':
        dists = p_mpjpe_p_avg(predicted, target)
    elif mode == 'p_best':
        dists = p_mpjpe_p_best(predicted, target)
    elif mode == 'j_best':
        dists = p_mpjpe_j_best(predicted, target)
    elif mode == 'j_avg':
        dists = p_mpjpe_j_avg(predicted, target,reproject_2d,input_2D)
    
    
    # dist = dists[0]

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        # action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
        action_error_sum[action_name]['p2'].update(np.dot(np.mean(dists,axis=-1),num), num) #可以改为直接乘法
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            # action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
            action_error_sum[action_name]['p2'].update(dists[:,i], 1)
            
    return action_error_sum



def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY 
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1':AccumLoss(), 'p2':AccumLoss()} 
        for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train,mode='p_avg'):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train, mode)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train,mode):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss()}
    
    
    for action, value in action_error_sum.items():
        # if is_train == 0:
        #     print("{0:<12} ".format(action))
            
        #     logging.info("{0:<12} ".format(action))

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

    _,index = torch.min(mean_error_all['p1'].avg,dim=0)
    if is_train == 0:
        
        print("Mode:{}   Step:{}".format(mode,index.item()+1))
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))
        
        logging.info("Mode:{}   Step:{}".format(mode,index.item()+1))
        logging.info("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))
        
        for action, value in action_error_sum.items():
            mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
            mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
            print("{0:<12} {1:>6.2f} {2:>10.2f}".format(action, mean_error_each['p1'][index], mean_error_each['p2'][index]))
            logging.info("{0:<12} {1:>6.2f} {2:>10.2f}".format(action, mean_error_each['p1'][index], mean_error_each['p2'][index]))
        
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg[index], \
                mean_error_all['p2'].avg[index]))
        print("{0:=^12} {1:=^10} {2:=^8}".format("Step", "p#1 mm", "p#2 mm"))
        
        
        logging.info("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg[index], \
                mean_error_all['p2'].avg[index]))
        
        logging.info("{0:=^12} {1:=^10} {2:=^8}".format("Step", "p#1 mm", "p#2 mm"))
        
        for i in range(0, len(mean_error_all['p1'].avg)):
            print("{0:<12.0f} {1:>6.2f} {2:>10.2f}".format(i+1, mean_error_all['p1'].avg[i], \
                mean_error_all['p2'].avg[i]))
            
            logging.info("{0:<12.0f} {1:>6.2f} {2:>10.2f}".format(i+1, mean_error_all['p1'].avg[i], \
                mean_error_all['p2'].avg[i]))
        
        print("\n")
        
        logging.info("\n")
    
    return mean_error_all['p1'].avg[index], mean_error_all['p2'].avg[index]


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))

    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name
    

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))







