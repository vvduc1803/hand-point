from torch.nn import functional as F
from config.conf import cfg


def hand_loss(pre_hand, targets):
    pred_mano_results, gt_mano_results, preds_joints_img = pre_hand

    loss = {}
    loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'],
                                                            gt_mano_results['verts3d'])
    loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'],
                                                              gt_mano_results['joints3d'])
    loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'],
                                                          gt_mano_results['mano_pose'])
    loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'],
                                                            gt_mano_results['mano_shape'])
    loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])

    return loss

def idisc_loss(pred_point, gt_point):
    bs = pred_point.shape[0]
    loss = F.mse_loss(pred_point.reshape((bs, -1, 3)), gt_point)
    return loss





