import os
import argparse
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from accelerate import Accelerator

from config.conf import cfg
from data.base import Trainer

from loss.losses import hand_loss, idisc_loss

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='continue_train', default=False)
    parser.add_argument("--gpu", type=str, dest="gpu_ids", default='0')
    parser.add_argument("--seed", default=42, type=int, help="Seed.")
    parser.add_argument("--batch", default=2, type=int)
    parser.add_argument("--num", default=0, type=int)
    parser.add_argument("--epoch", default=999, type=int)
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.batch, args.num, args.epoch, args.continue_train)

    # set seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    cudnn.benchmark = True

    accelerator = Accelerator(
        log_with="wandb",
        project_dir='/home/ana/Study/CVPR/handpoint'
    )

    # wandb
    accelerator.init_trackers(
        project_name="hand-point-pose-estimation",
        init_kwargs={"wandb": {"name": cfg.architecture}},
        config={
            "architecture": cfg.architecture,
            "train_dataset": cfg.trainset,
            "test_dataset": cfg.testset,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.end_epoch,
        }
    )

    # load train dataset
    trainer = Trainer()
    trainer._make_train_batch_generator()

    # load model, criterion, optimizer
    trainer._make_model('train')

    # initialize metric for save best model
    min_metrics = float('inf')
    # print(trainer.model.named_parameters())
    # for i in trainer.model.named_parameters():
    #     print(i)
    # input()

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        train_total_loss = 0.0
        test_total_loss = 0.0
        progress_bar = tqdm(trainer.train_batch_generator, colour="green")
        trainer.set_lr(epoch)
        losses = {}

        for train_itr, (inputs, targets, meta_info) in enumerate(progress_bar):
            inputs = inputs.to('cuda')
            targets = {k: v.to('cuda') for k, v in targets.items()}

            # forward
            trainer.optimizer.zero_grad()
            pre_point, hand_out = trainer.model(inputs, targets)

            loss_point = idisc_loss(pre_point, targets['joints_coord_cam'])
            loss_hand = hand_loss(hand_out, targets)

            # loss_hand = {k: loss_hand[k].mean() for k in loss_hand}

            # backward
            sum_loss = sum(loss_hand[k] for k in loss_hand)
            train_total_loss += sum_loss.item()
            train_total_loss += loss_point.item()

            if losses == {}:
                losses = {k: v.detach().item() for k, v in loss_hand.items()}
                losses['point'] = loss_point.detach().item()
            else:
                for k, v in loss_hand.items():
                    losses[k] += v.detach()
                losses['point'] += loss_point.detach()

            sum_loss = sum_loss + loss_point
            # loss_point.backward(retain_graph=True)
            sum_loss.backward()

            trainer.optimizer.step()
            # trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d train_itr %d/%d:' % (epoch, cfg.end_epoch, train_itr, trainer.train_itr_per_epoch),
                'lr: %g' % (trainer.get_lr())
            ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss_hand.items()]
            screen += ['%s: %.4f' % ('loss_point', loss_point)]
            trainer.train_logger.info(' '.join(screen))

            progress_bar.set_description(
                "Epoch %d/%d train_itr: %d/%d " % (epoch, cfg.end_epoch, train_itr, trainer.train_itr_per_epoch)
                + "%s: %.4f" % ("loss_mano_verts", loss_hand['mano_verts']) + ' '
                + "%s: %.4f" % ("loss_mano_joints", loss_hand['mano_joints']) + ' '
                + "%s: %.4f" % ("loss_mano_pose", loss_hand['mano_pose']) + ' '
                + "%s: %.4f" % ("loss_mano_shape", loss_hand['mano_shape']) + ' '
                + "%s: %.4f" % ("loss_joints_img", loss_hand['joints_img']) + ' '
                + "%s: %.4f" % ("loss_point", loss_point))

        trainer.save_model(
            {
                "epoch": epoch,
                "network": trainer.model.state_dict(),
                "optimizer": trainer.optimizer.state_dict()
            },
            epoch + 1,
            False,
            True
        )
        print("\nEpoch %d training result:" % epoch)
        print("loss_mano_verts : %.4f " % (losses['mano_verts'] / (train_itr + 1)))
        print("loss_mano_joints : %.4f " % (losses['mano_joints'] / (train_itr + 1)))
        print("loss_mano_pose : %.4f " % (losses['mano_pose'] / (train_itr + 1)))
        print("loss_mano_shape : %.4f " % (losses['mano_shape'] / (train_itr + 1)))
        print("loss_joints_img : %.4f " % (losses['joints_img'] / (train_itr + 1)))
        print("loss_point : %.4f " % (losses['point'] / (train_itr + 1)))

        if (epoch + 1) % 10 == 0 or epoch + 1 == cfg.end_epoch:
            trainer.save_model({
                'epoch': epoch,
                'network': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }, epoch + 1)
        print(f"%point: {1-torch.mean(trainer.model.rgb_depth.attention_module.weight.weight).item()}")
        # evaluation
        trainer.model.eval()
        trainer._make_test_batch_generator()
        test_losses = {}
        with torch.inference_mode():
            cur_sample_idx = 0
            progress_bar2 = tqdm(trainer.test_batch_generator, colour="green")
            for test_itr, (inputs, targets, meta_info) in enumerate(tqdm(progress_bar2)):
                inputs = inputs.to('cuda')
                targets = {k: v.to('cuda') for k, v in targets.items()}

                # forward
                pre_point, hand_out = trainer.model(inputs, targets)

                loss_point = idisc_loss(pre_point, targets['joints_coord_cam'])
                loss_hand = hand_loss(hand_out, targets)

                # loss_point = loss_point.mean()
                # loss = {k: loss_hand[k].mean() for k in loss_hand}

                # backward
                sum_loss = sum(loss_hand[k] for k in loss_hand)
                test_total_loss += sum_loss.item()
                test_total_loss += loss_point.item()

                if test_losses == {}:
                    test_losses = {k: v.detach().item() for k, v in loss_hand.items()}
                    test_losses['point'] = loss_point.detach().item()
                else:
                    for k, v in loss_hand.items():
                        test_losses[k] += v.detach()
                    test_losses['point'] += loss_point.detach()

                screen = [
                    'Epoch %d/%d test_itr %d/%d:' % (epoch, cfg.end_epoch, test_itr, trainer.test_itr_per_epoch),
                    'lr: %g' % (trainer.get_lr())
                ]
                screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k, v in loss_hand.items()]
                screen += ['%s: %.4f' % ('loss_point', loss_point)]
                trainer.test_logger.info(' '.join(screen))

                progress_bar.set_description(
                    "Epoch %d/%d test_itr: %d/%d " % (epoch, cfg.end_epoch, test_itr, trainer.test_itr_per_epoch)
                    + "%s: %.4f" % ("loss_mano_verts", loss_hand['mano_verts']) + ' '
                    + "%s: %.4f" % ("loss_mano_joints", loss_hand['mano_joints']) + ' '
                    + "%s: %.4f" % ("loss_mano_pose", loss_hand['mano_pose']) + ' '
                    + "%s: %.4f" % ("loss_mano_shape", loss_hand['mano_shape']) + ' '
                    + "%s: %.4f" % ("loss_joints_img", loss_hand['joints_img']) + ' '
                    + "%s: %.4f" % ("loss_point", loss_point))

                # save output
                out = {}
                out['joints_coord_cam'] = hand_out[0]['joints3d'].detach()
                out['mesh_coord_cam'] = hand_out[0]['verts3d'].detach()

                out = {k: v.cpu().numpy() for k, v in out.items()}
                for k, _ in out.items():
                    batch_size = out[k].shape[0]
                out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

                # evaluate
                trainer._evaluate(out, cur_sample_idx)
                cur_sample_idx += len(out)

                mpjpe, pa_mpjpe = trainer._eval_result()

                progress_bar2.set_description(
                    "Epoch %d/%d test_itr: %d/%d " % (epoch, cfg.end_epoch, test_itr, trainer.test_itr_per_epoch)
                    + "%s: %.4f" % ("mpjpe", mpjpe) + ' '
                    + "%s: %.4f" % ("pa_mpjpe", pa_mpjpe) + ' '
                    + "%s: %.4f" % ("loss_mano_verts", loss_hand['mano_verts']) + ' '
                    + "%s: %.4f" % ("loss_mano_joints", loss_hand['mano_joints']) + ' '
                    + "%s: %.4f" % ("loss_mano_pose", loss_hand['mano_pose']) + ' '
                    + "%s: %.4f" % ("loss_mano_shape", loss_hand['mano_shape']) + ' '
                    + "%s: %.4f" % ("loss_joints_img", loss_hand['joints_img']) + ' '
                    + "%s: %.4f" % ("loss_point", loss_point))

            mean_mpjpe, mean_pampjpe = trainer._mean_eval_result()
            print("\nEpoch %d evaluation result:" % epoch)
            print("loss_mano_verts : %.4f " % (test_losses['mano_verts'] / (test_itr + 1)))
            print("loss_mano_joints : %.4f " % (test_losses['mano_joints'] / (test_itr + 1)))
            print("loss_mano_pose : %.4f " % (test_losses['mano_pose'] / (test_itr + 1)))
            print("loss_mano_shape : %.4f " % (test_losses['mano_shape'] / (test_itr + 1)))
            print("loss_joints_img : %.4f " % (test_losses['joints_img'] / (test_itr + 1)))
            print("loss_point : %.4f " % (test_losses['point'] / (test_itr + 1)))
            print("MPJPE : %.2f mm" % mean_mpjpe)
            print("PA MPJPE : %.2f mm" % mean_pampjpe)

        trainer.model.train()

        # save best loss model
        if min_metrics > (mean_mpjpe+mean_pampjpe):
            min_metrics = mean_mpjpe+mean_pampjpe
            trainer.save_model(
                {
                    "epoch": epoch,
                    "network": trainer.model.state_dict(),
                    "optimizer": trainer.optimizer.state_dict()
                },
                epoch + 1,
                True
            )

        log_result = {
            "epoch": epoch + 1,
            "train_total_loss": train_total_loss / (train_itr+1),
            "test_total_loss": test_total_loss / (test_itr+1),
        }

        metric_log = {
            "epoch": epoch + 1,
            "mpjpe": mean_mpjpe,
            "pa_mpjpe": mean_pampjpe
        }
        for k, v in losses.items():
            log_result["train_loss_" + k] = v / (train_itr + 1)

        for k, v in losses.items():
            log_result["test_loss_" + k] = v / (test_itr + 1)


        accelerator.log(log_result, step=epoch)
        accelerator.log(metric_log, step=epoch)

    accelerator.end_training()


if __name__ == "__main__":
    main()
