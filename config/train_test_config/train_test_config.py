import numpy as np
from config.eval_config.eval import evaluate, evaluate_multi
import torch
import os
from PIL import Image
import torchio as tio

def print_train_loss_SPC(train_loss_sup_2D, train_loss_sup_3D, train_loss_unsup, train_loss, num_batches, print_num, print_num_half):
    train_epoch_loss_sup2D = train_loss_sup_2D / num_batches['train_sup']
    train_epoch_loss_sup3D = train_loss_sup_3D / num_batches['train_sup']
    train_epoch_loss_unsup = train_loss_unsup / num_batches['train_sup']
    train_epoch_loss = train_loss / num_batches['train_sup']
    print('-' * print_num)
    print('| Train Sup2D Loss: {:.4f}'.format(train_epoch_loss_sup2D).ljust(print_num_half, ' '), '| Train Sup3D Loss: {:.4f}'.format(train_epoch_loss_sup3D).ljust(print_num_half, ' '), '|')
    print('| Train Unsup Loss: {:.4f}'.format(train_epoch_loss_unsup).ljust(print_num_half, ' '), '| Train Total Loss: {:.4f}'.format(train_epoch_loss).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return train_epoch_loss_sup2D, train_epoch_loss_sup3D, train_epoch_loss_unsup, train_epoch_loss

def print_val_loss_SPC(val_loss_2D, val_loss_3D, num_batches, print_num, print_num_half):
    val_epoch_loss_2D = val_loss_2D / num_batches['val']
    val_epoch_loss_3D = val_loss_3D / num_batches['val']
    print('-' * print_num)
    print('| Val 2D Loss: {:.4f}'.format(val_epoch_loss_2D).ljust(print_num_half, ' '), '| Val 3D Loss: {:.4f}'.format(val_epoch_loss_3D).ljust(print_num_half, ' '), '|')
    print('-' * print_num)
    return val_epoch_loss_2D, val_epoch_loss_3D


def print_train_eval_SPC(num_classes, score_list_train2D, score_list_train3D, mask_list_train, print_num):

    if num_classes == 2:
        eval_list2D = evaluate(score_list_train2D, mask_list_train)
        eval_list3D = evaluate(score_list_train3D, mask_list_train)
        print('| Train Thr 2D: {:.4f}'.format(eval_list2D[0]).ljust(print_num, ' '), '| Train Thr 3D: {:.4f}'.format(eval_list3D[0]).ljust(print_num, ' '), '|')
        print('| Train  Jc 2D: {:.4f}'.format(eval_list2D[1]).ljust(print_num, ' '), '| Train  Jc 3D: {:.4f}'.format(eval_list3D[1]).ljust(print_num, ' '), '|')
        print('| Train  Dc 2D: {:.4f}'.format(eval_list2D[2]).ljust(print_num, ' '), '| Train  Dc 3D: {:.4f}'.format(eval_list3D[2]).ljust(print_num, ' '), '|')
        train_m_jc2D = eval_list2D[1]
        train_m_jc3D = eval_list3D[1]
    else:
        eval_list2D = evaluate_multi(score_list_train2D, mask_list_train)
        eval_list3D = evaluate_multi(score_list_train3D, mask_list_train)
        np.set_printoptions(precision=4, suppress=True)
        print('| Train  Jc 2D: {}'.format(eval_list2D[0]).ljust(print_num, ' '), '| Train  Jc 3D: {}'.format(eval_list3D[0]).ljust(print_num, ' '), '|')
        print('| Train  Dc 2D: {}'.format(eval_list2D[2]).ljust(print_num, ' '), '| Train  Dc 3D: {}'.format(eval_list3D[2]).ljust(print_num, ' '), '|')
        print('| Train mJc 2D: {:.4f}'.format(eval_list2D[1]).ljust(print_num, ' '), '| Train mJc 3D: {:.4f}'.format(eval_list3D[1]).ljust(print_num, ' '), '|')
        print('| Train mDc 2D: {:.4f}'.format(eval_list2D[3]).ljust(print_num, ' '), '| Train mDc 3D: {:.4f}'.format(eval_list3D[3]).ljust(print_num, ' '), '|')
        train_m_jc2D = eval_list2D[1]
        train_m_jc3D = eval_list3D[1]

    return eval_list2D, eval_list3D, train_m_jc2D, train_m_jc3D


def print_val_eval(num_classes, score_list_val1, score_list_val2, mask_list_val, print_num):
    if num_classes == 2:
        eval_list1 = evaluate(score_list_val1, mask_list_val)
        eval_list2 = evaluate(score_list_val2, mask_list_val)
        print('| Val Thr 2D: {:.4f}'.format(eval_list1[0]).ljust(print_num, ' '), '| Val Thr 3D: {:.4f}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Jc 2D: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val  Jc 3D: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val  Dc 2D: {:.4f}'.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 3D: {:.4f}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    else:
        eval_list1 = evaluate_multi(score_list_val1, mask_list_val)
        eval_list2 = evaluate_multi(score_list_val2, mask_list_val)
        np.set_printoptions(precision=4, suppress=True)
        print('| Val  Jc 2D: {}  '.format(eval_list1[0]).ljust(print_num, ' '), '| Val  Jc 3D: {}'.format(eval_list2[0]).ljust(print_num, ' '), '|')
        print('| Val  Dc 2D: {}  '.format(eval_list1[2]).ljust(print_num, ' '), '| Val  Dc 3D: {}'.format(eval_list2[2]).ljust(print_num, ' '), '|')
        print('| Val mJc 2D: {:.4f}'.format(eval_list1[1]).ljust(print_num, ' '), '| Val mJc 3D: {:.4f}'.format(eval_list2[1]).ljust(print_num, ' '), '|')
        print('| Val mDc 2D: {:.4f}'.format(eval_list1[3]).ljust(print_num, ' '), '| Val mDc 3D: {:.4f}'.format(eval_list2[3]).ljust(print_num, ' '), '|')
        val_m_jc1 = eval_list1[1]
        val_m_jc2 = eval_list2[1]
    return eval_list1, eval_list2, val_m_jc1, val_m_jc2

def save_val_best_sup_3d(num_classes, best_list, model, score_list_val, mask_list_val, eval_list, path_trained_model, path_seg_results, path_mask_results, model_name, format):

    if num_classes == 2:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

    else:
        if best_list[1] < eval_list[1]:
            best_list = eval_list

            torch.save(model.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format(model_name, best_list[1])))

    return best_list

def save_val_best_3d(best_model, best_list, best_result, model1, model2, eval_list_1, eval_list_2, path_trained_model):

    if eval_list_1[1] < eval_list_2[1]:
        if best_list[1] < eval_list_2[1]:

            best_model = model2
            best_list = eval_list_2
            best_result = 'Result2'
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    else:
        if best_list[1] < eval_list_1[1]:

            best_model = model1
            best_list = eval_list_1
            best_result = 'Result1'
        else:
            best_model = best_model
            best_list = best_list
            best_result = best_result

    torch.save(model1.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result1', eval_list_1[1])))
    torch.save(model2.state_dict(), os.path.join(path_trained_model, 'best_{}_Jc_{:.4f}.pth'.format('result2', eval_list_2[1])))

    return best_list, best_model, best_result


def print_best(num_classes, best_val_list, best_model, best_result, path_trained_model, print_num):
    if num_classes == 2:

        torch.save(best_model.state_dict(), os.path.join(path_trained_model, 'best_Jc_{:.4f}.pth'.format(best_val_list[1])))

        print('| Best  Result: {}'.format(best_result).ljust(print_num, ' '), '|')
        print('| Best Val Thr: {:.4f}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {:.4f}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
    else:

        torch.save(best_model.state_dict(), os.path.join(path_trained_model, 'best_Jc_{:.4f}.pth'.format(best_val_list[1])))

        np.set_printoptions(precision=4, suppress=True)
        print('| Best  Result: {}'.format(best_result).ljust(print_num, ' '), '|')
        print('| Best Val  Jc: {}'.format(best_val_list[0]).ljust(print_num, ' '), '|')
        print('| Best Val  Dc: {}'.format(best_val_list[2]).ljust(print_num, ' '), '|')
        print('| Best Val mJc: {:.4f}'.format(best_val_list[1]).ljust(print_num, ' '), '|')
        print('| Best Val mDc: {:.4f}'.format(best_val_list[3]).ljust(print_num, ' '), '|')

def save_test_3d(num_classes, score_test, name_test, threshold, path_seg_results, affine):

    if num_classes == 2:
        score_list_test = torch.softmax(score_test, dim=0)
        pred_results = score_list_test[1, ...].cpu()
        pred_results[pred_results > threshold] = 1
        pred_results[pred_results <= threshold] = 0

        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))

    else:
        pred_results = torch.max(score_test, 0)[1]
        pred_results = pred_results.cpu()
        pred_results = pred_results.type(torch.uint8)

        output_image = tio.ScalarImage(tensor=pred_results.unsqueeze(0), affine=affine)
        output_image.save(os.path.join(path_seg_results, name_test))



