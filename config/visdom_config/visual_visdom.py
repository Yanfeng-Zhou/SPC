from visdom import Visdom
import os

def visdom_initialization_SPC(env, port):
    visdom = Visdom(env=env, port=port)
    visdom.line([[0., 0., 0., 0.]], [0.], win='train_loss', opts=dict(title='Train Loss', xlabel='Epoch', ylabel='Train Loss', legend=['Total', '2D', '3D', 'Unsup'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='train_jc', opts=dict(title='Train Jc', xlabel='Epoch', ylabel='Train Jc', legend=['Jc 2D', 'Jc 3D'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_loss', opts=dict(title='Val Loss', xlabel='Epoch', ylabel='Val Loss', legend=['2D', '3D'], width=550, height=350))
    visdom.line([[0., 0.]], [0.], win='val_jc', opts=dict(title='Val Jc', xlabel='Epoch', ylabel='Val Jc', legend=['Jc 2D', 'Jc 3D'], width=550, height=350))
    return visdom

def visualization_SPC(vis, epoch, train_loss, train_loss_sup2D, train_loss_sup3D, train_loss_unsup, train_m_jc1, train_m_jc2, val_loss_sup2D, val_loss_sup3D, val_m_jc1, val_m_jc2):
    vis.line([[train_loss, train_loss_sup2D, train_loss_sup3D, train_loss_unsup]], [epoch], win='train_loss', update='append')
    vis.line([[train_m_jc1, train_m_jc2]], [epoch], win='train_jc', update='append')
    vis.line([[val_loss_sup2D, val_loss_sup3D]], [epoch], win='val_loss', update='append')
    vis.line([[val_m_jc1, val_m_jc2]], [epoch], win='val_jc', update='append')